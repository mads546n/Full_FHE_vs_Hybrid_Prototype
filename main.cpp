// main.cpp — Prototype 3 (one-shot loop): Full FHE vs Hybrid (feature boundary)
// Adds: realistic healthcare IoT "patient record" payload, explicit what is encrypted,
// output ciphertext size, end-to-end latency, plaintext baseline + CKKS error,
// CSV logging, and parameter/input sweeps.

#include <seal/seal.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace seal;

struct Timings {
    // setup (one-time)
    double keygen_ms = 0.0;

    // per-inference
    double encode_encrypt_ms = 0.0;
    double server_eval_ms = 0.0;
    double decrypt_decode_ms = 0.0;
    double end_to_end_ms = 0.0;

    // bandwidth
    size_t ct_in_bytes = 0;   // client -> server
    size_t ct_out_bytes = 0;  // server -> client
    size_t galois_keys_bytes = 0;
    size_t relin_keys_bytes = 0;
    size_t public_key_bytes = 0;

    // correctness
    double plaintext_score = 0.0;
    double fhe_score = 0.0;
    double abs_error = 0.0;
    double rel_error = 0.0;
};

static double ms_since(const std::chrono::high_resolution_clock::time_point &t0) {
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

template <typename T>
static size_t serialized_size(const T &seal_obj) {
    std::stringstream ss;
    seal_obj.save(ss);
    return static_cast<size_t>(ss.tellp());
}

// ---------------------------
// Realistic healthcare IoT data model
// ---------------------------
struct PatientRecord {
    std::string pseudonym;   // plaintext ID tag, e.g. "P042"
    std::string name;        // realism only (NOT encrypted)
    int age;                 // can be encrypted
    int sex;                 // 0/1 (can be encrypted)
    double weight_kg;        // can be encrypted
    double height_cm;        // can be encrypted
    std::vector<double> signal; // physiological window (encrypted in FULL_FHE_RAW)
};

// A realistic-ish heart-rate time series: baseline depends on age + mild rhythm + drift + noise + event spike.
static std::vector<double> make_hr_series(size_t n, int age, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> noise(0.0, 0.8);

    double base = 72.0 + (age > 60 ? 5.0 : 0.0); // toy realism: older patients slightly higher baseline
    std::vector<double> x(n);

    for (size_t i = 0; i < n; i++) {
        double t = static_cast<double>(i);
        double rhythm = 2.0 * std::sin(2.0 * 3.1415926535 * t / 60.0); // periodic variation
        double drift = 0.01 * t;                                       // slow drift
        double spike = (i == n / 2) ? 12.0 : 0.0;                      // simulated event

        x[i] = base + rhythm + drift + spike + noise(rng);
    }
    return x;
}

static PatientRecord make_patient(size_t n_raw, uint32_t seed = 1234) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> age_dist(20, 85);
    std::uniform_int_distribution<int> sex_dist(0, 1);
    std::uniform_real_distribution<double> w_dist(50.0, 110.0);
    std::uniform_real_distribution<double> h_dist(150.0, 200.0);

    int age = age_dist(rng);
    int sex = sex_dist(rng);
    double weight = w_dist(rng);
    double height = h_dist(rng);

    std::vector<std::string> names = {"Anna", "Peter", "Fatima", "Jonas", "Sara", "Omar", "Maja", "Noah"};
    std::uniform_int_distribution<int> name_dist(0, static_cast<int>(names.size() - 1));

    PatientRecord p;
    p.pseudonym = "P" + std::to_string((seed % 900) + 100);
    p.name = names[name_dist(rng)];
    p.age = age;
    p.sex = sex;
    p.weight_kg = weight;
    p.height_cm = height;
    p.signal = make_hr_series(n_raw, age, seed + 77);
    return p;
}

// ---------------------------
// Feature extractor (Hybrid)
// base: [mean, variance, min, max, slope]
// then adds [median, iqr, zcr-ish, energy] until reaching target_k
// ---------------------------
static std::vector<double> extract_features(const std::vector<double> &x, size_t target_k = 5) {
    if (x.empty()) throw std::invalid_argument("empty input");
    const size_t n = x.size();

    double mean = std::accumulate(x.begin(), x.end(), 0.0) / static_cast<double>(n);

    double var = 0.0;
    for (double v : x) {
        double d = v - mean;
        var += d * d;
    }
    var /= static_cast<double>(n);

    auto [mn_it, mx_it] = std::minmax_element(x.begin(), x.end());
    double mn = *mn_it;
    double mx = *mx_it;

    // slope via least squares to y = a + b t, with t = 0..n-1
    double sum_t = (n - 1) * static_cast<double>(n) / 2.0;
    double sum_t2 = (n - 1) * static_cast<double>(n) * (2.0 * n - 1) / 6.0;
    double sum_y = std::accumulate(x.begin(), x.end(), 0.0);

    double sum_ty = 0.0;
    for (size_t i = 0; i < n; i++) sum_ty += static_cast<double>(i) * x[i];

    double denom = static_cast<double>(n) * sum_t2 - sum_t * sum_t;
    double slope = 0.0;
    if (std::fabs(denom) > 1e-12) {
        slope = (static_cast<double>(n) * sum_ty - sum_t * sum_y) / denom;
    }

    std::vector<double> feats = {mean, var, mn, mx, slope};
    if (target_k <= feats.size()) {
        feats.resize(target_k);
        return feats;
    }

    std::vector<double> sorted = x;
    std::sort(sorted.begin(), sorted.end());

    auto percentile = [&](double p) -> double {
        double idx = p * (static_cast<double>(sorted.size() - 1));
        size_t i0 = static_cast<size_t>(std::floor(idx));
        size_t i1 = static_cast<size_t>(std::ceil(idx));
        double frac = idx - static_cast<double>(i0);
        return sorted[i0] * (1.0 - frac) + sorted[i1] * frac;
    };

    double median = percentile(0.5);
    double q1 = percentile(0.25);
    double q3 = percentile(0.75);
    double iqr = q3 - q1;

    // Zero-crossing-ish rate around mean
    double zcr = 0.0;
    for (size_t i = 1; i < n; i++) {
        double a = x[i - 1] - mean;
        double b = x[i] - mean;
        if ((a >= 0 && b < 0) || (a < 0 && b >= 0)) zcr += 1.0;
    }
    zcr /= static_cast<double>(std::max<size_t>(1, n - 1));

    // Energy (mean square)
    double energy = 0.0;
    for (double v : x) energy += v * v;
    energy /= static_cast<double>(n);

    std::vector<double> extras = {median, iqr, zcr, energy};
    for (double v : extras) {
        if (feats.size() >= target_k) break;
        feats.push_back(v);
    }

    while (feats.size() < target_k) feats.push_back(mean + 0.001 * static_cast<double>(feats.size()));
    return feats;
}

// ---------------------------
// Payload builders: define what we actually encrypt
// FULL: demographics + raw signal window
// HYBRID: demographics + extracted features
// ---------------------------
static std::vector<double> build_full_payload(const PatientRecord &p) {
    std::vector<double> v;
    v.reserve(4 + p.signal.size());

    v.push_back(static_cast<double>(p.age));
    v.push_back(static_cast<double>(p.sex));
    v.push_back(p.weight_kg);
    v.push_back(p.height_cm);

    v.insert(v.end(), p.signal.begin(), p.signal.end());
    return v;
}

static std::vector<double> build_hybrid_payload(const PatientRecord &p, size_t k_feat) {
    auto feats = extract_features(p.signal, k_feat);

    std::vector<double> v;
    v.reserve(4 + feats.size());

    v.push_back(static_cast<double>(p.age));
    v.push_back(static_cast<double>(p.sex));
    v.push_back(p.weight_kg);
    v.push_back(p.height_cm);

    v.insert(v.end(), feats.begin(), feats.end());
    return v;
}

// ---------------------------
// Model (toy): linear risk score = dot(x, w) + b
// ---------------------------
static std::vector<double> make_weights(size_t dim, uint32_t seed = 999) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-0.05, 0.05);
    std::vector<double> w(dim);
    for (size_t i = 0; i < dim; i++) w[i] = dist(rng);
    return w;
}

static double make_bias() { return 0.1; }

static double plaintext_dot(const std::vector<double> &x, const std::vector<double> &w, double b) {
    if (x.size() != w.size()) throw std::invalid_argument("plaintext_dot: size mismatch");
    double s = b;
    for (size_t i = 0; i < x.size(); i++) s += x[i] * w[i];
    return s;
}

// Encrypted dot product using CKKS packing (one ciphertext).
static Ciphertext encrypted_dot_product_one_ct(
        CKKSEncoder &encoder,
        Evaluator &evaluator,
        const Ciphertext &ct_x,
        const std::vector<double> &w,
        double bias,
        const GaloisKeys &galois_keys
) {
    if (w.empty()) throw std::invalid_argument("weights empty");

    Plaintext pt_w;
    encoder.encode(w, ct_x.scale(), pt_w);
    pt_w.parms_id() = ct_x.parms_id();

    Ciphertext ct = ct_x;
    evaluator.multiply_plain_inplace(ct, pt_w);

    // rotate-and-add reduction across slots
    size_t n = w.size();
    size_t steps = 1;
    while (steps < n) steps <<= 1;

    for (size_t step = 1; step < steps; step <<= 1) {
        Ciphertext rotated;
        evaluator.rotate_vector(ct, static_cast<int>(step), galois_keys, rotated);
        evaluator.add_inplace(ct, rotated);
    }

    Plaintext pt_b;
    encoder.encode(bias, ct.scale(), pt_b);
    pt_b.parms_id() = ct.parms_id();
    evaluator.add_plain_inplace(ct, pt_b);

    return ct;
}

static double decrypt_slot0(Decryptor &decryptor, CKKSEncoder &encoder, const Ciphertext &ct) {
    Plaintext pt;
    decryptor.decrypt(ct, pt);
    std::vector<double> vals;
    encoder.decode(pt, vals);
    return vals.empty() ? 0.0 : vals[0];
}

struct Setup {
    SEALContext context;
    double scale;

    PublicKey pk;
    SecretKey sk;
    RelinKeys rlk;
    GaloisKeys gk;

    std::unique_ptr<CKKSEncoder> encoder;
    std::unique_ptr<Encryptor> encryptor;
    std::unique_ptr<Decryptor> decryptor;
    std::unique_ptr<Evaluator> evaluator;

    size_t poly_modulus_degree = 0;
    std::vector<int> coeff_modulus_bits;
};

static Setup setup_ckks(size_t poly_modulus_degree, const std::vector<int> &coeff_modulus_bits, double scale, Timings &t) {
    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, coeff_modulus_bits));

    SEALContext context(parms, true, sec_level_type::tc128);
    if (!context.parameters_set()) {
        throw std::runtime_error("SEALContext: parameters_set() is false. Try different parms.");
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    KeyGenerator keygen(context);
    SecretKey sk = keygen.secret_key();

    PublicKey pk;
    keygen.create_public_key(pk);

    RelinKeys rlk;
    keygen.create_relin_keys(rlk);

    GaloisKeys gk;
    keygen.create_galois_keys(gk);

    t.keygen_ms = ms_since(t0);

    t.public_key_bytes = serialized_size(pk);
    t.relin_keys_bytes = serialized_size(rlk);
    t.galois_keys_bytes = serialized_size(gk);

    Setup s{
            context,
            scale,
            pk,
            sk,
            rlk,
            gk,
            std::make_unique<CKKSEncoder>(context),
            std::make_unique<Encryptor>(context, pk),
            std::make_unique<Decryptor>(context, sk),
            std::make_unique<Evaluator>(context),
            poly_modulus_degree,
            coeff_modulus_bits
    };
    return s;
}

static Timings run_one_shot(
        const Setup &s,
        const std::vector<double> &input_vec,
        const std::vector<double> &weights,
        double bias,
        const Timings &setup_sizes_for_logging,
        bool verbose = false
) {
    Timings t;
    t.public_key_bytes = setup_sizes_for_logging.public_key_bytes;
    t.relin_keys_bytes = setup_sizes_for_logging.relin_keys_bytes;
    t.galois_keys_bytes = setup_sizes_for_logging.galois_keys_bytes;

    // Plaintext baseline
    t.plaintext_score = plaintext_dot(input_vec, weights, bias);

    // Client: encode + encrypt
    auto t0 = std::chrono::high_resolution_clock::now();

    Plaintext pt_x;
    s.encoder->encode(input_vec, s.scale, pt_x);

    Ciphertext ct_x;
    s.encryptor->encrypt(pt_x, ct_x);

    t.encode_encrypt_ms = ms_since(t0);
    t.ct_in_bytes = serialized_size(ct_x);

    // Server: eval dot product
    auto t1 = std::chrono::high_resolution_clock::now();
    Ciphertext ct_score = encrypted_dot_product_one_ct(
            *s.encoder,
            *s.evaluator,
            ct_x,
            weights,
            bias,
            s.gk
    );
    t.server_eval_ms = ms_since(t1);
    t.ct_out_bytes = serialized_size(ct_score);

    // Client: decrypt + decode
    auto t2 = std::chrono::high_resolution_clock::now();
    t.fhe_score = decrypt_slot0(*s.decryptor, *s.encoder, ct_score);
    t.decrypt_decode_ms = ms_since(t2);

    // End-to-end (excluding keygen; includes compute only)
    t.end_to_end_ms = t.encode_encrypt_ms + t.server_eval_ms + t.decrypt_decode_ms;

    // Error metrics
    t.abs_error = std::fabs(t.fhe_score - t.plaintext_score);
    double denom = std::fabs(t.plaintext_score);
    t.rel_error = (denom > 1e-12) ? (t.abs_error / denom) : 0.0;

    if (verbose) {
        std::cout << "score_fhe=" << std::setprecision(10) << t.fhe_score
                  << " score_plain=" << t.plaintext_score
                  << " abs_err=" << t.abs_error
                  << " rel_err=" << t.rel_error << "\n";
        std::cout << "enc_ms=" << t.encode_encrypt_ms
                  << " eval_ms=" << t.server_eval_ms
                  << " dec_ms=" << t.decrypt_decode_ms
                  << " e2e_ms=" << t.end_to_end_ms
                  << " ct_in=" << t.ct_in_bytes
                  << " ct_out=" << t.ct_out_bytes << "\n";
    }

    return t;
}

static std::string bits_to_string(const std::vector<int> &bits) {
    std::ostringstream oss;
    for (size_t i = 0; i < bits.size(); i++) {
        oss << bits[i];
        if (i + 1 != bits.size()) oss << "-";
    }
    return oss.str();
}

int main() {
    try {
        struct ParamSet {
            size_t poly_modulus_degree;
            std::vector<int> coeff_modulus_bits;
            double scale;
            std::string name;
        };

        std::vector<ParamSet> param_sets = {
                {8192, {60, 40, 40, 60}, std::pow(2.0, 40), "P1_8192_60-40-40-60_scale2^40"},
                // {16384, {60, 45, 45, 45, 60}, std::pow(2.0, 45), "P2_16384_60-45-45-45-60_scale2^45"},
        };

        // Experiment sweeps
        std::vector<size_t> raw_dims = {128, 256, 512, 1024};
        std::vector<size_t> feat_dims = {5, 8, 16, 32};
        int runs_per_point = 5;

        std::ofstream csv("results.csv");
        if (!csv) throw std::runtime_error("Failed to open results.csv for writing");

        // Added patient_id and payload_type for clarity
        csv << "patient_id,mode,payload_type,param_name,poly_modulus_degree,coeff_modulus_bits,scale,"
               "N_raw,k_feat,dim_used,"
               "enc_ms,eval_ms,dec_ms,e2e_ms,"
               "ct_in_bytes,ct_out_bytes,pk_bytes,rlk_bytes,gk_bytes,"
               "plain_score,fhe_score,abs_err,rel_err\n";

        std::cout << "Writing results to results.csv\n";

        for (const auto &ps : param_sets) {
            Timings setup_t;
            Setup s = setup_ckks(ps.poly_modulus_degree, ps.coeff_modulus_bits, ps.scale, setup_t);

            std::cout << "\n=== ParamSet: " << ps.name << " ===\n";
            std::cout << "poly_modulus_degree=" << ps.poly_modulus_degree
                      << " coeff_modulus_bits=[" << bits_to_string(ps.coeff_modulus_bits) << "]"
                      << " scale=" << ps.scale << "\n";
            std::cout << "keygen_ms=" << setup_t.keygen_ms
                      << " pk_bytes=" << setup_t.public_key_bytes
                      << " rlk_bytes=" << setup_t.relin_keys_bytes
                      << " gk_bytes=" << setup_t.galois_keys_bytes << "\n";

            uint32_t base_seed = 1000;

            for (size_t N_raw : raw_dims) {
                for (size_t k_feat : feat_dims) {
                    for (int r = 0; r < runs_per_point; r++) {
                        uint32_t seed = base_seed + static_cast<uint32_t>(r)
                                        + static_cast<uint32_t>(N_raw * 10 + k_feat);

                        // Create realistic patient + signal
                        PatientRecord patient = make_patient(N_raw, seed);

                        // Build payloads (explicit what we encrypt)
                        auto full_payload = build_full_payload(patient);             // demographics + raw signal
                        auto hybrid_payload = build_hybrid_payload(patient, k_feat); // demographics + features

                        // Print explicit description occasionally (first run per point)
                        if (r == 0) {
                            std::cout << "\nPatient " << patient.pseudonym
                                      << " (name=" << patient.name
                                      << ", age=" << patient.age
                                      << ", sex=" << patient.sex
                                      << ")\n";
                            std::cout << "  FULL encrypts: [age,sex,weight,height] + raw_signal_window -> dim_used="
                                      << full_payload.size() << "\n";
                            std::cout << "  HYBRID encrypts: [age,sex,weight,height] + extracted_features -> dim_used="
                                      << hybrid_payload.size() << "\n";
                        }

                        // Weights must match payload sizes
                        auto w_full = make_weights(full_payload.size(), 900 + seed);
                        auto w_hyb  = make_weights(hybrid_payload.size(), 901 + seed);
                        double b = make_bias();

                        // FULL FHE
                        {
                            auto t_full = run_one_shot(s, full_payload, w_full, b, setup_t, false);

                            csv << patient.pseudonym << ","
                                << "FULL_FHE_RAW" << ","
                                << "\"demo+raw\"" << ","
                                << ps.name << ","
                                << ps.poly_modulus_degree << ","
                                << "\"" << bits_to_string(ps.coeff_modulus_bits) << "\"" << ","
                                << ps.scale << ","
                                << N_raw << ","
                                << k_feat << ","
                                << full_payload.size() << ","
                                << t_full.encode_encrypt_ms << ","
                                << t_full.server_eval_ms << ","
                                << t_full.decrypt_decode_ms << ","
                                << t_full.end_to_end_ms << ","
                                << t_full.ct_in_bytes << ","
                                << t_full.ct_out_bytes << ","
                                << setup_t.public_key_bytes << ","
                                << setup_t.relin_keys_bytes << ","
                                << setup_t.galois_keys_bytes << ","
                                << std::setprecision(17) << t_full.plaintext_score << ","
                                << std::setprecision(17) << t_full.fhe_score << ","
                                << t_full.abs_error << ","
                                << t_full.rel_error
                                << "\n";
                        }

                        // HYBRID
                        {
                            auto t_hyb = run_one_shot(s, hybrid_payload, w_hyb, b, setup_t, false);

                            csv << patient.pseudonym << ","
                                << "HYBRID_FEATURES" << ","
                                << "\"demo+feat\"" << ","
                                << ps.name << ","
                                << ps.poly_modulus_degree << ","
                                << "\"" << bits_to_string(ps.coeff_modulus_bits) << "\"" << ","
                                << ps.scale << ","
                                << N_raw << ","
                                << k_feat << ","
                                << hybrid_payload.size() << ","
                                << t_hyb.encode_encrypt_ms << ","
                                << t_hyb.server_eval_ms << ","
                                << t_hyb.decrypt_decode_ms << ","
                                << t_hyb.end_to_end_ms << ","
                                << t_hyb.ct_in_bytes << ","
                                << t_hyb.ct_out_bytes << ","
                                << setup_t.public_key_bytes << ","
                                << setup_t.relin_keys_bytes << ","
                                << setup_t.galois_keys_bytes << ","
                                << std::setprecision(17) << t_hyb.plaintext_score << ","
                                << std::setprecision(17) << t_hyb.fhe_score << ","
                                << t_hyb.abs_error << ","
                                << t_hyb.rel_error
                                << "\n";
                        }
                    } // runs
                } // k_feat
            } // N_raw
        } // param_sets

        csv.close();
        std::cout << "\nDone. CSV saved as results.csv\n";
        std::cout << "Next: plot e2e_ms vs dim_used for both modes, and ct_in/out_bytes vs dim_used.\n";
        return 0;

    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
