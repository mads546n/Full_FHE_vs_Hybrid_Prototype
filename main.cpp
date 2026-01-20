// main.cpp — Prototype 3 (one-shot loop): Full FHE vs Hybrid (feature boundary)
// Adds: output ciphertext size, end-to-end latency, plaintext baseline + CKKS error,
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

static std::vector<double> make_raw_window(size_t n, uint32_t seed = 123) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> noise(0.0, 1.0);

    std::vector<double> x(n);
    double base = 75.0;          // e.g., HR baseline
    double trend_per_step = 0.02;
    for (size_t i = 0; i < n; i++) {
        x[i] = base + trend_per_step * static_cast<double>(i) + noise(rng);
    }
    return x;
}

// Feature extractor for Hybrid:
// base: [mean, variance, min, max, slope]
// extended: adds [median, iqr, zcr-ish, energy] until reaching target_k (simple, deterministic)
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
    for (size_t i = 0; i < n; i++) {
        sum_ty += static_cast<double>(i) * x[i];
    }

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

    // Additional robust/simple stats (deterministic)
    std::vector<double> sorted = x;
    std::sort(sorted.begin(), sorted.end());

    auto percentile = [&](double p) -> double {
        if (sorted.empty()) return 0.0;
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

    // If still short, pad with deterministic combinations (keeps reproducible)
    while (feats.size() < target_k) {
        feats.push_back(mean + 0.001 * static_cast<double>(feats.size()));
    }

    return feats;
}

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
        const SEALContext &context,
        CKKSEncoder &encoder,
        Evaluator &evaluator,
        const Ciphertext &ct_x,
        const std::vector<double> &w,
        double bias,
        const GaloisKeys &galois_keys
) {
    if (w.empty()) throw std::invalid_argument("weights empty");

    // Encode weights at ct_x.scale() and at the same parms_id if needed.
    Plaintext pt_w;
    encoder.encode(w, ct_x.scale(), pt_w);
    pt_w.parms_id() = ct_x.parms_id();

    Ciphertext ct = ct_x;
    evaluator.multiply_plain_inplace(ct, pt_w);

    // Sum all slots: rotate-and-add
    size_t n = w.size();
    size_t steps = 1;
    while (steps < n) steps <<= 1;

    for (size_t step = 1; step < steps; step <<= 1) {
        Ciphertext rotated;
        evaluator.rotate_vector(ct, static_cast<int>(step), galois_keys, rotated);
        evaluator.add_inplace(ct, rotated);
    }

    // Add bias (plaintext) matched to ct parms/scale
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

    // For logging
    size_t poly_modulus_degree = 0;
    std::vector<int> coeff_modulus_bits;
};

static Setup setup_ckks(size_t poly_modulus_degree, const std::vector<int> &coeff_modulus_bits, double scale, Timings &t) {
    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, coeff_modulus_bits));

    SEALContext context(parms, /*expand_mod_chain*/ true, sec_level_type::tc128);

    // Validate parameters (helpful for reporting)
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
        const std::string &mode_label,
        size_t N_raw,
        size_t k_feat,
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
            s.context,
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
        std::cout << "\n[" << mode_label << "] score_fhe=" << std::setprecision(10) << t.fhe_score
                  << " score_plain=" << t.plaintext_score
                  << " abs_err=" << t.abs_error
                  << " rel_err=" << t.rel_error << "\n";
        std::cout << "[" << mode_label << "] enc_ms=" << t.encode_encrypt_ms
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
        // ---- CKKS parameter presets ----
        struct ParamSet {
            size_t poly_modulus_degree;
            std::vector<int> coeff_modulus_bits;
            double scale;
            std::string name;
        };

        std::vector<ParamSet> param_sets = {
                {8192,  {60, 40, 40, 60}, std::pow(2.0, 40), "P1_8192_60-40-40-60_scale2^40"},
                // Optional heavier set (uncomment if you want a second curve)
                // {16384, {60, 45, 45, 45, 60}, std::pow(2.0, 45), "P2_16384_60-45-45-45-60_scale2^45"},
        };

        // ---- Experiment sweeps ----
        std::vector<size_t> raw_dims = {128, 256, 512, 1024};
        std::vector<size_t> feat_dims = {5, 8, 16, 32};
        int runs_per_point = 5; // increase for smoother stats

        // CSV output
        std::ofstream csv("results.csv");
        if (!csv) throw std::runtime_error("Failed to open results.csv for writing");

        csv << "mode,param_name,poly_modulus_degree,coeff_modulus_bits,scale,"
               "N_raw,k_feat,dim_used,"
               "enc_ms,eval_ms,dec_ms,e2e_ms,"
               "ct_in_bytes,ct_out_bytes,pk_bytes,rlk_bytes,gk_bytes,"
               "plain_score,fhe_score,abs_err,rel_err\n";

        std::cout << "Writing results to results.csv\n";

        for (const auto &ps : param_sets) {
            Timings setup_t;
            Setup s = setup_ckks(ps.poly_modulus_degree, ps.coeff_modulus_bits, ps.scale, setup_t);

            // Print one-time setup stats
            std::cout << "\n=== ParamSet: " << ps.name << " ===\n";
            std::cout << "poly_modulus_degree=" << ps.poly_modulus_degree
                      << " coeff_modulus_bits=[" << bits_to_string(ps.coeff_modulus_bits) << "]"
                      << " scale=" << ps.scale << "\n";
            std::cout << "keygen_ms=" << setup_t.keygen_ms
                      << " pk_bytes=" << setup_t.public_key_bytes
                      << " rlk_bytes=" << setup_t.relin_keys_bytes
                      << " gk_bytes=" << setup_t.galois_keys_bytes << "\n";

            // Seed changes per run for variety (still reproducible)
            uint32_t base_seed = 1000;

            for (size_t N_raw : raw_dims) {
                for (size_t k_feat : feat_dims) {
                    for (int r = 0; r < runs_per_point; r++) {
                        uint32_t seed = base_seed + static_cast<uint32_t>(r) + static_cast<uint32_t>(N_raw * 10 + k_feat);

                        // Generate raw window and features
                        auto raw = make_raw_window(N_raw, seed);
                        auto feats = extract_features(raw, k_feat);

                        // Weights per dimension (keep seeded so repeatable)
                        auto w_raw = make_weights(N_raw, 900 + seed);
                        auto w_feat = make_weights(k_feat, 901 + seed);
                        double b = make_bias();

                        // FULL FHE: encrypt raw window and score
                        {
                            auto t_full = run_one_shot(
                                    s, raw, w_raw, b,
                                    "FULL_FHE_RAW",
                                    N_raw, k_feat,
                                    setup_t,
                                    /*verbose*/ false
                            );

                            csv << "FULL_FHE_RAW" << ","
                                << ps.name << ","
                                << ps.poly_modulus_degree << ","
                                << "\"" << bits_to_string(ps.coeff_modulus_bits) << "\"" << ","
                                << ps.scale << ","
                                << N_raw << ","
                                << k_feat << ","
                                << N_raw << ","
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

                        // HYBRID: extract features (plaintext on client), encrypt features and score
                        {
                            auto t_hyb = run_one_shot(
                                    s, feats, w_feat, b,
                                    "HYBRID_FEATURES",
                                    N_raw, k_feat,
                                    setup_t,
                                    /*verbose*/ false
                            );

                            csv << "HYBRID_FEATURES" << ","
                                << ps.name << ","
                                << ps.poly_modulus_degree << ","
                                << "\"" << bits_to_string(ps.coeff_modulus_bits) << "\"" << ","
                                << ps.scale << ","
                                << N_raw << ","
                                << k_feat << ","
                                << k_feat << ","
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
        std::cout << "Done. CSV saved as results.csv\n";
        std::cout << "Next: plot e2e_ms vs dim_used for both modes, and ct_in/out_bytes vs dim_used.\n";
        return 0;

    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
