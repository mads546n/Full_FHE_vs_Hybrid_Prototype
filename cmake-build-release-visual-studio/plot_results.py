import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

CSV_PATH = Path("results.csv")

def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Could not find {CSV_PATH.resolve()}")

    df = pd.read_csv(CSV_PATH)

    # Basic sanity checks (helps catch CSV mismatch early)
    required_cols = {
        "patient_id", "mode", "payload_type", "dim_used",
        "e2e_ms", "ct_in_bytes", "ct_out_bytes",
        "pk_bytes", "rlk_bytes", "gk_bytes",
        "rel_err"
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns in CSV: {sorted(missing)}")

    # Derived columns
    df["per_inference_bytes"] = df["ct_in_bytes"] + df["ct_out_bytes"]
    df["per_inference_kb"] = df["per_inference_bytes"] / 1024.0

    df["one_time_keys_bytes"] = df["pk_bytes"] + df["rlk_bytes"] + df["gk_bytes"]
    df["one_time_keys_mb"] = df["one_time_keys_bytes"] / (1024.0 * 1024.0)

    # Group by mode + payload_type + dim_used (averaging over runs/patients)
    agg = (
        df.groupby(["mode", "payload_type", "dim_used"], as_index=False)
        .agg(
            n=("e2e_ms", "count"),
            e2e_mean=("e2e_ms", "mean"),
            e2e_std=("e2e_ms", "std"),
            bw_mean_kb=("per_inference_kb", "mean"),
            bw_std_kb=("per_inference_kb", "std"),
            relerr_mean=("rel_err", "mean"),
            relerr_std=("rel_err", "std"),
            keys_mean_mb=("one_time_keys_mb", "mean"),
        )
        .sort_values(["mode", "payload_type", "dim_used"])
    )

    # Helper label
    def label_for(mode, payload):
        return f"{mode} ({payload})"

    # --- Plot 1: End-to-end latency vs dimension ---
    plt.figure()
    for (mode, payload), sub in agg.groupby(["mode", "payload_type"]):
        sub = sub.sort_values("dim_used")
        plt.errorbar(
            sub["dim_used"], sub["e2e_mean"], yerr=sub["e2e_std"],
            marker="o", capsize=3, label=label_for(mode, payload)
        )

    plt.xlabel("Encrypted payload dimension (dim_used)")
    plt.ylabel("End-to-end latency per inference (ms)")
    plt.title("End-to-end latency vs encrypted payload size (mean ± std)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig("plot_latency_e2e.png", dpi=200)

    # --- Plot 2: Per-inference bandwidth vs dimension ---
    plt.figure()
    for (mode, payload), sub in agg.groupby(["mode", "payload_type"]):
        sub = sub.sort_values("dim_used")
        plt.errorbar(
            sub["dim_used"], sub["bw_mean_kb"], yerr=sub["bw_std_kb"],
            marker="o", capsize=3, label=label_for(mode, payload)
        )

    plt.xlabel("Encrypted payload dimension (dim_used)")
    plt.ylabel("Per-inference bandwidth (KB): ct_in + ct_out")
    plt.title("Per-inference bandwidth vs encrypted payload size (mean ± std)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig("plot_bandwidth.png", dpi=200)

    # --- Plot 3: Relative error vs dimension ---
    plt.figure()
    for (mode, payload), sub in agg.groupby(["mode", "payload_type"]):
        sub = sub.sort_values("dim_used")
        plt.errorbar(
            sub["dim_used"], sub["relerr_mean"], yerr=sub["relerr_std"],
            marker="o", capsize=3, label=label_for(mode, payload)
        )

    plt.xlabel("Encrypted payload dimension (dim_used)")
    plt.ylabel("Relative error (CKKS)")
    plt.title("Relative error vs encrypted payload size (mean ± std)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig("plot_rel_error.png", dpi=200)

    # --- Plot 4 (Optional but great for discussion): One-time key overhead vs per-inference traffic ---
    # Shows that key material (especially Galois keys) is large but amortized over many queries.
    plt.figure()
    # Use one representative keys_mean_mb per group (it won't change across dim_used for a given param set)
    key_points = (
        agg.groupby(["mode", "payload_type"], as_index=False)
        .agg(keys_mb=("keys_mean_mb", "mean"),
             bw_kb=("bw_mean_kb", "mean"))
    )
    # Scatter (no fixed colors set)
    plt.scatter(key_points["bw_kb"], key_points["keys_mb"])
    for _, row in key_points.iterrows():
        plt.annotate(
            label_for(row["mode"], row["payload_type"]),
            (row["bw_kb"], row["keys_mb"]),
            textcoords="offset points", xytext=(5, 5)
        )

    plt.xlabel("Average per-inference bandwidth (KB)")
    plt.ylabel("One-time key material size (MB): pk + rlk + gk")
    plt.title("One-time key overhead vs per-inference traffic (amortization view)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig("plot_keys_vs_bandwidth.png", dpi=200)

    # --- Summary table for your report ---
    summary = agg[[
        "mode", "payload_type", "dim_used", "n",
        "e2e_mean", "bw_mean_kb", "relerr_mean", "keys_mean_mb"
    ]].copy()

    summary.to_csv("summary_table.csv", index=False)

    print("Saved:")
    print("  plot_latency_e2e.png")
    print("  plot_bandwidth.png")
    print("  plot_rel_error.png")
    print("  plot_keys_vs_bandwidth.png")
    print("  summary_table.csv")


if __name__ == "__main__":
    main()
