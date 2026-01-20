import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

CSV_PATH = Path("results.csv")

def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Could not find {CSV_PATH.resolve()}")

    df = pd.read_csv(CSV_PATH)

    # Add derived columns
    df["per_inference_bytes"] = df["ct_in_bytes"] + df["ct_out_bytes"]

    # Group by mode and dim_used (averaging over runs)
    agg = (
        df.groupby(["mode", "dim_used"], as_index=False)
        .agg(
            e2e_mean=("e2e_ms", "mean"),
            e2e_std=("e2e_ms", "std"),
            bw_mean=("per_inference_bytes", "mean"),
            bw_std=("per_inference_bytes", "std"),
            relerr_mean=("rel_err", "mean"),
            relerr_std=("rel_err", "std"),
        )
        .sort_values(["mode", "dim_used"])
    )

    # --- Plot 1: End-to-end latency vs dimension ---
    plt.figure()
    for mode in agg["mode"].unique():
        sub = agg[agg["mode"] == mode].sort_values("dim_used")
        plt.errorbar(sub["dim_used"], sub["e2e_mean"], yerr=sub["e2e_std"], marker="o", capsize=3, label=mode)

    plt.xlabel("Dimension used (dim_used)")
    plt.ylabel("End-to-end latency (ms)")
    plt.title("End-to-end latency vs input dimension (mean ± std)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig("plot_latency_e2e.png", dpi=200)

    # --- Plot 2: Bandwidth vs dimension ---
    plt.figure()
    for mode in agg["mode"].unique():
        sub = agg[agg["mode"] == mode].sort_values("dim_used")
        # plot in KB for readability
        plt.errorbar(sub["dim_used"], sub["bw_mean"] / 1024.0, yerr=sub["bw_std"] / 1024.0, marker="o", capsize=3, label=mode)

    plt.xlabel("Dimension used (dim_used)")
    plt.ylabel("Per-inference bandwidth (KB): ct_in + ct_out")
    plt.title("Per-inference bandwidth vs input dimension (mean ± std)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig("plot_bandwidth.png", dpi=200)

    # --- Optional Plot 3: Relative error vs dimension ---
    plt.figure()
    for mode in agg["mode"].unique():
        sub = agg[agg["mode"] == mode].sort_values("dim_used")
        plt.errorbar(sub["dim_used"], sub["relerr_mean"], yerr=sub["relerr_std"], marker="o", capsize=3, label=mode)

    plt.xlabel("Dimension used (dim_used)")
    plt.ylabel("Relative error (CKKS)")
    plt.title("Relative error vs input dimension (mean ± std)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig("plot_rel_error.png", dpi=200)

    # --- Summary table for your report ---
    # Make a compact table with just the mean values per mode/dim
    summary = agg[["mode", "dim_used", "e2e_mean", "bw_mean", "relerr_mean"]].copy()
    summary["bw_mean_kb"] = summary["bw_mean"] / 1024.0
    summary = summary.drop(columns=["bw_mean"])

    summary.to_csv("summary_table.csv", index=False)

    print("Saved:")
    print("  plot_latency_e2e.png")
    print("  plot_bandwidth.png")
    print("  plot_rel_error.png")
    print("  summary_table.csv")

if __name__ == "__main__":
    main()
