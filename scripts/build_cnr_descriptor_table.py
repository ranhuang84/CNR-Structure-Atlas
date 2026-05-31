#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build CNR descriptor table from Bethe and calibrated Graph summary CSV files."""
import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd

def parse_uniprot(pid):
    m = re.match(r"AF-([A-Z0-9]+)-F\d+-model", str(pid))
    return m.group(1) if m else np.nan

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bethe-summary", required=True)
    ap.add_argument("--graph-summary", required=True)
    ap.add_argument("--outdir", default="cnr_outputs")
    ap.add_argument("--lower-quantile", type=float, default=0.10)
    ap.add_argument("--upper-quantile", type=float, default=0.90)
    ap.add_argument("--n-cpu", type=int, default=1, help="Reserved; script runs serially.")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    bethe = pd.read_csv(args.bethe_summary)
    graph = pd.read_csv(args.graph_summary)
    key = "protein_id"

    keep_b = [key, "file", "length", "retained_fraction", "mean_plddt",
              "nonlocal_contacts", "z_nonlocal", "r_hydro_binary",
              "mean_kd_weight", "r_effective", "model_valid_flag",
              "bethe_lambda_f", "bethe_lambda_sat", "bethe_broadening_ratio",
              "bethe_m_at_lambda_1", "bethe_max_m", "bethe_susceptibility_peak"]
    keep_g = [key, "graph_lambda_f", "graph_lambda_sat", "graph_broadening_ratio",
              "graph_m_at_lambda_1", "graph_max_m", "graph_susceptibility_peak"]
    keep_b = [c for c in keep_b if c in bethe.columns]
    keep_g = [c for c in keep_g if c in graph.columns]

    df = pd.merge(bethe[keep_b], graph[keep_g], on=key, how="inner")
    df["UniProt_ACC"] = df[key].apply(parse_uniprot)
    df["rho_HNC"] = df["z_nonlocal"] * df["r_hydro_binary"]

    valid = df["rho_HNC"].notna() & (df["rho_HNC"] > 0)
    valid &= df["bethe_lambda_f"].notna() & df["graph_lambda_f"].notna()
    valid &= (df["bethe_lambda_f"] > 0) & (df["graph_lambda_f"] > 0)
    if "model_valid_flag" in df.columns:
        valid &= df["model_valid_flag"].fillna(False).astype(bool)

    d = df.loc[valid].copy()
    d["R_topo"] = d["graph_lambda_f"] / d["bethe_lambda_f"]
    d["Delta_lambda_f"] = d["graph_lambda_f"] - d["bethe_lambda_f"]
    d["log10_R_topo"] = np.log10(d["R_topo"])
    d["CNR_activation"] = d["graph_max_m"]
    d["CNR_cooperativity"] = d["graph_susceptibility_peak"]

    q_low = d["log10_R_topo"].quantile(args.lower_quantile)
    q_high = d["log10_R_topo"].quantile(args.upper_quantile)
    d["topology_class"] = np.select(
        [d["log10_R_topo"] <= q_low, d["log10_R_topo"] >= q_high],
        ["topology-facilitated", "topology-suppressed"],
        default="density-dominated"
    )

    act_low, act_high = d["CNR_activation"].quantile([0.10, 0.90])
    coop_low, coop_high = d["CNR_cooperativity"].quantile([0.10, 0.90])
    d["activation_class"] = np.select(
        [d["CNR_activation"] <= act_low, d["CNR_activation"] >= act_high],
        ["low-activation", "high-activation"], default="intermediate-activation")
    d["cooperativity_class"] = np.select(
        [d["CNR_cooperativity"] <= coop_low, d["CNR_cooperativity"] >= coop_high],
        ["low-cooperativity", "high-cooperativity"], default="intermediate-cooperativity")

    bins = [0, 100, 300, 600, 1000, np.inf]
    labels = ["≤100", "101–300", "301–600", "601–1000", ">1000"]
    d["length_bin"] = pd.cut(d["length"], bins=bins, labels=labels, right=True)

    d.to_csv(outdir / "AF_CNR_descriptor_table.csv", index=False)

    def topn(label, col, ascending=False, n=20):
        tmp = d.sort_values(col, ascending=ascending).head(n).copy()
        tmp.insert(0, "case_category", label)
        return tmp
    cases = pd.concat([
        topn("highest_topology_suppression_R_topo", "R_topo", False),
        topn("strongest_topology_facilitation_R_topo", "R_topo", True),
        topn("highest_graph_activation_mmax", "graph_max_m", False),
        topn("lowest_graph_activation_mmax", "graph_max_m", True),
        topn("highest_graph_cooperativity_chi", "graph_susceptibility_peak", False),
        topn("lowest_graph_cooperativity_chi", "graph_susceptibility_peak", True),
    ], ignore_index=True)
    cases.to_csv(outdir / "AF_CNR_topology_cases_top20_each.csv", index=False)

    summary = {
        "n_valid_CNR": int(len(d)),
        "lower_log10_R_topo_threshold": float(q_low),
        "upper_log10_R_topo_threshold": float(q_high),
        "topology_class_counts": d["topology_class"].value_counts().to_dict(),
        "spearman": {
            "rho_HNC_vs_bethe_lambda_f": float(d[["rho_HNC","bethe_lambda_f"]].corr(method="spearman").iloc[0,1]),
            "rho_HNC_vs_graph_lambda_f": float(d[["rho_HNC","graph_lambda_f"]].corr(method="spearman").iloc[0,1]),
            "rho_HNC_vs_R_topo": float(d[["rho_HNC","R_topo"]].corr(method="spearman").iloc[0,1]),
        }
    }
    (outdir / "AF_CNR_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
