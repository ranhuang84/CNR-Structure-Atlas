#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate CNR descriptors against curated DisProt disorder annotations."""
import argparse, json
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnr-table", required=True)
    ap.add_argument("--disprot-tsv", required=True)
    ap.add_argument("--outdir", default="cnr_disprot_validation")
    ap.add_argument("--n-cpu", type=int, default=1, help="Reserved; script runs serially.")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    cnr = pd.read_csv(args.cnr_table)
    dis = pd.read_csv(args.disprot_tsv, sep="\t")

    if "UniProt ACC" not in dis.columns or "Protein Disorder Content" not in dis.columns:
        raise ValueError("Expected DisProt columns: 'UniProt ACC' and 'Protein Disorder Content'.")

    if "NCBI Taxon ID" in dis.columns:
        dis_h = dis[dis["NCBI Taxon ID"].astype(str) == "9606"].copy()
    elif "Organism" in dis.columns:
        dis_h = dis[dis["Organism"].astype(str).str.contains("Homo sapiens", case=False, na=False)].copy()
    else:
        dis_h = dis.copy()

    dis_h["Protein Disorder Content"] = pd.to_numeric(dis_h["Protein Disorder Content"], errors="coerce")
    agg = dis_h.groupby("UniProt ACC", as_index=False).agg({"Protein Disorder Content": "max"})
    agg = agg.rename(columns={"UniProt ACC": "UniProt_ACC", "Protein Disorder Content": "disprot_disorder_content"})
    merged = cnr.merge(agg, on="UniProt_ACC", how="inner")

    descriptors = ["rho_HNC","bethe_lambda_f","graph_lambda_f","R_topo","Delta_lambda_f",
                   "graph_max_m","graph_susceptibility_peak","mean_plddt","retained_fraction"]
    corrs = {}
    for col in descriptors:
        if col in merged.columns:
            sub = merged[[col, "disprot_disorder_content"]].dropna()
            if len(sub) > 2:
                corrs[col] = float(sub.corr(method="spearman").iloc[0, 1])

    merged.to_csv(outdir / "AF_CNR_DisProt_merged_table.csv", index=False)
    pd.DataFrame({"descriptor": list(corrs.keys()),
                  "spearman_vs_disprot_disorder_content": list(corrs.values())}).to_csv(
        outdir / "AF_CNR_DisProt_spearman_correlations.csv", index=False)

    summary = {
        "disprot_rows": int(len(dis)),
        "disprot_human_rows": int(len(dis_h)),
        "merged_rows": int(len(merged)),
        "spearman_vs_disprot_disorder_content": corrs,
    }
    (outdir / "AF_CNR_DisProt_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
