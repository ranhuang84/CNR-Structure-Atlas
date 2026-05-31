#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate main CNR figures from descriptor and validation tables."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def binned_summary(x, y, n_bins=24):
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    mids, meds, q1s, q3s = [], [], [], []
    for i in range(len(edges)-1):
        mask = (x >= edges[i]) & (x <= edges[i+1] if i == len(edges)-2 else x < edges[i+1])
        ys = y[mask]; xs = x[mask]
        if len(ys) < 10:
            continue
        mids.append(np.median(xs)); meds.append(np.median(ys))
        q1s.append(np.quantile(ys, 0.25)); q3s.append(np.quantile(ys, 0.75))
    return np.array(mids), np.array(meds), np.array(q1s), np.array(q3s)

def make_figure3(df, outpath):
    fig = plt.figure(figsize=(12,10))
    axes = [fig.add_subplot(2,2,i+1) for i in range(4)]

    ax = axes[0]
    hb = ax.hexbin(df["rho_HNC"], df["bethe_lambda_f"], gridsize=55, mincnt=1)
    x_min = max(df["rho_HNC"].min(), 2.167/10.0)
    xfit = np.linspace(x_min, df["rho_HNC"].max(), 400)
    ax.plot(xfit, 2.167/xfit, linewidth=1.5)
    ax.set_xlabel(r'$\rho_{\mathrm{HNC}}$')
    ax.set_ylabel(r'$\lambda_f^{\mathrm{Bethe}}$')
    ax.set_ylim(0, 10)
    ax.set_title("a  Bethe density baseline")
    ax.text(0.03, 0.95, "Inverse density trend in the main response range", transform=ax.transAxes, va="top", fontsize=9)
    fig.colorbar(hb, ax=ax).set_label("count")

    ax = axes[1]
    hb = ax.hexbin(df["rho_HNC"], df["graph_lambda_f"], gridsize=55, mincnt=1)
    ax.set_xlabel(r'$\rho_{\mathrm{HNC}}$')
    ax.set_ylabel(r'$\lambda_f^{\mathrm{Graph}}$')
    ax.set_title("b  Graph threshold retains density dependence")
    ax.text(0.03,0.95,r"Spearman $\rho=-0.738$", transform=ax.transAxes, va="top", fontsize=9)
    fig.colorbar(hb, ax=ax).set_label("count")

    ax = axes[2]
    markers = {"topology-facilitated":"o","density-dominated":"s","topology-suppressed":"^"}
    for cls in ["topology-facilitated","density-dominated","topology-suppressed"]:
        part = df[df["topology_class"]==cls]
        if len(part)>2500:
            part = part.iloc[np.linspace(0, len(part)-1, 2500, dtype=int)]
        ax.scatter(part["bethe_lambda_f"], part["graph_lambda_f"], s=8, alpha=0.35, marker=markers[cls], label=cls)
    lim_min = min(df["bethe_lambda_f"].min(), df["graph_lambda_f"].min())
    lim_max = max(df["bethe_lambda_f"].quantile(0.995), df["graph_lambda_f"].quantile(0.995))
    ax.plot([lim_min, lim_max], [lim_min, lim_max], linestyle="--", linewidth=1)
    ax.set_xlim(lim_min, lim_max); ax.set_ylim(lim_min, lim_max)
    ax.set_xlabel(r'$\lambda_f^{\mathrm{Bethe}}$')
    ax.set_ylabel(r'$\lambda_f^{\mathrm{Graph}}$')
    ax.set_title("c  Graph thresholds deviate from the Bethe baseline")
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    ax = axes[3]
    mids, meds, q1s, q3s = binned_summary(df["rho_HNC"].values, df["R_topo"].values)
    ax.fill_between(mids, q1s, q3s, alpha=0.25)
    ax.plot(mids, meds, linewidth=1.6)
    ax.axhline(1.0, linestyle="--", linewidth=1)
    ax.set_xlabel(r'$\rho_{\mathrm{HNC}}$')
    ax.set_ylabel(r'$R_{\mathrm{topo}}=\lambda_f^{Graph}/\lambda_f^{Bethe}$')
    ax.set_title("d  Topology correction emerges as a residual axis")

    fig.suptitle("Figure 3. Density baseline and graph-derived topology correction", fontsize=14)
    fig.tight_layout(rect=[0,0,1,0.97])
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)

def make_figure4(df, outpath):
    d = df.copy()
    d["log10_rho_HNC"] = np.log10(d["rho_HNC"])
    d["log10_R_topo"] = np.log10(d["R_topo"])
    q10, q90 = d["log10_R_topo"].quantile([0.10,0.90])

    fig = plt.figure(figsize=(12,10))
    axes = [fig.add_subplot(2,2,i+1) for i in range(4)]

    ax = axes[0]
    hb = ax.hexbin(d["log10_rho_HNC"], d["log10_R_topo"], gridsize=55, mincnt=1)
    ax.set_xlabel(r'$\log_{10}\rho_{\mathrm{HNC}}$')
    ax.set_ylabel(r'$\log_{10}R_{\mathrm{topo}}$')
    ax.set_title("a  CNR density–topology atlas")
    fig.colorbar(hb, ax=ax).set_label("count")

    ax = axes[1]
    ax.hist(d["log10_R_topo"], bins=60)
    ax.axvline(q10, linestyle="--", linewidth=1)
    ax.axvline(q90, linestyle="--", linewidth=1)
    ax.set_xlabel(r'$\log_{10}R_{\mathrm{topo}}$')
    ax.set_ylabel("Protein count")
    ax.set_title("b  Empirical distribution of topology correction")

    ax = axes[2]
    order = ["≤100","101–300","301–600","601–1000",">1000"]
    data = [d.loc[d["length_bin"]==lab, "log10_R_topo"].dropna().values for lab in order]
    ax.boxplot(data, tick_labels=order, showfliers=False)
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel("Protein length bin")
    ax.set_ylabel(r'$\log_{10}R_{\mathrm{topo}}$')
    ax.set_title("c  Topology correction across protein length bins")
    ax.tick_params(axis="x", rotation=15)

    ax = axes[3]
    hb = ax.hexbin(d["log10_R_topo"], d["graph_max_m"], gridsize=55, mincnt=1)
    mids, meds, q1s, q3s = binned_summary(d["log10_R_topo"].values, d["graph_max_m"].values)
    ax.plot(mids, meds, linewidth=1.4)
    ax.set_xlabel(r'$\log_{10}R_{\mathrm{topo}}$')
    ax.set_ylabel(r'$m_{\max}^{\mathrm{Graph}}$')
    ax.set_title("d  Topology correction relates to graph activation amplitude")
    fig.colorbar(hb, ax=ax).set_label("count")

    fig.suptitle("Figure 4. Atlas-level organization of topology correction in CNR space", fontsize=14)
    fig.tight_layout(rect=[0,0,1,0.97])
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)

def make_disorder_figures(merged, outdir):
    order = ["none","low (0–0.1)","moderate (0.1–0.3)","high (0.3–0.6)","very high (>0.6)"]
    merged["uniprot_disorder_class"] = pd.Categorical(merged["uniprot_disorder_class"], categories=order, ordered=True)

    metrics = [
        ("rho_HNC", r"$\rho_{\mathrm{HNC}}$", "a  Contact density declines with disorder"),
        ("bethe_lambda_f", r"$\lambda_f^{Bethe}$", "b  Bethe threshold increases with disorder"),
        ("graph_max_m", r"$m_{\max}^{Graph}$", "c  Graph activation decreases with disorder"),
        ("graph_susceptibility_peak", r"$\chi_{\max}^{Graph}$", "d  Graph cooperativity decreases with disorder"),
    ]
    fig = plt.figure(figsize=(13,10))
    for i,(metric,ylabel,title) in enumerate(metrics):
        ax = fig.add_subplot(2,2,i+1)
        data = [merged.loc[merged["uniprot_disorder_class"]==cls, metric].dropna().values for cls in order]
        ax.boxplot(data, tick_labels=order, showfliers=False)
        ax.tick_params(axis="x", rotation=22)
        ax.set_ylabel(ylabel); ax.set_title(title)
    fig.suptitle("CNR descriptors align with UniProt/MobiDB-lite disorder annotations", fontsize=14)
    fig.tight_layout(rect=[0,0,1,0.97])
    fig.savefig(outdir/"Figure5_CNR_disorder_validation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    pairs = [
        ("rho_HNC", r"$\rho_{\mathrm{HNC}}$", -0.288, "a  Disorder versus contact density"),
        ("bethe_lambda_f", r"$\lambda_f^{Bethe}$", 0.289, "b  Disorder versus Bethe threshold"),
        ("graph_max_m", r"$m_{\max}^{Graph}$", -0.244, "c  Disorder versus graph activation"),
        ("graph_susceptibility_peak", r"$\chi_{\max}^{Graph}$", -0.215, "d  Disorder versus graph cooperativity"),
    ]
    fig = plt.figure(figsize=(13,10))
    for i,(metric,ylabel,rho,title) in enumerate(pairs):
        ax = fig.add_subplot(2,2,i+1)
        ax.scatter(merged["uniprot_disorder_fraction"], merged[metric], s=5, alpha=0.13, color="#333333")
        ax.set_xlabel("UniProt/MobiDB-lite disorder fraction")
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.text(0.04,0.94,f"Spearman rho = {rho:.3f}\nn = {len(merged):,}", transform=ax.transAxes, va="top", fontsize=10)
    fig.suptitle("Broad-coverage disorder validation of CNR descriptors", fontsize=14)
    fig.tight_layout(rect=[0,0,1,0.97])
    fig.savefig(outdir/"Figure6_CNR_continuous_disorder_validation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnr-table", required=True)
    ap.add_argument("--uniprot-merged-table", default=None)
    ap.add_argument("--outdir", default="cnr_figures")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.cnr_table)
    make_figure3(df, outdir/"Figure3_density_baseline_graph_correction.png")
    make_figure4(df, outdir/"Figure4_atlas_level_topology_organization.png")
    if args.uniprot_merged_table:
        merged = pd.read_csv(args.uniprot_merged_table)
        make_disorder_figures(merged, outdir)
    print(f"Figures saved to: {outdir}")

if __name__ == "__main__":
    import argparse
    main()
