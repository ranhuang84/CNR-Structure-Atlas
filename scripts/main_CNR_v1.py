#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AF-Thermo v1 rebuild

Purpose
-------
1. Extract physically consistent per-protein contact statistics from AlphaFold/PDB structures:
   - Cβ-based contacts, with Gly using Cα.
   - Correct 3-letter to 1-letter residue mapping.
   - Non-local contact filtering by sequence separation.
   - pLDDT/B-factor filtering.

2. Compute two levels of conformational-response descriptors:
   A. bethe_meanfield:
      A per-protein parameterized mean-field Bethe-like response using z_nonlocal and r_hydro.
      This is the cleaned v1 baseline.

   B. graph_meanfield:
      A sequence/contact-map-aware residue-level mean-field response using the actual non-local
      contact graph of each protein. This is a stronger v1.5 option and uses the actual sequence
      through residue-specific contact weights.

Definitions
-----------
lambda_f:
    The lambda value where the conformational susceptibility dm/dlambda reaches its maximum.

lambda_sat:
    The lambda value where folded occupancy m(lambda) first reaches the saturation threshold
    m >= sat_threshold, default 0.90.

broadening_ratio:
    lambda_sat / lambda_f.

Important terminology
---------------------
This script does NOT calculate the macroscopic thermodynamic heat capacity of a protein material.
It calculates conformational occupancy response under a reduced effective non-local interaction
strength lambda at fixed physiological temperature.

Dependencies
------------
pip install biopython numpy pandas scipy tqdm matplotlib

Example usage
-------------
Single file, both models:
python af_thermo_v1_rebuild.py --input AF-P12345-F1-model_v6.cif.gz --outdir results --plot --model both

Folder batch:
python af_thermo_v1_rebuild.py --input ./alphafold_human --outdir results --model bethe --max-files 1000

Graph-aware batch, slower:
python af_thermo_v1_rebuild.py --input ./alphafold_human --outdir results_graph --model graph --max-files 100

"""

import argparse
import gzip
import math
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

try:
    from Bio.PDB import MMCIFParser, PDBParser
except Exception as exc:
    raise RuntimeError("Biopython is required. Install with: pip install biopython") from exc

try:
    from scipy.signal import find_peaks
except Exception:
    find_peaks = None


# -----------------------------
# Residue and hydrophobicity maps
# -----------------------------

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # Common modified residues mapped conservatively
    "MSE": "M",  # selenomethionine
    "SEC": "U", "PYL": "O",
}

KYTE_DOOLITTLE = {
    "A": 1.8,  "C": 2.5,  "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5,  "K": -3.9, "L": 3.8,
    "M": 1.9,  "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2,  "W": -0.9, "Y": -1.3,
    "U": 0.0,  "O": 0.0,  "X": 0.0,
}

HYDROPHOBIC_SET = {aa for aa, kd in KYTE_DOOLITTLE.items() if kd > 0.0}


@dataclass
class ProteinContacts:
    file: str
    protein_id: str
    length: int
    retained_residues: int
    retained_fraction: float
    mean_plddt: float
    sequence: str
    coords: np.ndarray
    aa: List[str]
    contact_pairs: np.ndarray  # shape (M, 2), residue indices after filtering
    contact_weights_binary: np.ndarray  # shape (M,), 1 for hydro-hydro, else 0
    contact_weights_kd: np.ndarray  # shape (M,), continuous hydrophobic weight


# -----------------------------
# Structure parsing
# -----------------------------

def open_structure_file(path: Path):
    """Return a text handle or path-like object accepted by Bio.PDB parsers."""
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return str(path)


def get_parser(path: Path):
    lower = path.name.lower()
    if lower.endswith(".cif") or lower.endswith(".cif.gz") or lower.endswith(".mmcif") or lower.endswith(".mmcif.gz"):
        return MMCIFParser(QUIET=True)
    return PDBParser(QUIET=True)


def protein_id_from_filename(path: Path) -> str:
    name = path.name
    for suffix in [".cif.gz", ".pdb.gz", ".mmcif.gz", ".cif", ".pdb", ".mmcif"]:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    return name


def parse_structure_contacts(
    path: Path,
    contact_cutoff: float = 8.0,
    plddt_threshold: float = 70.0,
    min_seq_sep: int = 2,
    chain_id: Optional[str] = None,
    use_cb: bool = True,
    kd_weight_mode: str = "positive_product",
) -> Optional[ProteinContacts]:
    """
    Parse one structure and extract non-local contact graph.

    min_seq_sep:
        A contact is retained only if abs(i-j) > min_seq_sep.
        min_seq_sep=1 excludes direct peptide neighbors.
        min_seq_sep=2 also excludes i,i+2 local geometry.
        min_seq_sep=3 is a stricter non-local setting.

    use_cb:
        If True, use Cβ for all non-Gly residues and Cα for Gly.
        If False, use Cα for all residues.

    kd_weight_mode:
        "binary": continuous weight is same as hydrophobic binary.
        "positive_product": max(KD_i,0)*max(KD_j,0), normalized by max KD^2.
        "average_positive": (max(KD_i,0)+max(KD_j,0))/(2*max KD).
    """
    parser = get_parser(path)
    handle = open_structure_file(path)
    try:
        if isinstance(handle, str):
            structure = parser.get_structure("prot", handle)
        else:
            with handle as fh:
                structure = parser.get_structure("prot", fh)
    except Exception:
        return None

    try:
        model = structure[0]
    except Exception:
        return None

    chains = list(model.get_chains())
    if not chains:
        return None

    if chain_id is not None:
        selected = [c for c in chains if c.id == chain_id]
        if not selected:
            return None
        chain = selected[0]
    else:
        # AlphaFold model files are usually single-chain. For multi-chain PDBs, choose longest polymer chain.
        polymer_chains = []
        for c in chains:
            residues = [r for r in c.get_residues() if r.get_id()[0] == " "]
            polymer_chains.append((len(residues), c))
        chain = max(polymer_chains, key=lambda x: x[0])[1]

    raw_residues = [r for r in chain.get_residues() if r.get_id()[0] == " "]
    if len(raw_residues) < 5:
        return None

    coords = []
    aa_list = []
    plddts = []
    retained = 0

    for res in raw_residues:
        resname = res.get_resname().upper()
        aa = AA3_TO_1.get(resname, "X")

        # Choose coordinate atom
        atom_name = "CA"
        if use_cb and aa != "G" and res.has_id("CB"):
            atom_name = "CB"
        elif not res.has_id("CA"):
            continue

        atom = res[atom_name]
        plddt = float(atom.get_bfactor())

        if plddt < plddt_threshold:
            continue

        coords.append(atom.get_coord().astype(float))
        aa_list.append(aa)
        plddts.append(plddt)
        retained += 1

    N = len(coords)
    if N < 10:
        return None

    coords_arr = np.asarray(coords, dtype=float)
    aa_arr = np.asarray(aa_list, dtype=object)

    # Pairwise distances. For typical proteins this is fine. For very large proteins, still manageable.
    diff = coords_arr[:, None, :] - coords_arr[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)

    # Keep upper triangle only
    i_idx, j_idx = np.where(np.triu(np.ones((N, N), dtype=bool), k=1))

    seq_sep_mask = np.abs(i_idx - j_idx) > min_seq_sep
    cutoff_mask = dist[i_idx, j_idx] < contact_cutoff
    mask = seq_sep_mask & cutoff_mask

    pairs = np.column_stack([i_idx[mask], j_idx[mask]]).astype(int)

    if len(pairs) == 0:
        contact_weights_binary = np.asarray([], dtype=float)
        contact_weights_kd = np.asarray([], dtype=float)
    else:
        ai = aa_arr[pairs[:, 0]]
        aj = aa_arr[pairs[:, 1]]

        hydro_i = np.array([a in HYDROPHOBIC_SET for a in ai], dtype=bool)
        hydro_j = np.array([a in HYDROPHOBIC_SET for a in aj], dtype=bool)
        contact_weights_binary = (hydro_i & hydro_j).astype(float)

        kd_i = np.array([KYTE_DOOLITTLE.get(a, 0.0) for a in ai], dtype=float)
        kd_j = np.array([KYTE_DOOLITTLE.get(a, 0.0) for a in aj], dtype=float)
        kd_pos_i = np.maximum(kd_i, 0.0)
        kd_pos_j = np.maximum(kd_j, 0.0)
        kd_max = max(v for v in KYTE_DOOLITTLE.values())

        if kd_weight_mode == "binary":
            contact_weights_kd = contact_weights_binary.copy()
        elif kd_weight_mode == "average_positive":
            contact_weights_kd = (kd_pos_i + kd_pos_j) / (2.0 * kd_max)
        else:
            # positive_product, normalized to [0,1]
            contact_weights_kd = (kd_pos_i * kd_pos_j) / (kd_max * kd_max)

    return ProteinContacts(
        file=str(path),
        protein_id=protein_id_from_filename(path),
        length=N,
        retained_residues=N,
        retained_fraction=N / max(len(raw_residues), 1),
        mean_plddt=float(np.mean(plddts)) if plddts else float("nan"),
        sequence="".join(aa_list),
        coords=coords_arr,
        aa=list(aa_list),
        contact_pairs=pairs,
        contact_weights_binary=contact_weights_binary,
        contact_weights_kd=contact_weights_kd,
    )


# -----------------------------
# Summary statistics
# -----------------------------

def contact_summary(pc: ProteinContacts, weight_source: str = "binary") -> Dict[str, float]:
    N = pc.length
    M = len(pc.contact_pairs)

    # Each undirected contact contributes two neighbors over N residues
    z_nonlocal = (2.0 * M / N) if N > 0 else float("nan")

    if M == 0:
        r_hydro_binary = 0.0
        mean_kd_weight = 0.0
    else:
        r_hydro_binary = float(np.mean(pc.contact_weights_binary))
        mean_kd_weight = float(np.mean(pc.contact_weights_kd))

    if weight_source == "kd":
        r_effective = mean_kd_weight
    else:
        r_effective = r_hydro_binary

    return {
        "file": pc.file,
        "protein_id": pc.protein_id,
        "length": pc.length,
        "retained_fraction": pc.retained_fraction,
        "mean_plddt": pc.mean_plddt,
        "nonlocal_contacts": M,
        "z_nonlocal": z_nonlocal,
        "r_hydro_binary": r_hydro_binary,
        "mean_kd_weight": mean_kd_weight,
        "r_effective": r_effective,
    }


# -----------------------------
# Bethe-like mean-field model
# -----------------------------

def solve_bethe_q3(
    lam: float,
    z: float,
    r_eff: float,
    q: int = 3,
    delta: float = 0.0,
    omega_unfolded: float = 1.0,
    max_iter: int = 5000,
    tol: float = 1e-11,
) -> float:
    """
    q-state Bethe-like cavity iteration.

    State 0 = folded/contact-competent.
    States 1..q-1 = partial/unfolded states.

    z can be non-integer. This is an analytic-continuation mean-field descriptor.
    It should be interpreted as an effective non-local coordination number, not a literal
    integer tree branching number.

    delta:
        Local dimensionless penalty for folded/contact-competent state.
        delta=0 reproduces the minimal equal-state toy model.

    omega_unfolded:
        Extra degeneracy weight for non-folded states. Used as a state weight, not as q.
        omega_unfolded=1 reproduces the minimal equal-state toy model.
    """
    if z <= 0 or r_eff <= 0:
        return 1.0 / q

    g = np.ones(q, dtype=float) / q

    # State statistical weights. Folded state gets exp(-delta).
    state_w = np.ones(q, dtype=float)
    state_w[0] = math.exp(-delta)
    if q > 1:
        state_w[1:] *= float(omega_unfolded)

    for _ in range(max_iter):
        g_old = g.copy()
        g_new = np.zeros(q, dtype=float)

        # Pair Boltzmann factor: contact attraction only if both are folded.
        for sigma in range(q):
            s = 0.0
            for sp in range(q):
                pair_factor = math.exp(lam * r_eff) if (sigma == 0 and sp == 0) else 1.0
                s += pair_factor * state_w[sp] * (g_old[sp] ** max(z - 1.0, 0.0))
            g_new[sigma] = s

        total = np.sum(g_new)
        if not np.isfinite(total) or total <= 0:
            return float("nan")
        g_new /= total

        if np.max(np.abs(g_new - g_old)) < tol:
            g = g_new
            break
        g = g_new

    # Single-site probability using z branches
    site_weights = np.zeros(q, dtype=float)
    for sigma in range(q):
        site_weights[sigma] = state_w[sigma] * (g[sigma] ** z)

    denom = np.sum(site_weights)
    if not np.isfinite(denom) or denom <= 0:
        return float("nan")
    p = site_weights / denom
    return float(p[0])


def scan_bethe(
    z: float,
    r_eff: float,
    lambda_min: float = 0.1,
    lambda_max: float = 4.0,
    lambda_points: int = 200,
    q: int = 3,
    delta: float = 0.0,
    omega_unfolded: float = 1.0,
    sat_threshold: float = 0.90,
) -> Dict[str, object]:
    lam_values = np.linspace(lambda_min, lambda_max, lambda_points)
    m_values = np.array([
        solve_bethe_q3(lam, z, r_eff, q=q, delta=delta, omega_unfolded=omega_unfolded)
        for lam in lam_values
    ], dtype=float)

    return summarize_scan(lam_values, m_values, sat_threshold=sat_threshold)


# -----------------------------
# Graph-aware residue-level mean-field model
# -----------------------------

def solve_graph_meanfield(
    lam: float,
    N: int,
    pairs: np.ndarray,
    weights: np.ndarray,
    delta: float = 0.0,
    omega_unfolded: float = 1.0,
    max_iter: int = 3000,
    tol: float = 1e-10,
    damping: float = 0.5,
) -> Tuple[float, np.ndarray]:
    """
    Sequence/contact-map-aware mean-field model on the actual non-local contact graph.

    Binary folded/contact-competent variable n_i in {0,1}.
    Mean-field self-consistency:

        m_i = 1 / [1 + omega_unfolded * exp(delta - lambda * sum_j w_ij m_j)]

    delta:
        Local dimensionless penalty for entering contact-competent folded state.

    omega_unfolded:
        Entropic degeneracy of the non-folded ensemble.

    This model explicitly uses:
        - N
        - actual non-local contact graph
        - residue-pair weights derived from sequence hydrophobicity
    """
    if N <= 0:
        return float("nan"), np.asarray([])

    if pairs is None or len(pairs) == 0 or np.sum(weights) <= 0:
        # No stabilizing non-local contact graph; folded occupancy is local baseline only.
        baseline = 1.0 / (1.0 + omega_unfolded * math.exp(delta))
        return float(baseline), np.full(N, baseline, dtype=float)

    neighbors = [[] for _ in range(N)]
    for (i, j), w in zip(pairs, weights):
        if w <= 0:
            continue
        neighbors[int(i)].append((int(j), float(w)))
        neighbors[int(j)].append((int(i), float(w)))

    # Start from weakly folded homogeneous state
    m = np.full(N, 1.0 / (1.0 + omega_unfolded * math.exp(delta)), dtype=float)

    for _ in range(max_iter):
        m_old = m.copy()
        field = np.zeros(N, dtype=float)

        for i in range(N):
            if neighbors[i]:
                field[i] = sum(w * m_old[j] for j, w in neighbors[i])

        arg = delta - lam * field
        # Avoid overflow in exp
        arg = np.clip(arg, -80, 80)
        m_new = 1.0 / (1.0 + omega_unfolded * np.exp(arg))

        m = damping * m_new + (1.0 - damping) * m_old

        if np.max(np.abs(m - m_old)) < tol:
            break

    return float(np.mean(m)), m


def scan_graph(
    pc: ProteinContacts,
    weight_source: str = "binary",
    lambda_min: float = 0.1,
    lambda_max: float = 4.0,
    lambda_points: int = 200,
    delta: float = 0.0,
    omega_unfolded: float = 1.0,
    sat_threshold: float = 0.90,
) -> Dict[str, object]:
    lam_values = np.linspace(lambda_min, lambda_max, lambda_points)

    if weight_source == "kd":
        weights = pc.contact_weights_kd
    else:
        weights = pc.contact_weights_binary

    m_values = []
    # Warm-start is possible, but separate solves are more robust and reproducible.
    for lam in lam_values:
        m, _ = solve_graph_meanfield(
            lam=lam,
            N=pc.length,
            pairs=pc.contact_pairs,
            weights=weights,
            delta=delta,
            omega_unfolded=omega_unfolded,
        )
        m_values.append(m)

    m_values = np.asarray(m_values, dtype=float)
    return summarize_scan(lam_values, m_values, sat_threshold=sat_threshold)


# -----------------------------
# Scan summary
# -----------------------------

def summarize_scan(lam_values: np.ndarray, m_values: np.ndarray, sat_threshold: float = 0.90) -> Dict[str, object]:
    if len(lam_values) != len(m_values) or len(lam_values) < 3:
        raise ValueError("Invalid scan arrays.")

    dm = np.gradient(m_values, lam_values)

    # lambda_f = main susceptibility peak. Use global maximum for robustness.
    if np.all(~np.isfinite(dm)):
        idx_f = 0
    else:
        idx_f = int(np.nanargmax(dm))

    lambda_f = float(lam_values[idx_f])
    susceptibility_peak = float(dm[idx_f])

    # Saturation threshold, not glass transition.
    sat_indices = np.where(m_values >= sat_threshold)[0]
    if len(sat_indices) > 0:
        idx_sat = int(sat_indices[0])
        lambda_sat = float(lam_values[idx_sat])
        broadening_ratio = float(lambda_sat / lambda_f) if lambda_f > 0 else float("nan")
    else:
        lambda_sat = float("nan")
        broadening_ratio = float("nan")

    return {
        "lambda_values": lam_values,
        "m_values": m_values,
        "dm_dlambda": dm,
        "lambda_f": lambda_f,
        "lambda_sat": lambda_sat,
        "broadening_ratio": broadening_ratio,
        "m_at_lambda_1": float(m_values[int(np.argmin(np.abs(lam_values - 1.0)))]),
        "max_m": float(np.nanmax(m_values)),
        "susceptibility_peak": susceptibility_peak,
    }


# -----------------------------
# Plotting
# -----------------------------

def save_scan_curve(scan: Dict[str, object], out_csv: Path, out_png: Optional[Path] = None, title: str = ""):
    df = pd.DataFrame({
        "lambda": scan["lambda_values"],
        "m": scan["m_values"],
        "dm_dlambda": scan["dm_dlambda"],
    })
    df.to_csv(out_csv, index=False)

    if out_png is not None:
        import matplotlib.pyplot as plt

        lam = scan["lambda_values"]
        m = scan["m_values"]
        dm = scan["dm_dlambda"]

        fig, ax1 = plt.subplots(figsize=(8, 5.5))
        ax1.plot(lam, m, linewidth=2.2, label="m(λ)")
        ax1.set_xlabel("λ (effective non-local interaction strength)")
        ax1.set_ylabel("Folded/contact-competent occupancy m")

        ax2 = ax1.twinx()
        ax2.plot(lam, dm, linestyle="--", linewidth=1.8, label="dm/dλ")
        ax2.set_ylabel("Conformational susceptibility dm/dλ")

        if np.isfinite(scan["lambda_f"]):
            ax1.axvline(scan["lambda_f"], linestyle=":", linewidth=1.5)
            ax1.text(scan["lambda_f"], 0.05, f"λ_f={scan['lambda_f']:.3f}", rotation=90, va="bottom")
        if np.isfinite(scan["lambda_sat"]):
            ax1.axvline(scan["lambda_sat"], linestyle=":", linewidth=1.5)
            ax1.text(scan["lambda_sat"], 0.05, f"λ_sat={scan['lambda_sat']:.3f}", rotation=90, va="bottom")

        ax1.set_title(title)
        ax1.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)


# -----------------------------
# File discovery and main
# -----------------------------

def iter_structure_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]

    patterns = ["*.cif.gz", "*.pdb.gz", "*.mmcif.gz", "*.cif", "*.pdb", "*.mmcif"]
    files = []
    for pat in patterns:
        files.extend(input_path.rglob(pat))

    files = sorted(set(files))

    # Deduplicate AlphaFold structures:
    # if both .cif.gz and .pdb.gz exist for the same protein/model, keep .cif.gz.
    priority = {
        ".cif.gz": 0,
        ".mmcif.gz": 1,
        ".cif": 2,
        ".mmcif": 3,
        ".pdb.gz": 4,
        ".pdb": 5,
    }

    def strip_structure_suffix(path: Path) -> str:
        name = path.name
        for suffix in [".cif.gz", ".mmcif.gz", ".pdb.gz", ".cif", ".mmcif", ".pdb"]:
            if name.endswith(suffix):
                return name[:-len(suffix)]
        return name

    chosen = {}
    for f in files:
        key = strip_structure_suffix(f)
        suffix = next((s for s in priority if f.name.endswith(s)), "")
        rank = priority.get(suffix, 99)

        if key not in chosen:
            chosen[key] = (rank, f)
        else:
            old_rank, _ = chosen[key]
            if rank < old_rank:
                chosen[key] = (rank, f)

    return [f for _, f in sorted(chosen.values(), key=lambda x: str(x[1]))]


def process_one_file(path: Path, args) -> Tuple[Optional[Dict[str, object]], List[Tuple[str, Dict[str, object]]]]:
    pc = parse_structure_contacts(
        path=path,
        contact_cutoff=args.contact_cutoff,
        plddt_threshold=args.plddt_threshold,
        min_seq_sep=args.min_seq_sep,
        chain_id=args.chain_id,
        use_cb=not args.use_ca,
        kd_weight_mode=args.kd_weight_mode,
    )
    if pc is None:
        return None, []

    stats = contact_summary(pc, weight_source=args.weight_source)

    # Flags for interpretation
    stats["low_contact_flag"] = bool(stats["nonlocal_contacts"] < args.min_contacts)
    stats["zero_effective_weight_flag"] = bool(stats["r_effective"] <= 0)
    stats["model_valid_flag"] = bool(
        stats["nonlocal_contacts"] >= args.min_contacts
        and stats["r_effective"] > 0
        and stats["length"] >= args.min_length
    )

    scans = []

    run_bethe = args.model in ("bethe", "both")
    run_graph = args.model in ("graph", "both")

    if run_bethe and stats["model_valid_flag"]:
        scan = scan_bethe(
            z=stats["z_nonlocal"],
            r_eff=stats["r_effective"],
            lambda_min=args.lambda_min,
            lambda_max=args.lambda_max,
            lambda_points=args.lambda_points,
            q=args.q_states,
            delta=args.delta,
            omega_unfolded=args.omega_unfolded,
            sat_threshold=args.sat_threshold,
        )
        prefix = "bethe"
        stats.update({
            "bethe_lambda_f": scan["lambda_f"],
            "bethe_lambda_sat": scan["lambda_sat"],
            "bethe_broadening_ratio": scan["broadening_ratio"],
            "bethe_m_at_lambda_1": scan["m_at_lambda_1"],
            "bethe_max_m": scan["max_m"],
            "bethe_susceptibility_peak": scan["susceptibility_peak"],
        })
        scans.append((prefix, scan))
    else:
        stats.update({
            "bethe_lambda_f": np.nan,
            "bethe_lambda_sat": np.nan,
            "bethe_broadening_ratio": np.nan,
            "bethe_m_at_lambda_1": np.nan,
            "bethe_max_m": np.nan,
            "bethe_susceptibility_peak": np.nan,
        })

    if run_graph and stats["model_valid_flag"]:
        scan = scan_graph(
            pc=pc,
            weight_source=args.weight_source,
            lambda_min=args.lambda_min,
            lambda_max=args.lambda_max,
            lambda_points=args.lambda_points,
            delta=args.delta,
            omega_unfolded=args.omega_unfolded,
            sat_threshold=args.sat_threshold,
        )
        prefix = "graph"
        stats.update({
            "graph_lambda_f": scan["lambda_f"],
            "graph_lambda_sat": scan["lambda_sat"],
            "graph_broadening_ratio": scan["broadening_ratio"],
            "graph_m_at_lambda_1": scan["m_at_lambda_1"],
            "graph_max_m": scan["max_m"],
            "graph_susceptibility_peak": scan["susceptibility_peak"],
        })
        scans.append((prefix, scan))
    else:
        stats.update({
            "graph_lambda_f": np.nan,
            "graph_lambda_sat": np.nan,
            "graph_broadening_ratio": np.nan,
            "graph_m_at_lambda_1": np.nan,
            "graph_max_m": np.nan,
            "graph_susceptibility_peak": np.nan,
        })

    # Store sequence only if requested because it makes CSV large.
    if args.save_sequence:
        stats["sequence"] = pc.sequence

    return stats, scans


def build_argparser():
    p = argparse.ArgumentParser(description="AF-Thermo v1 rebuild: contact extraction and conformational-response scanning.")

    p.add_argument("--input", required=True, help="Input structure file or directory.")
    p.add_argument("--outdir", default="af_thermo_results", help="Output directory.")
    p.add_argument("--model", choices=["bethe", "graph", "both"], default="bethe",
                   help="bethe = z/r mean-field; graph = actual contact-graph mean-field; both = run both.")
    p.add_argument("--max-files", type=int, default=None, help="Process only the first N files.")
    p.add_argument("--chain-id", default=None, help="Specific chain ID. Default: longest chain.")
    p.add_argument("--save-sequence", action="store_true", help="Save protein sequence in summary CSV.")

    # Contact extraction
    p.add_argument("--contact-cutoff", type=float, default=8.0, help="Contact cutoff in Å.")
    p.add_argument("--plddt-threshold", type=float, default=70.0, help="Minimum pLDDT/B-factor.")
    p.add_argument("--min-seq-sep", type=int, default=2,
                   help="Retain contacts only if |i-j| > min_seq_sep. Default 2.")
    p.add_argument("--use-ca", action="store_true", help="Use CA instead of CB/Gly-CA contacts.")
    p.add_argument("--weight-source", choices=["binary", "kd"], default="binary",
                   help="Use binary hydrophobic-hydrophobic contacts or continuous KD-based weights.")
    p.add_argument("--kd-weight-mode", choices=["positive_product", "average_positive", "binary"],
                   default="positive_product", help="How to construct continuous KD pair weights.")

    # Model parameters
    p.add_argument("--q-states", type=int, default=3, help="Internal states for Bethe q-state model. Not coordination.")
    p.add_argument("--delta", type=float, default=0.0,
                   help="Dimensionless local penalty for folded/contact-competent state.")
    p.add_argument("--omega-unfolded", type=float, default=1.0,
                   help="Entropic degeneracy weight for non-folded states.")
    p.add_argument("--lambda-min", type=float, default=0.1)
    p.add_argument("--lambda-max", type=float, default=4.0)
    p.add_argument("--lambda-points", type=int, default=200)
    p.add_argument("--sat-threshold", type=float, default=0.90,
                   help="Occupancy threshold for lambda_sat, default m>=0.90.")

    # Validity filters
    p.add_argument("--min-length", type=int, default=30, help="Minimum retained length for model scan.")
    p.add_argument("--min-contacts", type=int, default=10, help="Minimum non-local contacts for model scan.")

    # Outputs
    p.add_argument("--save-curves", action="store_true", help="Save per-protein full lambda scan CSVs.")
    p.add_argument("--plot", action="store_true", help="Save per-protein scan PNGs. Implies --save-curves.")
    p.add_argument("--summary-name", default="AF_Thermo_v1_rebuild_summary.csv", help="Summary CSV filename.")

    return p


def main():
    args = build_argparser().parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    curves_dir = outdir / "curves"
    if args.save_curves or args.plot:
        curves_dir.mkdir(parents=True, exist_ok=True)

    files = iter_structure_files(input_path)
    if args.max_files is not None:
        files = files[:args.max_files]

    if not files:
        print(f"No structure files found at: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(files)} structure file(s).")
    print(f"Model: {args.model}")
    print(f"Contact: {'CA' if args.use_ca else 'CB/Gly-CA'}, cutoff={args.contact_cutoff} Å, |i-j|>{args.min_seq_sep}")
    print(f"Weight source: {args.weight_source}")
    print(f"Note: q_states={args.q_states} is internal-state count, not lattice coordination.")

    all_stats = []
    failed = 0

    for path in tqdm(files, desc="Processing"):
        stats, scans = process_one_file(path, args)
        if stats is None:
            failed += 1
            continue

        all_stats.append(stats)

        if (args.save_curves or args.plot) and scans:
            pid = str(stats["protein_id"]).replace("/", "_").replace("\\", "_")
            for prefix, scan in scans:
                out_csv = curves_dir / f"{pid}_{prefix}_full_scan.csv"
                out_png = curves_dir / f"{pid}_{prefix}_full_scan.png" if args.plot else None
                title = f"{pid} {prefix} response"
                save_scan_curve(scan, out_csv=out_csv, out_png=out_png, title=title)

    df = pd.DataFrame(all_stats)
    summary_path = outdir / args.summary_name
    df.to_csv(summary_path, index=False)

    print("\nDone.")
    print(f"Parsed proteins: {len(df)}")
    print(f"Failed parses: {failed}")
    print(f"Summary saved to: {summary_path}")

    if len(df) > 0:
        cols = [
            "length", "nonlocal_contacts", "z_nonlocal", "r_hydro_binary",
            "mean_kd_weight", "r_effective",
            "bethe_lambda_f", "bethe_lambda_sat", "bethe_broadening_ratio",
            "graph_lambda_f", "graph_lambda_sat", "graph_broadening_ratio",
        ]
        cols = [c for c in cols if c in df.columns]
        print("\nDescriptive statistics:")
        print(df[cols].describe().to_string())


if __name__ == "__main__":
    main()
