# CNR: Structure-derived Contact-Network Responsiveness

This repository contains the code used to compute **Contact-Network Responsiveness (CNR)** descriptors from protein structures.

CNR is a structure-source-agnostic framework: it can be applied to experimentally determined PDB structures, AlphaFold models, or other predicted structures with residue-level coordinates. In the associated manuscript, human AlphaFold structures are used as the high-coverage structural substrate.

## What CNR computes

For each protein structure, the workflow computes:

- `rho_HNC`: hydrophobic non-local contact density
- `bethe_lambda_f`: density-controlled Bethe response threshold
- `graph_lambda_f`: graph-aware response threshold
- `R_topo = graph_lambda_f / bethe_lambda_f`: topology correction ratio
- `Delta_lambda_f`
- `graph_max_m`: maximum graph activation amplitude
- `graph_susceptibility_peak`: graph response sharpness/cooperativity

## Repository structure

```text
scripts/
  af_thermo_v1_rebuild.py          # main contact-extraction + Bethe/Graph engine
  build_cnr_descriptor_table.py    # merge Bethe/Graph summaries and build CNR table
  validate_disprot.py              # curated DisProt validation
  validate_uniprot_mobidb.py       # UniProt/MobiDB-lite disorder validation
  make_main_figures.py             # regenerate major manuscript figures
examples/
  params_template.csv              # parameter template
docs/
  methods_notes.md
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows PowerShell

pip install -r requirements.txt
```

## Step 1. Run Bethe baseline

```bash
python scripts/af_thermo_v1_rebuild.py \
  --input ./data \
  --outdir ./results/bethe_lam8 \
  --model bethe \
  --lambda-min 0.1 \
  --lambda-max 8 \
  --lambda-points 300 \
  --summary-name AF_Thermo_v1_rebuild_summary.csv
```

## Step 2. Run calibrated graph-aware model

```bash
python scripts/af_thermo_v1_rebuild.py \
  --input ./data \
  --outdir ./results/graph_delta25_omega2 \
  --model graph \
  --delta 2.5 \
  --omega-unfolded 2.0 \
  --lambda-min 0.1 \
  --lambda-max 20 \
  --lambda-points 400 \
  --summary-name AF_Thermo_v1_rebuild_summary.csv
```

## Step 3. Build CNR descriptor table

```bash
python scripts/build_cnr_descriptor_table.py \
  --bethe-summary ./results/bethe_lam8/AF_Thermo_v1_rebuild_summary.csv \
  --graph-summary ./results/graph_delta25_omega2/AF_Thermo_v1_rebuild_summary.csv \
  --outdir ./results/cnr_descriptors
```

## Step 4. Validate against UniProt/MobiDB-lite disorder annotations

Download a UniProt reviewed human proteome TSV containing at least:

- Entry
- Entry Name
- Protein names
- Gene Names
- Length
- Region

Then run:

```bash
python scripts/validate_uniprot_mobidb.py \
  --cnr-table ./results/cnr_descriptors/AF_CNR_descriptor_table.csv \
  --uniprot-tsv ./data/uniprot_human_reviewed_with_region.tsv \
  --outdir ./results/uniprot_mobidb_validation
```

## Step 5. Validate against DisProt

```bash
python scripts/validate_disprot.py \
  --cnr-table ./results/cnr_descriptors/AF_CNR_descriptor_table.csv \
  --disprot-tsv ./data/DisProt_current_IDPO-GO.tsv \
  --outdir ./results/disprot_validation
```

## Step 6. Regenerate main figures

```bash
python scripts/make_main_figures.py \
  --cnr-table ./results/cnr_descriptors/AF_CNR_descriptor_table.csv \
  --uniprot-merged-table ./results/uniprot_mobidb_validation/AF_CNR_UniProt_MobiDBlite_merged_table.csv \
  --outdir ./results/figures
```

## Notes on reproducibility

- The main engine deduplicates structure files by model identifier and keeps `.cif.gz` preferentially when both `.cif.gz` and `.pdb.gz` are present.
- Contacts are Cβ-based, with glycine represented by Cα.
- The default contact cutoff is 8 Å.
- The default pLDDT/B-factor threshold is 70.
- Peptide-neighbor contacts are excluded by the `--min-seq-sep` setting.
- Bethe `q_states` is the number of internal states, not the lattice coordination number.
- `lambda` is an effective interaction-strength coordinate, not physical temperature.
- The `n_cpu` parameter is retained in the parameter template for workflow consistency. The current reproducible scripts are serial unless explicitly parallelized by an external runner.

## Minimal citation statement

If using this code, cite the associated manuscript and the relevant structural databases used as input.
