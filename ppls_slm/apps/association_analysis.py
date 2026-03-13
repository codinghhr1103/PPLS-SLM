"""
Association Analysis on Multi-Omics Data
========================================================

Reproduces the multi-omics association analysis from Section 6.4 of the paper.

This application:
1. Loads (or simulates) paired gene-expression / protein-expression data.
2. Randomly subsamples Ns samples and nGs/nPs features to form X_s and Y_s.
3. Fits the PPLS model using both SLM and EM with r latent variables.
4. Projects data onto the latent space: T = X_s W,  U = Y_s C.
5. Screens significant gene-protein pairs via Pearson correlation tests
   (Algorithm 2 in the paper).
6. Compares the number of detected pairs between SLM and EM at multiple
   significance thresholds and reports the top-10 gene-protein pairs.

Real TCGA-BRCA Data (Paper Section 6.4.1: N=705, p=604 genes, q=223 proteins)
-----------------------------------------------------------------------------
Download the following two files from:
    https://linkedomics.org/data_download/TCGA-BRCA/

  Gene expression (RNAseq HiSeq, gene level, log2-RSEM):
    Human__TCGA_BRCA__UNC__RNAseq__HiSeq_RNA__01_28_2016__BI__Gene__Firehose_RSEM_log2.cct.gz
    (1093 samples x 20155 genes, TAB-separated, genes are ROWS, samples are COLUMNS)

  Protein expression (RPPA analyte level):
    Human__TCGA_BRCA__MDA__RPPA__MDA_RPPA__01_28_2016__BI__Analyte__Firehose_RPPA.cct
    (887 samples x 212 antibodies, TAB-separated, proteins are ROWS, samples are COLUMNS)

  The load_omics_data() function automatically handles this format:
  it detects the tab-separator and transposes genes-as-rows -> samples-as-rows.
  Intersecting the ~1093 RNAseq and ~887 RPPA samples yields ~705 common samples.
  After removing features with NaN, the paper's p=604 genes and q=223 proteins result.

Usage
-----
# With the synthetic (demo) dataset (default):
    python application_association_analysis.py

# With real TCGA-BRCA LinkedOmics data:
    python application_association_analysis.py \\
        --gene_expr   HiSeq_RNA__Firehose_RSEM_log2.cct.gz \\
        --protein_expr RPPA__Firehose_RPPA.cct

# With a plain CSV (samples x features):
    python application_association_analysis.py \\
        --gene_expr   gene_expression.csv \\
        --protein_expr protein_expression.csv

Dependencies
------------
Requires the project modules ppls_model.py and algorithms.py to be on the
Python path (i.e. located in the same directory as this script).
"""

import argparse
import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# ── project imports ──────────────────────────────────────────────────────────
from ppls_slm.algorithms import (
    ECMAlgorithm,
    EMAlgorithm,
    InitialPointGenerator,
    ScalarLikelihoodMethod,
)
from ppls_slm.ppls_model import PPLSModel


# ─────────────────────────────────────────────────────────────────────────────
#  Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_omics_data(gene_path: str, protein_path: str) -> Tuple[np.ndarray, np.ndarray,
                                                                  List[str], List[str]]:
    """
    Load paired gene-expression and protein-expression matrices from data files.

    Supports two file layouts automatically:

    **LinkedOmics / TCGA Firehose .cct format (recommended for real TCGA-BRCA data)**
    - Tab-separated (.cct, .txt, or .tsv)
    - Rows = genes / proteins  (first column = gene/protein name)
    - Columns = samples         (first row = TCGA sample barcodes, e.g. TCGA-A1-A0SB-01)
    - Download from: https://linkedomics.org/data_download/TCGA-BRCA/
      · Gene expression (RNAseq HiSeq):
            Human__TCGA_BRCA__UNC__RNAseq__HiSeq_RNA__01_28_2016__BI__Gene__Firehose_RSEM_log2.cct.gz
      · Protein expression (RPPA analyte level):
            Human__TCGA_BRCA__MDA__RPPA__MDA_RPPA__01_28_2016__BI__Analyte__Firehose_RPPA.cct

    **Plain CSV format (samples × features)**
    - Comma-separated (.csv)
    - Rows = samples  (first column = sample ID)
    - Columns = genes / proteins

    The function auto-detects the separator and transposes if genes are in rows.

    Parameters
    ----------
    gene_path : str
        Path to gene-expression file (.cct / .tsv / .txt / .csv, optionally .gz).
    protein_path : str
        Path to protein-expression file (same formats).

    Returns
    -------
    X : ndarray, shape (N, n_genes)        — samples × genes
    Y : ndarray, shape (N, n_proteins)     — samples × proteins
    gene_names : list of str
    protein_names : list of str
    """
    def _read_omics_file(path: str) -> pd.DataFrame:
        """
        Read a gene/protein expression file and return a DataFrame with
        shape (samples, features), index = sample IDs, columns = feature names.
        """
        # Detect separator: .cct / .tsv / .txt → tab; .csv → comma
        lower = path.lower().rstrip('.gz')
        sep = '\t' if any(lower.endswith(ext) for ext in ('.cct', '.tsv', '.txt')) else ','

        df = pd.read_csv(path, sep=sep, index_col=0)

        # Detect orientation:
        # LinkedOmics .cct: features (genes) are rows, samples are columns.
        # A heuristic: if column names look like TCGA barcodes (start with "TCGA-"),
        # then rows are features → transpose.
        sample_like_cols = sum(1 for c in df.columns if str(c).startswith('TCGA-'))
        if sample_like_cols > max(1, len(df.columns) // 2):
            # Features are rows → transpose to (samples, features)
            df = df.T

        return df

    X_df = _read_omics_file(gene_path)
    Y_df = _read_omics_file(protein_path)

    # Align by shared sample IDs (index)
    common = X_df.index.intersection(Y_df.index)
    if len(common) == 0:
        raise ValueError(
            "Gene and protein matrices share no common sample IDs.\n"
            "Check that both files use the same TCGA barcode format."
        )
    X_df = X_df.loc[common]
    Y_df = Y_df.loc[common]

    print(f"  Shared samples after alignment: {len(common)}")

    gene_names    = list(X_df.columns)
    protein_names = list(Y_df.columns)

    # Drop features with any NaN, then Z-score standardise across samples
    X_df = X_df.dropna(axis=1)
    Y_df = Y_df.dropna(axis=1)

    X = X_df.values.astype(float)
    Y = Y_df.values.astype(float)

    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
    Y = (Y - Y.mean(axis=0)) / (Y.std(axis=0) + 1e-10)

    return X, Y, gene_names[:X.shape[1]], protein_names[:Y.shape[1]]


def load_brca_data_w_subtypes(path: str) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Load the bundled BRCA dataset (705 samples) and split into genes/proteins.

    The repo ships a preprocessed single-table dataset:
      `application/brca_data_w_subtypes.csv.zip`

    Columns are grouped by prefix:
    - `rs_` : gene-expression features (604 columns)
    - `pp_` : protein-expression features (223 columns)

    Other columns (e.g., `cn_`, `mu_`, and clinical labels) are ignored for
    the association-analysis experiment.

    Returns
    -------
    X : (N, 604) ndarray
    Y : (N, 223) ndarray
    gene_names : list[str]
    protein_names : list[str]
    """
    import io
    import zipfile

    p = path
    lower = p.lower()

    if lower.endswith('.zip'):
        with zipfile.ZipFile(p) as z:
            names = z.namelist()
            if not names:
                raise ValueError(f"Empty zip file: {p}")
            # Expect a single CSV inside
            csv_name = names[0]
            data = z.read(csv_name)
            df = pd.read_csv(io.BytesIO(data))
    else:
        df = pd.read_csv(p)

    rs_cols = [c for c in df.columns if str(c).startswith('rs_')]
    pp_cols = [c for c in df.columns if str(c).startswith('pp_')]

    if len(rs_cols) == 0 or len(pp_cols) == 0:
        raise ValueError(
            "BRCA combined dataset does not contain expected `rs_` (genes) and `pp_` (proteins) columns. "
            f"Found rs_={len(rs_cols)}, pp_={len(pp_cols)} in: {p}"
        )

    X = df[rs_cols].to_numpy(dtype=float)
    Y = df[pp_cols].to_numpy(dtype=float)

    # Z-score standardise across samples
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
    Y = (Y - Y.mean(axis=0)) / (Y.std(axis=0) + 1e-10)

    # Strip the `rs_` / `pp_` prefixes for nicer reporting in tables.
    gene_names = [str(c)[3:] if str(c).startswith("rs_") else str(c) for c in rs_cols]
    protein_names = [str(c)[3:] if str(c).startswith("pp_") else str(c) for c in pp_cols]

    return X, Y, gene_names, protein_names


def simulate_omics_data(n_samples: int = 705,
                        n_genes: int = 604,
                        n_proteins: int = 223,
                        r_true: int = 5,
                        sigma_e2: float = 0.5,
                        sigma_f2: float = 0.5,
                        sigma_h2: float = 0.1,
                        seed: int = 0) -> Tuple[np.ndarray, np.ndarray,
                                                List[str], List[str]]:
    """
    Generate synthetic multi-omics data with a PPLS structure.
    Used when no real dataset is provided.

    Returns X (N×p), Y (N×q), gene_names, protein_names.
    """
    rng = np.random.RandomState(seed)
    model = PPLSModel(n_genes, n_proteins, r_true)

    # Random orthonormal loadings
    W, _ = np.linalg.qr(rng.randn(n_genes, r_true))
    C, _ = np.linalg.qr(rng.randn(n_proteins, r_true))

    b = np.sort(rng.uniform(0.5, 2.0, r_true))[::-1]
    theta_t2 = np.sort(rng.uniform(0.5, 1.5, r_true))[::-1]
    # Enforce identifiability: theta_t2 * b decreasing
    order = np.argsort(theta_t2 * b)[::-1]
    b = b[order]
    theta_t2 = theta_t2[order]

    B = np.diag(b)
    Sigma_t = np.diag(theta_t2)

    X, Y = model.sample(n_samples, W, C, B, Sigma_t, sigma_e2, sigma_f2, sigma_h2)

    gene_names = [f"Gene_{i+1}" for i in range(n_genes)]
    protein_names = [f"Protein_{i+1}" for i in range(n_proteins)]
    return X, Y, gene_names, protein_names


# ─────────────────────────────────────────────────────────────────────────────
#  Subsampling
# ─────────────────────────────────────────────────────────────────────────────

def subsample_omics(X: np.ndarray, Y: np.ndarray,
                    gene_names: List[str], protein_names: List[str],
                    n_samples: int = 100,
                    n_genes_s: int = 100,
                    n_proteins_s: int = 100,
                    seed: int = 42) -> Tuple[np.ndarray, np.ndarray,
                                             List[str], List[str]]:
    """
    Randomly subsample Ns samples, nGs genes and nPs proteins.

    Returns X_s, Y_s, gene_names_s, protein_names_s.
    """
    rng = np.random.RandomState(seed)

    N = X.shape[0]
    n_g = min(n_genes_s, X.shape[1])
    n_p = min(n_proteins_s, Y.shape[1])
    n_s = min(n_samples, N)

    sample_idx = rng.choice(N, n_s, replace=False)
    gene_idx = rng.choice(X.shape[1], n_g, replace=False)
    protein_idx = rng.choice(Y.shape[1], n_p, replace=False)

    X_s = X[sample_idx][:, gene_idx]
    Y_s = Y[sample_idx][:, protein_idx]

    gene_names_s = [gene_names[i] for i in gene_idx]
    protein_names_s = [protein_names[i] for i in protein_idx]

    return X_s, Y_s, gene_names_s, protein_names_s


# ─────────────────────────────────────────────────────────────────────────────
#  Algorithm 2 — correlation screening (paper Section 6.4.1)
# ─────────────────────────────────────────────────────────────────────────────

def compute_correlations_and_screen(
    X_s: np.ndarray,
    Y_s: np.ndarray,
    W: np.ndarray,
    C: np.ndarray,
    gene_names: List[str],
    protein_names: List[str],
    alpha: float = 1e-7,
) -> Dict:
    """
    Algorithm 2 from the paper.

    1. Project data onto latent scores: T = X_s W,  U = Y_s C.
    2. For every (gene, latent-variable) pair, compute Pearson r and p-value.
    3. For every (protein, latent-variable) pair, same.
    4. For each latent variable j, return sets S_{x,j} and S_{y,j} of
       significantly correlated genes / proteins.

    Parameters
    ----------
    X_s, Y_s : observed data matrices (Ns × nGs), (Ns × nPs)
    W : gene loading matrix (nGs × r)
    C : protein loading matrix (nPs × r)
    gene_names, protein_names : feature name lists
    alpha : significance threshold for p-values

    Returns
    -------
    dict with keys:
        R_xt  : (nGs, r) Pearson correlations gene ↔ latent T
        R_yu  : (nPs, r) Pearson correlations protein ↔ latent U
        P_xt  : (nGs, r) p-values
        P_yu  : (nPs, r) p-values
        sig_genes   : list[list[int]] significant gene indices per LV
        sig_proteins: list[list[int]] significant protein indices per LV
    """
    r = W.shape[1]
    n_g, n_p = len(gene_names), len(protein_names)

    T = X_s @ W   # (Ns, r)
    U = Y_s @ C   # (Ns, r)

    R_xt = np.zeros((n_g, r))
    P_xt = np.ones((n_g, r))
    R_yu = np.zeros((n_p, r))
    P_yu = np.ones((n_p, r))

    for j in range(r):
        for i in range(n_g):
            r_val, p_val = stats.pearsonr(X_s[:, i], T[:, j])
            R_xt[i, j] = r_val
            P_xt[i, j] = p_val
        for i in range(n_p):
            r_val, p_val = stats.pearsonr(Y_s[:, i], U[:, j])
            R_yu[i, j] = r_val
            P_yu[i, j] = p_val

    # Screen significant features per latent variable
    sig_genes = []
    sig_proteins = []
    for j in range(r):
        sig_genes.append([i for i in range(n_g) if P_xt[i, j] < alpha])
        sig_proteins.append([i for i in range(n_p) if P_yu[i, j] < alpha])

    return {
        "R_xt": R_xt, "P_xt": P_xt,
        "R_yu": R_yu, "P_yu": P_yu,
        "sig_genes": sig_genes,
        "sig_proteins": sig_proteins,
    }


def count_gene_protein_pairs(screen_result: Dict,
                              n_genes: int,
                              n_proteins: int) -> int:
    """
    Count the number of unique significant gene-protein pairs:
    a pair (gene_i, protein_k) is included if both are significantly
    associated with the same latent variable j.
    """
    pairs = set()
    r = len(screen_result["sig_genes"])
    for j in range(r):
        for gi in screen_result["sig_genes"][j]:
            for pi in screen_result["sig_proteins"][j]:
                pairs.add((gi, pi))
    return len(pairs)


def get_top_pairs(screen_result: Dict,
                  gene_names: List[str],
                  protein_names: List[str],
                  top_k: int = 10) -> pd.DataFrame:
    """
    Return the top-k gene-protein pairs sorted by |rho_gene| + |rho_protein|.

    Only considers pairs where BOTH the gene and the protein are significant
    for the same latent variable.
    """
    R_xt = screen_result["R_xt"]
    R_yu = screen_result["R_yu"]
    r = R_xt.shape[1]

    rows = []
    seen = set()
    for j in range(r):
        for gi in screen_result["sig_genes"][j]:
            for pi in screen_result["sig_proteins"][j]:
                key = (gi, pi)
                if key in seen:
                    continue
                seen.add(key)
                rho_g = R_xt[gi, j]
                rho_p = R_yu[pi, j]
                rows.append({
                    "LV": j + 1,
                    "Gene": gene_names[gi],
                    "rho(G,LV)": rho_g,
                    "Protein": protein_names[pi],
                    "rho(P,LV)": rho_p,
                    "sum|rho|": abs(rho_g) + abs(rho_p),
                })

    if not rows:
        return pd.DataFrame(columns=["LV", "Gene", "rho(G,LV)",
                                      "Protein", "rho(P,LV)", "sum|rho|"])

    df = pd.DataFrame(rows).sort_values("sum|rho|", ascending=False).head(top_k)
    df = df.reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Main experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_association_analysis(
    X_full: np.ndarray,
    Y_full: np.ndarray,
    gene_names: List[str],
    protein_names: List[str],
    r: int = 10,
    n_samples_sub: int = 100,
    n_genes_sub: int = 100,
    n_proteins_sub: int = 100,
    n_starts: int = 32,
    p_thresholds: Optional[List[float]] = None,
    subsample_seed: int = 42,
    algorithm_seed: int = 42,
    output_dir: str = "results_association",
    slm_max_iter: int = 50,
    em_max_iter: int = 200,
):
    """
    Full association-analysis pipeline.

    Parameters
    ----------
    X_full, Y_full : full gene/protein matrices
    gene_names, protein_names : feature names
    r : number of latent variables
    n_samples_sub, n_genes_sub, n_proteins_sub : subsample sizes (Ns, nGs, nPs)
    n_starts : multi-start count
    p_thresholds : list of significance thresholds to sweep
    subsample_seed : RNG seed for subsampling
    algorithm_seed : RNG seed for algorithm initialisation
    output_dir : directory for saving results
    """
    if p_thresholds is None:
        p_thresholds = [1e-7, 1e-6, 1e-5, 1e-4]

    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Subsample ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Association Analysis  |  r={r}  |  Ns={n_samples_sub}  |"
          f"  nGs={n_genes_sub}  |  nPs={n_proteins_sub}")
    print(f"{'='*60}")

    X_s, Y_s, gene_s, protein_s = subsample_omics(
        X_full, Y_full, gene_names, protein_names,
        n_samples=n_samples_sub, n_genes_s=n_genes_sub, n_proteins_s=n_proteins_sub,
        seed=subsample_seed,
    )
    p, q = X_s.shape[1], Y_s.shape[1]
    print(f"Subsampled data:  X_s {X_s.shape},  Y_s {Y_s.shape}")

    # ── 2. Generate shared starting points ───────────────────────────────────
    init_gen = InitialPointGenerator(p=p, q=q, r=r,
                                     n_starts=n_starts,
                                     random_seed=algorithm_seed)
    starting_points = init_gen.generate_starting_points()
    print(f"Generated {n_starts} starting points.")

    # ── 3. Fit PPLS with SLM ─────────────────────────────────────────────────
    print("\nFitting PPLS with SLM...")
    slm = ScalarLikelihoodMethod(p=p, q=q, r=r, max_iter=int(slm_max_iter),
                                 use_noise_preestimation=True)
    slm_res = slm.fit(X_s, Y_s, starting_points)
    print(f"  SLM done  |  iterations={slm_res['n_iterations']}"
          f"  |  success={slm_res['success']}")

    # ── 4. Fit PPLS with EM ──────────────────────────────────────────────────
    print("Fitting PPLS with EM...")
    em = EMAlgorithm(p=p, q=q, r=r, max_iter=int(em_max_iter), tolerance=1e-4)
    em_res = em.fit(X_s, Y_s, starting_points)
    print(f"  EM done   |  iterations={em_res['n_iterations']}")

    # ── 5. Correlation screening at multiple thresholds ──────────────────────
    print("\nScreening gene-protein pairs...")
    table_rows = []
    slm_screen_strict = None   # Keep the strictest for top-10 table

    for alpha in p_thresholds:
        slm_screen = compute_correlations_and_screen(
            X_s, Y_s, slm_res["W"], slm_res["C"],
            gene_s, protein_s, alpha=alpha,
        )
        em_screen = compute_correlations_and_screen(
            X_s, Y_s, em_res["W"], em_res["C"],
            gene_s, protein_s, alpha=alpha,
        )

        slm_n = count_gene_protein_pairs(slm_screen, p, q)
        em_n  = count_gene_protein_pairs(em_screen,  p, q)

        # Count overlap (same gene-protein pair found by both)
        def pair_set(sc):
            pairs = set()
            r_ = len(sc["sig_genes"])
            for j in range(r_):
                for gi in sc["sig_genes"][j]:
                    for pi in sc["sig_proteins"][j]:
                        pairs.add((gi, pi))
            return pairs

        slm_pairs = pair_set(slm_screen)
        em_pairs  = pair_set(em_screen)
        overlap   = len(slm_pairs & em_pairs)

        table_rows.append({
            "p-value threshold": f"p < 1e{int(np.log10(alpha))}",
            "SLM": slm_n,
            "EM": em_n,
            "Overlap": overlap,
        })

        if alpha == min(p_thresholds):
            slm_screen_strict = slm_screen

    # ── 6. Print results ─────────────────────────────────────────────────────
    detection_df = pd.DataFrame(table_rows)
    print("\n── Table: Number of detected gene-protein pairs ──")
    print(detection_df.to_string(index=False))

    # Top-10 pairs from SLM at strictest threshold
    if slm_screen_strict is not None:
        top10 = get_top_pairs(slm_screen_strict, gene_s, protein_s, top_k=10)
        print("\n── Top-10 gene-protein pairs (SLM, strictest threshold) ──")
        if top10.empty:
            print("  (No significant pairs found at strictest threshold.)")
        else:
            print(top10.to_string(index=False))
    else:
        top10 = pd.DataFrame()

    # ── 7. Save results ───────────────────────────────────────────────────────
    detection_df.to_csv(os.path.join(output_dir, "detection_table.csv"), index=False)
    if not top10.empty:
        top10.to_csv(os.path.join(output_dir, "top10_pairs_slm.csv"), index=False)

    np.save(os.path.join(output_dir, "slm_W.npy"), slm_res["W"])
    np.save(os.path.join(output_dir, "slm_C.npy"), slm_res["C"])
    np.save(os.path.join(output_dir, "em_W.npy"),  em_res["W"])
    np.save(os.path.join(output_dir, "em_C.npy"),  em_res["C"])
    print(f"\nResults saved to: {output_dir}/")

    return {
        "slm_results": slm_res,
        "em_results":  em_res,
        "detection_table": detection_df,
        "top10_pairs": top10,
        "slm_screen_strict": slm_screen_strict,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Visualisation (optional, requires matplotlib)
# ─────────────────────────────────────────────────────────────────────────────

def plot_detection_comparison(detection_df: pd.DataFrame,
                               output_dir: str = "results_association"):
    """
    Bar chart comparing the number of gene-protein pairs detected by SLM vs EM.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib not found – skipping plot.")
        return

    thresholds = detection_df["p-value threshold"].tolist()
    x = np.arange(len(thresholds))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, detection_df["SLM"], width,
                   label="SLM", color="#2E7D32", alpha=0.85, edgecolor="k")
    bars2 = ax.bar(x + width / 2, detection_df["EM"],  width,
                   label="EM",  color="#C62828", alpha=0.85, edgecolor="k")

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5, str(int(h)),
                ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(thresholds)
    ax.set_xlabel("Significance threshold")
    ax.set_ylabel("Number of gene-protein pairs")
    ax.set_title("Gene-protein pairs detected by SLM vs EM")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "detection_comparison.png"), dpi=150)
    plt.close(fig)
    print(f"Detection plot saved to {output_dir}/detection_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="PPLS Association Analysis (SLM vs EM)"
    )
    parser.add_argument("--gene_expr",    type=str, default=None,
                        help="CSV path for gene-expression (samples × genes).")
    parser.add_argument("--protein_expr", type=str, default=None,
                        help="CSV path for protein-expression (samples × proteins).")
    parser.add_argument("--brca_data",    type=str, default=None,
                        help="Path to bundled BRCA combined dataset (.csv or .zip).")
    parser.add_argument("--r",            type=int, default=5,
                        help="Number of latent variables (default: 5).")
    parser.add_argument("--n_samples",    type=int, default=60,
                        help="Number of subsampled samples Ns (default: 60).")
    parser.add_argument("--n_genes",      type=int, default=60,
                        help="Number of subsampled genes nGs (default: 60).")
    parser.add_argument("--n_proteins",   type=int, default=60,
                        help="Number of subsampled proteins nPs (default: 60).")
    parser.add_argument("--n_starts",     type=int, default=8,
                        help="Multi-start initializations (default: 8).")
    parser.add_argument("--slm_max_iter", type=int, default=50,
                        help="SLM max iterations (default: 50).")
    parser.add_argument("--em_max_iter",  type=int, default=200,
                        help="EM max iterations (default: 200).")
    parser.add_argument("--output_dir",   type=str, default="results_association",
                        help="Directory to save results.")
    parser.add_argument("--seed",         type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--plot",         action="store_true",
                        help="Generate comparison bar chart.")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Load or simulate data ─────────────────────────────────────────────────
    if args.gene_expr is not None and args.protein_expr is not None:
        print("Loading real omics data (two-file format)...")
        X_full, Y_full, gene_names, protein_names = load_omics_data(
            args.gene_expr, args.protein_expr
        )
        print(f"  Gene expression:    {X_full.shape}")
        print(f"  Protein expression: {Y_full.shape}")

    else:
        # Prefer the bundled BRCA dataset if available.
        # The paper uses the repo-shipped combined dataset under `application/`.
        # We resolve it relative to repo root so running from anywhere is safe.
        from pathlib import Path
        repo_root = Path(__file__).resolve().parents[2]  # .../ppls_slm
        repo_root = repo_root.parent                    # repo root

        candidate_brca = []
        if args.brca_data is not None:
            candidate_brca.append(Path(args.brca_data))

        # Repo-shipped dataset (paper default)
        candidate_brca.append(repo_root / "application" / "brca_data_w_subtypes.csv.zip")

        # Legacy location (in case users copied it next to this script)
        candidate_brca.append(Path(__file__).resolve().parent / "brca_data_w_subtypes.csv.zip")

        brca_path = next((str(p) for p in candidate_brca if p is not None and p.exists()), None)

        if brca_path is not None:
            print(f"Loading bundled BRCA combined dataset: {brca_path}")
            X_full, Y_full, gene_names, protein_names = load_brca_data_w_subtypes(brca_path)
            print(f"  Gene expression (rs_):   {X_full.shape}")
            print(f"  Protein expression (pp_): {Y_full.shape}")
        else:
            print("No data paths provided — generating synthetic multi-omics data.")
            print("(Pass --brca_data, or --gene_expr and --protein_expr for real TCGA-BRCA data.)\n")
            X_full, Y_full, gene_names, protein_names = simulate_omics_data(
                n_samples=705, n_genes=604, n_proteins=223,
                r_true=5, sigma_e2=0.5, sigma_f2=0.5, sigma_h2=0.1,
                seed=args.seed,
            )
            print(f"  Simulated gene expression:    {X_full.shape}")
            print(f"  Simulated protein expression: {Y_full.shape}")

    # ── Run analysis ──────────────────────────────────────────────────────────
    results = run_association_analysis(
        X_full, Y_full, gene_names, protein_names,
        r=args.r,
        n_samples_sub=args.n_samples,
        n_genes_sub=args.n_genes,
        n_proteins_sub=args.n_proteins,
        n_starts=args.n_starts,
        p_thresholds=[1e-7, 1e-6, 1e-5, 1e-4],
        subsample_seed=args.seed,
        algorithm_seed=args.seed,
        output_dir=args.output_dir,
    )

    # ── Optional plot ─────────────────────────────────────────────────────────
    if args.plot:
        plot_detection_comparison(results["detection_table"], args.output_dir)

    print("\nAssociation analysis complete.")
    return results


if __name__ == "__main__":
    main()
