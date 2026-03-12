"""Prepare PBMC CITE-seq (Hao et al., 2021) for PPLS protein prediction.

This script can either:
  A) download the Hao et al. (2021) PBMC CITE-seq dataset via scvi-tools, then preprocess, or
  B) preprocess a user-provided `.h5ad`/`.h5mu` file.

Outputs (default under `application/`)
------------------------------------
  - citeseq_rna.csv   (N × p; log-normalized HVGs, centered)
  - citeseq_adt.csv   (N × q; CLR-normalized proteins, centered)

Notes
-----
- The exported CSVs follow the format used by the paper workflow: **no row index** (index=False).
- Downstream loader `ppls_slm.apps.data_utils.load_citeseq_data()` accepts both CSV variants
  (with or without a leading index column).

Typical usage (recommended)
---------------------------
  python scripts/prepare_citeseq.py --scvi-hao-pbmc

Or, if you already have an AnnData/MuData file:
  python scripts/prepare_citeseq.py --input application/pbmc_citeseq.h5ad

"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _to_dense(X) -> np.ndarray:
    if hasattr(X, "toarray"):
        return X.toarray()
    return np.asarray(X)


def _adt_clr(counts: np.ndarray) -> np.ndarray:
    """CLR normalization per cell: log(1+y) - mean_j log(1+y)."""

    Y_raw = np.asarray(counts, dtype=np.float64)
    Y_log = np.log1p(Y_raw)
    geo_mean_per_cell = np.mean(Y_log, axis=1, keepdims=True)
    return Y_log - geo_mean_per_cell


def _center_cols(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=np.float64)
    return M - M.mean(axis=0, keepdims=True)


def _load_from_h5ad(path: Path) -> Tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Legacy helper (kept for compatibility).

    WARNING: For very large CITE-seq datasets, do NOT use this to load full `adata.X` into a
    dense array. Prefer `_prepare_from_local()` which keeps matrices sparse until HVG selection.
    """

    import anndata  # type: ignore

    adata = anndata.read_h5ad(str(path))

    # Case A: ADT in obsm
    obsm_keys = list(getattr(adata, "obsm", {}).keys())
    candidate_keys = [
        "protein",
        "proteins",
        "adt",
        "ADT",
        "Antibody Capture",
        "antibody_capture",
        "protein_expression",
        "protein_counts",
    ]
    hit = next((k for k in candidate_keys if k in obsm_keys), None)

    if hit is not None:
        X_counts = _to_dense(adata.X)
        Y_counts = np.asarray(adata.obsm[hit])
        gene_names = [str(x) for x in getattr(adata, "var_names", [])]
        protein_names = [f"protein_{j}" for j in range(int(Y_counts.shape[1]))]
        return X_counts, Y_counts, gene_names, protein_names

    # Case B: split by feature_types
    if getattr(adata, "var", None) is None or "feature_types" not in adata.var.columns:
        raise NotImplementedError(
            "Unsupported .h5ad schema: expected ADT in adata.obsm['protein'|'adt'|...] or adata.var['feature_types']."
        )

    ft = adata.var["feature_types"].astype(str).to_numpy()
    is_rna = np.array([s.lower() in ("gene expression", "rna") for s in ft], dtype=bool)
    is_adt = np.array(["antibody" in s.lower() or "adt" in s.lower() for s in ft], dtype=bool)
    if not is_rna.any() or not is_adt.any():
        raise ValueError("feature_types did not contain both RNA and ADT features")

    X_counts = _to_dense(adata.X[:, is_rna])
    Y_counts = _to_dense(adata.X[:, is_adt])

    var_names = [str(x) for x in getattr(adata, "var_names", [])]
    gene_names = [var_names[i] for i in np.where(is_rna)[0]]
    protein_names = [var_names[i] for i in np.where(is_adt)[0]]
    return X_counts, Y_counts, gene_names, protein_names



def _load_from_h5mu(path: Path) -> Tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    import muon as mu  # type: ignore

    mdata = mu.read(str(path))
    mod_keys = list(mdata.mod.keys())

    rna_key = next((k for k in ("rna", "RNA") if k in mod_keys), None)
    adt_key = next((k for k in ("protein", "prot", "adt", "ADT") if k in mod_keys), None)
    if rna_key is None or adt_key is None:
        raise ValueError(f"Expected modalities 'rna' and 'protein/adt' in .h5mu; found {mod_keys}")

    ad_rna = mdata.mod[rna_key]
    ad_adt = mdata.mod[adt_key]

    X_counts = _to_dense(ad_rna.X)
    Y_counts = _to_dense(ad_adt.X)
    gene_names = [str(x) for x in getattr(ad_rna, "var_names", [])]
    protein_names = [str(x) for x in getattr(ad_adt, "var_names", [])]
    return X_counts, Y_counts, gene_names, protein_names


def _prepare_from_scvi(
    *,
    save_path: str,
    n_top_genes: int,
    subsample_n: Optional[int],
    seed: int,
    download_url: str,
) -> Tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Download + preprocess Hao et al. PBMC CITE-seq.

    We avoid depending on `scvi-tools` because it may not be available for very new Python
    versions. Instead, we download the official `.h5ad` (the same file used by scvi-tools)
    and run the preprocessing locally.
    """

    import urllib.request

    cache_dir = Path(save_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    h5ad_path = cache_dir / "pbmc_seurat_v4_cite_seq.h5ad"

    print("=" * 60)
    print("[1/6] Downloading PBMC CITE-seq dataset (Hao et al., 2021)...")
    print(f"      url: {download_url}")
    print(f"      dst: {h5ad_path}")
    print("=" * 60, flush=True)

    if not h5ad_path.exists() or h5ad_path.stat().st_size < 1024 * 1024:
        if h5ad_path.exists() and h5ad_path.stat().st_size < 1024 * 1024:
            try:
                h5ad_path.unlink()
            except Exception:
                pass

        tmp_path = h5ad_path.with_suffix(h5ad_path.suffix + ".part")

        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass

        # Some hosts (e.g. Figshare) may be protected by AWS WAF and return a JS challenge.
        # In that case, a programmatic download will fail; we ask the user to download via browser.
        req = urllib.request.Request(str(download_url), headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req) as r:
                status = getattr(r, "status", None)
                waf = r.headers.get("x-amzn-waf-action")
                ctype = str(r.headers.get("Content-Type", "")).lower()
                total = r.headers.get("Content-Length")
                total_i = int(total) if total is not None and str(total).isdigit() else None

                if waf == "challenge" or status == 202 or ("text/html" in ctype and (total_i in (None, 0))):
                    raise RuntimeError(
                        "Dataset host returned an access challenge (AWS WAF) instead of the .h5ad file. "
                        "Please open the download URL in a browser, save the file as "
                        f"{h5ad_path}, then re-run this script."
                    )

                chunk = 1024 * 1024
                read = 0
                t0 = time.time()

                with open(tmp_path, "wb") as f:
                    while True:
                        b = r.read(chunk)
                        if not b:
                            break
                        f.write(b)
                        read += len(b)
                        if total_i is not None and total_i > 0:
                            pct = 100.0 * read / max(1, total_i)
                            mb = read / (1024 * 1024)
                            speed = mb / max(1e-6, (time.time() - t0))
                            print(f"  downloaded: {pct:5.1f}% ({mb:,.0f} MB, {speed:.1f} MB/s)", end="\r", flush=True)
                    print("", flush=True)

            # Basic integrity check: the file should be large.
            if tmp_path.exists() and tmp_path.stat().st_size < 10 * 1024 * 1024:
                raise RuntimeError(
                    f"Downloaded file is unexpectedly small ({tmp_path.stat().st_size} bytes). "
                    "The host likely blocked the request. Please download via browser and retry."
                )

            tmp_path.replace(h5ad_path)

        except Exception:
            # Don't leave a corrupt/empty file around.
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass
            raise


    print(f"  Loading AnnData from: {h5ad_path}", flush=True)

    try:
        import anndata  # type: ignore
    except Exception as e:
        raise ImportError("Reading .h5ad requires `anndata` (pip install anndata)") from e

    adata = anndata.read_h5ad(str(h5ad_path))

    print(f"  Loaded AnnData: {adata.shape[0]} cells × {adata.shape[1]} genes")

    if "protein_expression" not in adata.obsm:
        for k in ("protein", "proteins", "adt", "ADT", "Antibody Capture", "antibody_capture"):
            if k in adata.obsm:
                adata.obsm["protein_expression"] = adata.obsm[k]
                break

    if "protein_expression" not in adata.obsm:
        raise KeyError("Expected a protein matrix in `adata.obsm['protein_expression']` (or compatible key)")

    # Optional subsample early to reduce peak memory
    if subsample_n is not None and int(subsample_n) < int(adata.shape[0]):
        rng = np.random.RandomState(int(seed))
        idx = rng.choice(adata.shape[0], size=int(subsample_n), replace=False)
        idx = np.sort(idx)
        adata = adata[idx].copy()
        print(f"  Subsampled to N={adata.shape[0]}")

    print("\n" + "=" * 60)
    print("[2/6] RNA preprocessing: normalize → log1p → HVG selection")
    print("=" * 60, flush=True)

    import scanpy as sc  # type: ignore

    # Preserve raw counts for HVG selection (on counts)
    adata.layers["counts"] = adata.X.copy()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    try:
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=int(n_top_genes),
            flavor="seurat_v3",
            layer="counts",
        )
    except Exception as e:
        print(f"[WARN] HVG selection with flavor='seurat_v3' failed: {e}")
        print("       Falling back to flavor='seurat'. For exact reproduction, install scikit-misc.")
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=int(n_top_genes),
            flavor="seurat",
            layer="counts",
        )

    n_hvg = int(adata.var["highly_variable"].sum())
    print(f"  Selected {n_hvg} highly variable genes")

    X = adata[:, adata.var["highly_variable"]].X
    X = _to_dense(X)
    X = np.asarray(X, dtype=np.float64)
    gene_names = adata.var_names[adata.var["highly_variable"]].astype(str).tolist()
    print(f"  RNA matrix shape: {X.shape}")

    print("\n" + "=" * 60)
    print("[3/6] ADT preprocessing: CLR normalization")
    print("=" * 60, flush=True)

    prot = adata.obsm["protein_expression"]
    if hasattr(prot, "values") and hasattr(prot, "columns"):
        Y_raw = np.asarray(prot.values, dtype=np.float64)
        prot_names = [str(c) for c in list(prot.columns)]
    else:
        Y_raw = np.asarray(prot, dtype=np.float64)
        prot_names = [f"protein_{j}" for j in range(int(Y_raw.shape[1]))]

    Y = _adt_clr(Y_raw)
    print(f"  ADT matrix shape: {Y.shape}")
    print(f"  Number of proteins: {len(prot_names)}")

    print("\n" + "=" * 60)
    print("[4/6] Centering both matrices (zero-mean columns)")
    print("=" * 60, flush=True)

    X = _center_cols(X)
    Y = _center_cols(Y)

    if not np.allclose(X.mean(axis=0), 0, atol=1e-8):
        raise AssertionError("X not centered")
    if not np.allclose(Y.mean(axis=0), 0, atol=1e-8):
        raise AssertionError("Y not centered")

    print("  Centering verified: column means ≈ 0", flush=True)

    return X, Y, gene_names, prot_names



def _prepare_from_local(
    *,
    input_path: str,
    n_top_genes: int,
    subsample_n: Optional[int],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Preprocess a local `.h5ad`/`.h5mu`.

    For large PBMC CITE-seq, we keep RNA sparse until HVG selection to avoid OOM.
    """

    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(p)

    lower = p.name.lower()

    if lower.endswith(".h5mu"):
        # MuData path is generally much smaller; keep the existing helper.
        X_counts, Y_counts, gene_names, protein_names = _load_from_h5mu(p)
        Xc = np.asarray(X_counts, dtype=np.float64)
        lib = np.sum(Xc, axis=1)
        lib_safe = np.where(lib > 0, lib, 1.0)
        Xn = Xc / lib_safe[:, None] * 1e4
        X_log = np.log1p(Xn)

        # HVGs by variance
        p0 = int(X_log.shape[1])
        k = int(max(1, min(int(n_top_genes), p0)))
        if p0 > k:
            v = np.var(X_log, axis=0)
            idx = np.argsort(-v, kind="mergesort")[:k]
            idx = np.sort(idx)
            X_log = X_log[:, idx]
            gene_names = [gene_names[i] for i in idx]

        Y = _adt_clr(Y_counts)
        return _center_cols(X_log), _center_cols(Y), gene_names, protein_names

    if not lower.endswith(".h5ad"):
        raise ValueError(f"Unsupported input file: {p}")

    # --- AnnData (.h5ad) ---
    import anndata  # type: ignore
    import scanpy as sc  # type: ignore

    adata = anndata.read_h5ad(str(p))

    # Optional subsample early
    if subsample_n is not None and int(subsample_n) < int(adata.shape[0]):
        rng = np.random.RandomState(int(seed))
        idx = rng.choice(adata.shape[0], size=int(subsample_n), replace=False)
        idx = np.sort(idx)
        adata = adata[idx].copy()

    # Preserve raw counts for HVG selection.
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    try:
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=int(n_top_genes),
            flavor="seurat_v3",
            layer="counts",
        )
    except Exception as e:
        print(f"[WARN] HVG selection with flavor='seurat_v3' failed: {e}")
        print("       Falling back to flavor='seurat'. For exact reproduction, install scikit-misc.")
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=int(n_top_genes),
            flavor="seurat",
            layer="counts",
        )

    if "highly_variable" not in adata.var.columns:
        raise RuntimeError("HVG selection failed: `adata.var['highly_variable']` not found")

    X = adata[:, adata.var["highly_variable"]].X
    X = _to_dense(X)
    X = np.asarray(X, dtype=np.float64)
    gene_names = adata.var_names[adata.var["highly_variable"]].astype(str).tolist()

    # Protein matrix: accept scvi naming variants.
    obsm_keys = list(getattr(adata, "obsm", {}).keys())
    prot_key = next(
        (k for k in ("protein_expression", "protein_counts", "protein", "proteins", "adt", "ADT") if k in obsm_keys),
        None,
    )
    if prot_key is None:
        raise KeyError(f"No protein matrix found in adata.obsm. Available keys: {obsm_keys}")

    prot = adata.obsm[prot_key]
    if hasattr(prot, "values") and hasattr(prot, "columns"):
        Y_raw = np.asarray(prot.values, dtype=np.float64)
        protein_names = [str(c) for c in list(prot.columns)]
    else:
        Y_raw = np.asarray(prot, dtype=np.float64)
        protein_names = [f"protein_{j}" for j in range(int(Y_raw.shape[1]))]

    Y = _adt_clr(Y_raw)

    return _center_cols(X), _center_cols(Y), gene_names, protein_names



def prepare(
    *,
    out_dir: str,
    scvi_hao_pbmc: bool,
    input_path: Optional[str],
    save_path: str,
    download_url: str,
    n_top_genes: int = 2000,
    subsample_n: Optional[int] = None,
    seed: int = 42,
) -> None:

    t_start = time.time()

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if bool(scvi_hao_pbmc):
        X, Y, gene_names, prot_names = _prepare_from_scvi(
            save_path=str(save_path),
            n_top_genes=int(n_top_genes),
            subsample_n=subsample_n,
            seed=int(seed),
            download_url=str(download_url),
        )

    else:
        if input_path is None:
            raise ValueError("Provide --input <.h5ad/.h5mu> or use --scvi-hao-pbmc")
        X, Y, gene_names, prot_names = _prepare_from_local(
            input_path=str(input_path),
            n_top_genes=int(n_top_genes),
            subsample_n=subsample_n,
            seed=int(seed),
        )

    print("\n" + "=" * 60)
    print("[5/6] Sanity checks")
    print("=" * 60, flush=True)

    N, p = X.shape
    _, q = Y.shape

    print(f"  N (cells):    {N:,}")
    print(f"  p (genes):    {p:,}")
    print(f"  q (proteins): {q:,}")
    print(f"  X has NaN:    {np.any(np.isnan(X))}")
    print(f"  Y has NaN:    {np.any(np.isnan(Y))}")
    print(f"  X has Inf:    {np.any(np.isinf(X))}")
    print(f"  Y has Inf:    {np.any(np.isinf(Y))}")

    if np.any(np.isnan(X)) or np.any(np.isnan(Y)):
        raise ValueError("Matrices contain NaN")
    if np.any(np.isinf(X)) or np.any(np.isinf(Y)):
        raise ValueError("Matrices contain Inf")

    print("\n" + "=" * 60)
    print("[6/6] Exporting to CSV")
    print("=" * 60, flush=True)

    rna_path = out / "citeseq_rna.csv"
    adt_path = out / "citeseq_adt.csv"

    # Use float32 on disk to reduce size; downstream casts to float anyway.
    X_out = np.asarray(X, dtype=np.float32)
    Y_out = np.asarray(Y, dtype=np.float32)

    pd.DataFrame(X_out, columns=gene_names).to_csv(rna_path, index=False)
    pd.DataFrame(Y_out, columns=prot_names).to_csv(adt_path, index=False)

    rna_size_mb = rna_path.stat().st_size / (1024 * 1024)
    adt_size_mb = adt_path.stat().st_size / (1024 * 1024)

    elapsed = time.time() - t_start
    print(f"  wrote: {rna_path} ({rna_size_mb:.0f}MB)")
    print(f"  wrote: {adt_path} ({adt_size_mb:.0f}MB)")
    print(f"Done in {elapsed:.0f}s ({elapsed/60:.1f} min)")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare PBMC CITE-seq CSV matrices")
    p.add_argument(
        "--scvi-hao-pbmc",
        action="store_true",
        help="Download Hao et al. PBMC CITE-seq (uses the same .h5ad as scvi-tools, but no scvi dependency)",
    )
    p.add_argument("--download_url", type=str, default="https://figshare.com/ndownloader/files/28747828", help="Override the dataset download URL")
    p.add_argument("--input", type=str, default=None, help="Path to .h5ad or .h5mu (ignored when --scvi-hao-pbmc)")
    p.add_argument("--out_dir", type=str, default="application", help="Output directory (default: application)")
    p.add_argument("--save_path", type=str, default="data", help="Download/cache directory for the .h5ad (default: data)")
    p.add_argument("--n_top_genes", type=int, default=2000)
    p.add_argument("--subsample_n", type=int, default=None, help="Optional subsample N cells to reduce memory")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(list(argv) if argv is not None else None)



def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    prepare(
        out_dir=str(args.out_dir),
        scvi_hao_pbmc=bool(args.scvi_hao_pbmc),
        input_path=args.input,
        save_path=str(args.save_path),
        download_url=str(args.download_url),
        n_top_genes=int(args.n_top_genes),
        subsample_n=args.subsample_n,
        seed=int(args.seed),
    )


    return 0


if __name__ == "__main__":
    raise SystemExit(main())
