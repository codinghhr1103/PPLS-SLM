"""Sync reproducible experiment outputs into `paper/artifacts/`.

This script copies the small, paper-relevant outputs (JSON/CSV/PNG/XLSX)
from generated result folders into a stable location tracked with the paper.

Run from repo root (recommended):
    python scripts/sync_artifacts.py

It is safe to re-run; files are overwritten.
"""

from __future__ import annotations

import shutil
import subprocess
import sys

from pathlib import Path




def copy_file(src: Path, dst: Path) -> bool:
    if not src.exists():
        print(f"[MISS] {src}")
        return False

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"[COPY] {src} -> {dst}")
    return True


def copy_glob(src_dir: Path, pattern: str, dst_dir: Path) -> int:
    if not src_dir.exists():
        print(f"[MISS] {src_dir} (dir)")
        return 0

    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for src in sorted(src_dir.glob(pattern)):
        if src.is_file():
            shutil.copy2(src, dst_dir / src.name)
            n += 1
    if n == 0:
        print(f"[MISS] {src_dir}/{pattern}")
    else:
        print(f"[COPY] {n} files: {src_dir}/{pattern} -> {dst_dir}")
    return n


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    artifacts = repo_root / "paper" / "artifacts"

    # --- Simulation / Monte Carlo (ppls_slm.cli.montecarlo) ---
    out = repo_root / "output"
    sim_dir = artifacts / "simulation"

    copied = 0
    copied += int(copy_file(out / "robustness_summary.json", sim_dir / "robustness_summary.json"))
    copied += int(copy_file(out / "results" / "mse_table.json", sim_dir / "mse_table_low.json"))
    copied += int(copy_file(out / "results_high" / "mse_table.json", sim_dir / "mse_table_high.json"))
    copied += int(copy_file(out / "results" / "results_summary.json", sim_dir / "results_summary_low.json"))
    copied += int(copy_file(out / "results_high" / "results_summary.json", sim_dir / "results_summary_high.json"))
    copied += int(copy_file(out / "results" / "experiment_summary.json", sim_dir / "experiment_summary_low.json"))
    copied += int(copy_file(out / "results_high" / "experiment_summary.json", sim_dir / "experiment_summary_high.json"))

    # Figures/tables exported by visualization stage.
    # Keep backward compatibility: low-noise figures are synced into `simulation/figures/`.
    copied += copy_glob(out / "figures", "*.png", sim_dir / "figures")
    copied += copy_glob(out / "figures", "*.xlsx", sim_dir / "figures")

    # High-noise figures may share filenames with low-noise ones, so store separately.
    copied += copy_glob(out / "figures_high", "*.png", sim_dir / "figures_high")
    copied += copy_glob(out / "figures_high", "*.xlsx", sim_dir / "figures_high")

    # --- PCCA simulation (Table 1 extension) ---
    pcca_out = out / "pcca_simulation"
    pcca_dir = artifacts / "pcca_simulation"
    copied += int(copy_file(pcca_out / "mse_table.json", pcca_dir / "mse_table.json"))
    copied += int(copy_file(pcca_out / "experiment_summary.json", pcca_dir / "experiment_summary.json"))
    copied += int(copy_file(pcca_out / "per_trial_mse.csv", pcca_dir / "per_trial_mse.csv"))

    # --- PPCA verification (Appendix) ---
    ppca_out = out / "ppca_verification"
    ppca_dir = artifacts / "ppca_verification"
    copied += int(copy_file(ppca_out / "summary.json", ppca_dir / "summary.json"))
    copied += int(copy_file(ppca_out / "summary.csv", ppca_dir / "summary.csv"))
    copied += int(copy_file(ppca_out / "per_trial.csv", ppca_dir / "per_trial.csv"))

    # --- Large-scale parameter recovery ---
    scale_out = out / "parameter_recovery_scale" / "full"
    scale_dir = artifacts / "simulation_scale"

    copied += int(copy_file(scale_out / "parameter_recovery_scale_summary.csv", scale_dir / "parameter_recovery_scale_summary.csv"))
    copied += int(copy_file(scale_out / "parameter_recovery_scale_runtime.csv", scale_dir / "parameter_recovery_scale_runtime.csv"))
    copied += int(copy_file(scale_out / "run_manifest.json", scale_dir / "run_manifest.json"))

    # --- Speed experiment ---
    # Some workflows keep the figure under paper/ for LaTeX compilation.
    speed_src_candidates = [
        repo_root / "output" / "figures" / "speed_comparison.png",
        repo_root / "speed_comparison.png",
        repo_root / "paper" / "speed_comparison.png",
    ]


    speed_src = next((p for p in speed_src_candidates if p.exists()), None)
    if speed_src is None:
        print(f"[MISS] speed comparison figure (tried: {', '.join(str(p) for p in speed_src_candidates)})")
    else:
        copied += int(copy_file(speed_src, artifacts / "speed" / "speed_comparison.png"))


    # --- Association application ---
    assoc_root = repo_root / "results_association"
    assoc_dir = artifacts / "association"
    copied += int(copy_file(assoc_root / "detection_table.csv", assoc_dir / "detection_table.csv"))
    copied += int(copy_file(assoc_root / "top10_pairs_slm.csv", assoc_dir / "top10_pairs_slm.csv"))
    copied += int(copy_file(assoc_root / "detection_comparison.png", assoc_dir / "detection_comparison.png"))

    # --- Prediction application (synthetic) ---
    pred_root = repo_root / "results_prediction"
    pred_dir = artifacts / "prediction"
    pred_syn = pred_dir / "synthetic"

    # New outputs (predictive accuracy + calibration comparison)
    copied += int(copy_file(pred_root / "prediction_metrics_per_fold.csv", pred_syn / "prediction_metrics_per_fold.csv"))
    copied += int(copy_file(pred_root / "prediction_metrics_summary.csv", pred_syn / "prediction_metrics_summary.csv"))
    copied += int(copy_file(pred_root / "calibration_comparison.csv", pred_syn / "calibration_comparison.csv"))
    copied += int(copy_file(pred_root / "selected_shrinkage_alpha.csv", pred_syn / "selected_shrinkage_alpha.csv"))


    # Optional plots
    copied += int(copy_file(pred_root / "calibration_plot.png", pred_syn / "calibration_plot.png"))

    # Backward-compatible legacy outputs (may be absent after refactor)
    copied += int(copy_file(pred_root / "coverage_results.csv", pred_syn / "coverage_results.csv"))
    copied += int(copy_file(pred_root / "coverage_table.csv", pred_syn / "coverage_table.csv"))
    copied += int(copy_file(pred_root / "prediction_example.png", pred_syn / "prediction_example.png"))

    # --- Prediction application (BRCA) ---
    brca_root = repo_root / "results_prediction_brca"
    pred_brca = pred_dir / "brca"

    copied += int(copy_file(brca_root / "brca_prediction_summary.csv", pred_brca / "brca_prediction_summary.csv"))
    copied += int(copy_file(brca_root / "brca_prediction_by_r.csv", pred_brca / "brca_prediction_by_r.csv"))
    copied += int(copy_file(brca_root / "brca_prediction_per_fold.csv", pred_brca / "brca_prediction_per_fold.csv"))
    copied += int(copy_file(brca_root / "brca_selected_shrinkage_alpha.csv", pred_brca / "brca_selected_shrinkage_alpha.csv"))
    copied += int(copy_file(brca_root / "brca_calibration_table.csv", pred_brca / "brca_calibration_table.csv"))

    # --- Prediction application (CITE-seq PBMC) ---
    cit_root = repo_root / "results_citeseq"
    pred_cit = pred_dir / "citeseq"


    copied += int(copy_file(cit_root / "citeseq_prediction_per_fold.csv", pred_cit / "citeseq_prediction_per_fold.csv"))
    copied += int(copy_file(cit_root / "citeseq_prediction_by_r.csv", pred_cit / "citeseq_prediction_by_r.csv"))
    copied += int(copy_file(cit_root / "citeseq_prediction_summary.csv", pred_cit / "citeseq_prediction_summary.csv"))
    copied += int(copy_file(cit_root / "citeseq_calibration_per_fold.csv", pred_cit / "citeseq_calibration_per_fold.csv"))
    copied += int(copy_file(cit_root / "citeseq_calibration_summary.csv", pred_cit / "citeseq_calibration_summary.csv"))
    copied += int(copy_file(cit_root / "citeseq_selected_shrinkage_alpha.csv", pred_cit / "citeseq_selected_shrinkage_alpha.csv"))
    copied += int(copy_file(cit_root / "citeseq_loadings_top.csv", pred_cit / "citeseq_loadings_top.csv"))








    # --- Model selection (latent dimension r) ---

    ms_root = repo_root / "results_model_selection"
    ms_dir = artifacts / "model_selection"

    ms_syn = ms_root / "synthetic"
    ms_syn_dst = ms_dir / "synthetic"
    copied += int(copy_file(ms_syn / "selection_accuracy_table.csv", ms_syn_dst / "selection_accuracy_table.csv"))
    copied += int(copy_file(ms_syn / "bic_per_trial.csv", ms_syn_dst / "bic_per_trial.csv"))
    copied += int(copy_file(ms_syn / "cv_mse_per_trial.csv", ms_syn_dst / "cv_mse_per_trial.csv"))
    copied += copy_glob(ms_syn / "figures", "*.png", ms_syn_dst / "figures")

    ms_brca = ms_root / "brca"
    ms_brca_dst = ms_dir / "brca"
    copied += int(copy_file(ms_brca / "brca_r_selection.csv", ms_brca_dst / "brca_r_selection.csv"))
    copied += copy_glob(ms_brca / "figures", "*.png", ms_brca_dst / "figures")

    # Also generate LaTeX tables from the synced artifacts so the paper stays consistent.

    gen_tables = repo_root / "scripts" / "generate_paper_tables.py"
    if gen_tables.exists():
        try:
            subprocess.check_call([sys.executable, str(gen_tables)], cwd=str(repo_root))
        except Exception as e:
            print(f"[WARN] table generation failed: {e}")
    else:
        print(f"[WARN] missing generator script: {gen_tables}")

    print("\n" + "=" * 72)
    print(f"Done. Copied {copied} file(s) into {artifacts}")
    print("=" * 72)

    return 0



if __name__ == "__main__":
    raise SystemExit(main())
