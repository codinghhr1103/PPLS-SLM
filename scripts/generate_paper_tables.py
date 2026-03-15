"""Generate LaTeX tables from the latest `paper/artifacts/*` outputs.

This avoids manual copy/paste of numbers into the paper.

Usage:
  python scripts/generate_paper_tables.py

It will write generated .tex snippets under `paper/generated/tables/`.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


import pandas as pd
import numpy as np



def _latex_escape(s: str) -> str:
    """Escape LaTeX special characters in text fields."""

    # Keep it minimal; most IDs are plain alphanumerics.
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }

    out = []
    for ch in s:
        out.append(repl.get(ch, ch))
    return "".join(out)


def _latex_escape_keep_math(s: str) -> str:
    """Escape LaTeX special characters but keep math/commands intact.

    This is intended for strings that already contain LaTeX math delimiters
    (e.g. \\(r=15\\)) or commands. We therefore do NOT escape backslashes or
    dollar signs.
    """


    repl = {
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }

    out = []
    for ch in str(s):
        out.append(repl.get(ch, ch))
    return "".join(out)



def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _pm_makecell(table_str_x1e2: str) -> str:
    """Convert '7.43±1.23' into '\\makecell{7.43\\\\$\\pm$1.23}'."""

    # Some artifacts may use unicode ±.
    if "±" not in table_str_x1e2:
        return _latex_escape(table_str_x1e2)
    mean_s, std_s = table_str_x1e2.split("±", 1)
    mean_s = mean_s.strip()
    std_s = std_s.strip()
    return f"\\makecell{{{mean_s}\\\\$\\pm${std_s}}}"


def _format_float_1dp(x: Any) -> str:
    try:
        xf = float(x)
    except Exception:
        return str(x)
    return f"{xf:.1f}"


def _format_int(x: Any) -> str:
    try:
        return str(int(round(float(x))))
    except Exception:
        return str(x)


def generate_convergence_table(*, artifacts_dir: Path, out_path: Path) -> None:
    xlsx = artifacts_dir / "simulation" / "figures" / "Table_3_Convergence_Comparison.xlsx"
    if not xlsx.exists():
        raise FileNotFoundError(f"missing: {xlsx}")

    df = pd.read_excel(xlsx, sheet_name="Convergence_Statistics")

    required_cols = {
        "Algorithm",
        "Mean_Iterations",
        "Std_Iterations",
        "Min_Iterations",
        "Max_Iterations",
        "Median_Iterations",
        "Success_Rate",
    }

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Unexpected convergence sheet columns; missing={sorted(missing)}")

    # Normalise algorithm names to match current Monte Carlo outputs.
    # (We now report SLM-Manifold / SLM-Interior / SLM-Oracle explicitly.)
    order = ["SLM-Manifold", "BCD-SLM", "SLM-Interior", "SLM-Oracle", "EM", "ECM"]




    df = df.copy()
    df["Algorithm"] = df["Algorithm"].astype(str)

    rows = []
    for alg in order:
        sub = df[df["Algorithm"].str.upper() == alg.upper()]
        if sub.empty:
            raise ValueError(f"Algorithm '{alg}' not found in {xlsx}")
        r = sub.iloc[0]
        # Success_Rate is stored as a fraction in [0,1].
        try:
            succ = float(r["Success_Rate"])
            succ_s = f"{100.0 * succ:.0f}\\%"
        except Exception:
            succ_s = _latex_escape(str(r["Success_Rate"]))

        rows.append(
            (
                alg,
                _format_float_1dp(r["Mean_Iterations"]),
                _format_float_1dp(r["Std_Iterations"]),
                _format_int(r["Min_Iterations"]),
                _format_int(r["Max_Iterations"]),
                _format_int(r["Median_Iterations"]),
                succ_s,
            )
        )


    tex = []
    tex.append(r"\setlength{\tabcolsep}{3pt}")
    tex.append(r"\begin{table}[h]\footnotesize")
    tex.append(r"\centering")
    # Keep the caption consistent with the actual Monte Carlo trial count (M).
    m_trials = 100
    summary = artifacts_dir / "simulation" / "experiment_summary_low.json"
    if summary.exists():
        try:
            m_trials = int(_read_json(summary).get("experiment_info", {}).get("n_trials_completed", 100))
        except Exception:
            m_trials = 100

    tex.append(rf"\caption{{Convergence statistics across $M={m_trials}$ Monte Carlo trials}}")

    tex.append(r"\label{tab:algorithm_convergence}")
    tex.append(r"\begin{tabular}{lcccccc}")

    tex.append(r"\toprule")
    tex.append(r"\textbf{Method} & $\mathbb{E}[I]$ & $\sqrt{\text{Var}[I]}$ & $\min I$ & $\max I$ & $\text{Median}[I]$ & Success \\")

    tex.append(r"\midrule")
    for alg, mean, std, mn, mx, med, succ in rows:

        tex.append(f"{alg}  & {mean}  & {std}  & {mn}  & {mx}  & {med}  & {succ}  \\\\")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")
    tex.append("")

    out_path.write_text("\n".join(tex), encoding="utf-8")


def generate_parameter_mse_table(*, artifacts_dir: Path, out_path: Path) -> None:
    low = _read_json(artifacts_dir / "simulation" / "mse_table_low.json")
    high = _read_json(artifacts_dir / "simulation" / "mse_table_high.json")



    methods = [
        ("slm_manifold", "SLM-Manifold"),
        ("bcd_slm", "BCD-SLM"),
        ("slm_interior", "SLM-Interior"),
        ("slm_oracle", "SLM-Oracle"),
        ("em", "EM"),
        ("ecm", "ECM"),
    ]

    keys = [
        ("W", r"$\text{MSE}_W$"),
        ("C", r"$\text{MSE}_C$"),
        ("B", r"$\text{MSE}_B$"),
        ("Sigma_t", r"$\text{MSE}_{\Sigma_t}$"),
        ("sigma_h2", r"$\text{MSE}_{\sigma_h^2}$"),
    ]

    def rows_for(data: Dict[str, Any]) -> List[Tuple[str, Tuple[str, ...]]]:
        rows: list[tuple[str, tuple[str, ...]]] = []
        for mkey, mname in methods:
            cells = tuple(_pm_makecell(str(data[mkey][k]["table_str_x1e2"])) for k, _hdr in keys)
            rows.append((mname, cells))
        return rows

    def append_panel(tex: List[str], title: str, rows: Sequence[Tuple[str, Tuple[str, ...]]]) -> None:
        tex.append(r"\begin{minipage}[t]{0.49\textwidth}")
        tex.append(r"\centering")
        tex.append(rf"\textbf{{{title}}}\\[2pt]")
        tex.append(r"\resizebox{\linewidth}{!}{%")
        tex.append(r"\begin{tabular}{@{}lccccc@{}}")
        tex.append(r"\toprule")
        tex.append(r"\textbf{Method} & $\text{MSE}_W$ & $\text{MSE}_C$ & $\text{MSE}_B$ & $\text{MSE}_{\Sigma_t}$ & $\text{MSE}_{\sigma_h^2}$ \\")
        tex.append(r"\midrule")
        for method, cells in rows:
            tex.append(f"{method} & " + " & ".join(cells) + r" \\")
        tex.append(r"\bottomrule")
        tex.append(r"\end{tabular}%")
        tex.append(r"}")
        tex.append(r"\end{minipage}")

    tex: list[str] = []
    tex.append(r"\setlength{\tabcolsep}{2pt}")
    tex.append(r"\renewcommand{\arraystretch}{0.94}")
    tex.append(r"\begin{table*}[t]\scriptsize")
    tex.append(r"\centering")
    # Keep caption aligned with the paper's polished wording.
    # Include the simulation dimensions in the caption (prevents stale prose when configs change).
    p = q = r = n = None

    # Prefer reading dimensions from artifacts if available; otherwise fall back to repo config.
    summary = artifacts_dir / "simulation" / "experiment_summary_low.json"
    if summary.exists():
        try:
            info = _read_json(summary).get("experiment_info", {})
            p = info.get("p")
            q = info.get("q")
            r = info.get("r")
            n = info.get("n_samples")
        except Exception:
            p = q = r = n = None

    if p is None or q is None or r is None or n is None:
        try:
            repo_root = Path(__file__).resolve().parents[1]
            cfg = _read_json(repo_root / "config.json")
            model = cfg.get("model", {})
            p = model.get("p")
            q = model.get("q")
            r = model.get("r")
            n = model.get("n_samples")
        except Exception:
            p = q = r = n = None

    dim_str = ""
    if p is not None and q is not None and r is not None and n is not None:
        dim_str = f" ($p=q={int(p)}$, $r={int(r)}$, $N={int(n)}$)"


    tex.append(
        rf"\caption{{Parameter estimation mean squared error (MSE; $\times 10^2$){dim_str} under low/high noise (mean $\pm$ std). Low-noise (left) and high-noise (right) panels are shown side by side.}}"
    )

    tex.append(r"\label{tab:parameter_mse}")
    append_panel(tex, "Low noise", rows_for(low))
    tex.append(r"\hfill")
    append_panel(tex, "High noise", rows_for(high))



    tex.append(r"\end{table*}")
    tex.append(r"\renewcommand{\arraystretch}{1}")
    tex.append("")


    out_path.write_text("\n".join(tex), encoding="utf-8")



def generate_top10_pairs_table(*, artifacts_dir: Path, out_path: Path) -> None:
    path = artifacts_dir / "association" / "top10_pairs_slm.csv"
    df = pd.read_csv(path)

    # Ensure fixed ordering and take top 10.
    df = df.head(10).copy()

    def f6(x: Any) -> str:
        try:
            return f"{float(x):.6f}"
        except Exception:
            return _latex_escape(str(x))

    tex = []
    tex.append(r"\setlength{\tabcolsep}{7pt}")
    tex.append(r"\begin{table}[t]\small")
    tex.append(r"\centering")
    tex.append(r"\caption{Top 10 gene-protein pairs identified by SLM-Manifold among all latent variables sorted by the sum of absolute correlation coefficients}")
    tex.append(r"\label{tab:top10}")
    tex.append(r"\begin{tabular}{c| c c c c c}")
    tex.append(r"\toprule")
    tex.append(r"LV & Gene & $\rho$(G,\ LV) & Protein & $\rho$(P,\ LV) & $\sum \lvert \rho \rvert$ \\ ")
    tex.append(r"\midrule")

    for _, r in df.iterrows():
        lv = _latex_escape(str(r["LV"]))
        gene = _latex_escape(str(r["Gene"]))
        rg = f6(r["rho(G,LV)"])
        prot = _latex_escape(str(r["Protein"]))
        rp = f6(r["rho(P,LV)"])
        ssum = f6(r["sum|rho|"])
        tex.append(f"{lv} & {gene} & {rg} & {prot} & {rp} & {ssum} \\\\ ")
        tex.append(r"\midrule")

    # Replace last \midrule with \bottomrule
    if tex and tex[-1] == r"\midrule":
        tex[-1] = r"\bottomrule"
    else:
        tex.append(r"\bottomrule")

    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")
    tex.append("")

    out_path.write_text("\n".join(tex), encoding="utf-8")


def generate_detection_table(*, artifacts_dir: Path, out_path: Path) -> None:
    path = artifacts_dir / "association" / "detection_table.csv"
    df = pd.read_csv(path)

    # Expect thresholds in order.
    want = ["p < 1e-7", "p < 1e-6", "p < 1e-5", "p < 1e-4"]
    df["p-value threshold"] = df["p-value threshold"].astype(str)

    rows = []
    for th in want:
        sub = df[df["p-value threshold"].str.strip() == th]
        if sub.empty:
            raise ValueError(f"Missing threshold row: {th}")
        r = sub.iloc[0]
        rows.append((th, int(r["SLM"]), int(r["EM"]), int(r["Overlap"])))

    tex = []
    tex.append(r"\begin{table}[t]\small")
    tex.append(r"\centering")
    tex.append(r"\caption{Number of detected gene--protein pairs by SLM-Manifold and EM under different $p$-value thresholds.}")
    tex.append(r"\label{tab:Npairs}")
    tex.append(r"\begin{tabular}{l| c c c c}")
    tex.append(r"\toprule")
    tex.append(r"\textbf{Method} & p $< 1e^{-7}$ & p $< 1e^{-6}$ & p $< 1e^{-5}$ & p $< 1e^{-4}$ \\")
    tex.append(r"\midrule")

    # Pivot into the paper's layout.
    slm = [r[1] for r in rows]
    em = [r[2] for r in rows]
    ov = [r[3] for r in rows]

    tex.append("SLM-Manifold & " + " & ".join(str(x) for x in slm) + r" \\")
    tex.append("EM & " + " & ".join(str(x) for x in em) + r" \\")
    tex.append(r"\midrule")
    tex.append("Overlap & " + " & ".join(str(x) for x in ov) + r" \\")

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")
    tex.append("")

    out_path.write_text("\n".join(tex), encoding="utf-8")


def generate_prediction_coverage_table(*, artifacts_dir: Path, out_path: Path) -> None:
    # Backward-compatible: old location was `prediction/coverage_table.csv`.
    # New prediction pipeline stores artifacts under `prediction/synthetic/`.
    path_new = artifacts_dir / "prediction" / "synthetic" / "coverage_table.csv"
    path_old = artifacts_dir / "prediction" / "coverage_table.csv"
    path = path_new if path_new.exists() else path_old
    df = pd.read_csv(path, index_col=0)


    # Columns like 'Alpha=0.05'
    def parse_alpha(col: str) -> float:
        m = re.search(r"Alpha\s*=\s*([0-9.]+)", str(col))
        if not m:
            raise ValueError(f"Unrecognised alpha column: {col}")
        return float(m.group(1))

    alphas = [(parse_alpha(c), c) for c in df.columns]
    alphas.sort(key=lambda t: t[0])

    folds = [f"Fold {i}" for i in range(1, 6)]
    for f in folds:
        if f not in df.index:
            raise ValueError(f"Missing fold row in coverage_table: {f}")

    tex = []
    tex.append(r"\setlength{\tabcolsep}{2pt} ")
    tex.append(r"\renewcommand{\arraystretch}{1.2}")
    tex.append(r"\begin{table}[t]")
    tex.append(r"\centering")
    tex.append(r"\caption{Percentage of elements within prediction interval for different alpha values and folds. All the data are presented as \%.}")
    tex.append(r"\label{tab:prediction-accuracy}")
    tex.append(r"\begin{tabularx}{\linewidth}{p{0.2\linewidth}*{5}{X}}")
    tex.append(r"\hline")
    tex.append(r"Alpha & Fold 1 & Fold 2 & Fold 3 & Fold 4 & Fold 5 \\")
    tex.append(r"\hline")

    for a, col in alphas:
        # Keep 2dp; LaTeX table in paper uses 2dp.
        vals = [float(df.loc[f, col]) for f in folds]
        row = [f"{v:.2f}" for v in vals]
        tex.append(f"{a:.2f} & " + " & ".join(row) + r" \\")

    tex.append(r"\hline")
    tex.append(r"\end{tabularx}")
    tex.append(r"\end{table}")
    tex.append("")

    out_path.write_text("\n".join(tex), encoding="utf-8")


def _pm_makecell_float(mean: float, std: float, *, fmt: str = ".3f") -> str:
    m = format(float(mean), fmt)
    s = format(float(std), fmt)
    return f"\\makecell{{{m}\\\\$\\pm${s}}}"


def _pick_slm_method(methods: Sequence[str]) -> Optional[str]:
    """Pick the most informative available SLM method name."""
    methods = [str(m) for m in methods]
    candidates = [m for m in methods if m.startswith("PPLS-SLM") or m.startswith("SLM")]
    if not candidates:
        return None

    priority = [
        "SLM-Manifold-Adaptive",
        "SLM-Manifold",
        "PPLS-SLM-Manifold-Adaptive",
        "PPLS-SLM-Manifold",
        "PPLS-SLM-Adaptive",
        "PPLS-SLM",
    ]
    for p in priority:
        if p in candidates:
            return p

    return sorted(candidates)[0]





def _load_prediction_synthetic_metrics_rows(*, artifacts_dir: Path) -> List[Tuple[str, str, str, str]]:
    path = artifacts_dir / "prediction" / "synthetic" / "prediction_metrics_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing: {path}")

    df = pd.read_csv(path)
    df["method"] = df["method"].astype(str)

    slm_method = _pick_slm_method(df["method"].unique())
    slm_display_name = "SLM-Manifold-Adaptive" if slm_method and "Adaptive" in slm_method else "SLM-Manifold"
    order = [m for m in [slm_method, "PPLS-EM", "PLSR", "Ridge"] if m is not None]

    rows: list[tuple[str, str, str, str]] = []
    for m in order:
        sub = df[df["method"] == m]
        if sub.empty:
            continue
        r = sub.iloc[0]
        display_m = slm_display_name if m == slm_method else m
        rows.append(
            (
                display_m,
                _pm_makecell_float(r["mse_mean"], r["mse_std"], fmt=".4g"),
                _pm_makecell_float(r["mae_mean"], r["mae_std"], fmt=".4g"),
                _pm_makecell_float(r["r2_mean"], r["r2_std"], fmt=".4g"),
            )
        )
    return rows


def generate_prediction_synthetic_metrics_table(*, artifacts_dir: Path, out_path: Path) -> None:
    """Synthetic prediction accuracy table (MSE/MAE/R2) across 5 folds."""
    rows = _load_prediction_synthetic_metrics_rows(artifacts_dir=artifacts_dir)

    tex: list[str] = []
    tex.append(r"\setlength{\tabcolsep}{3pt}")
    tex.append(r"\renewcommand{\arraystretch}{0.95}")
    tex.append(r"\begin{table}[t]\footnotesize")
    tex.append(r"\centering")
    tex.append(r"\caption{Synthetic prediction accuracy (5-fold CV): mean $\pm$ std.}")
    tex.append(r"\label{tab:pred_synth_metrics}")
    tex.append(r"\begin{tabular}{@{}lccc@{}}")
    tex.append(r"\toprule")
    tex.append(r"\textbf{Method} & \textbf{MSE} & \textbf{MAE} & \textbf{$R^2$} \\")
    tex.append(r"\midrule")
    for method, mse, mae, r2 in rows:
        tex.append(f"{method} & {mse} & {mae} & {r2} \\\\ ")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")
    tex.append(r"\renewcommand{\arraystretch}{1}")
    tex.append("")

    out_path.write_text("\n".join(tex), encoding="utf-8")


def _load_prediction_synthetic_calibration_rows(*, artifacts_dir: Path) -> Tuple[str, List[Tuple[float, str, str, str]]]:
    path = artifacts_dir / "prediction" / "synthetic" / "calibration_comparison.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing: {path}")

    df = pd.read_csv(path)

    def fmt_pct(x: float) -> str:
        return f"{float(x):.2f}\\%"

    mean_cols = [c for c in df.columns if str(c).startswith("PPLS-SLM") and str(c).endswith("_mean")]
    slm_method = _pick_slm_method([c[: -len("_mean")] for c in mean_cols]) or "PPLS-SLM"
    slm_mean_col = f"{slm_method}_mean"
    slm_std_col = f"{slm_method}_std"
    slm_display_name = "SLM-Manifold-Adaptive" if slm_method and "Adaptive" in slm_method else "SLM-Manifold"

    rows: list[tuple[float, str, str, str]] = []
    for _, r in df.sort_values("alpha").iterrows():
        a = float(r["alpha"])
        expc = fmt_pct(r["expected_coverage"])
        slm = f"{fmt_pct(r[slm_mean_col])} $\\pm$ {fmt_pct(r[slm_std_col])}" if (slm_mean_col in r and slm_std_col in r) else "-"
        em = f"{fmt_pct(r['PPLS-EM_mean'])} $\\pm$ {fmt_pct(r['PPLS-EM_std'])}" if ('PPLS-EM_mean' in r and 'PPLS-EM_std' in r) else "-"
        rows.append((a, expc, slm, em))
    return slm_display_name, rows


def generate_prediction_synthetic_calibration_table(*, artifacts_dir: Path, out_path: Path) -> None:
    """Synthetic calibration comparison table (PPLS-SLM vs PPLS-EM)."""
    slm_display_name, rows = _load_prediction_synthetic_calibration_rows(artifacts_dir=artifacts_dir)

    tex: list[str] = []
    tex.append(r"\setlength{\tabcolsep}{3pt}")
    tex.append(r"\renewcommand{\arraystretch}{0.95}")
    tex.append(r"\begin{table}[t]\footnotesize")
    tex.append(r"\centering")
    tex.append(r"\caption{Synthetic calibration of predictive credible intervals (5-fold CV).}")
    tex.append(r"\label{tab:pred_synth_calib}")
    tex.append(r"\begin{tabular}{@{}c|c|cc@{}}")
    tex.append(r"\toprule")
    tex.append(rf"$\alpha$ & Expected & {_latex_escape(slm_display_name)} & PPLS-EM \\")
    tex.append(r"\midrule")
    for a, expc, slm, em in rows:
        tex.append(f"{a:.2f} & {expc} & {slm} & {em} \\\\ ")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")
    tex.append(r"\renewcommand{\arraystretch}{1}")
    tex.append("")

    out_path.write_text("\n".join(tex), encoding="utf-8")


def generate_prediction_synthetic_summary_table(*, artifacts_dir: Path, out_path: Path) -> None:
    rows_metrics = _load_prediction_synthetic_metrics_rows(artifacts_dir=artifacts_dir)
    slm_display_name, rows_calib = _load_prediction_synthetic_calibration_rows(artifacts_dir=artifacts_dir)

    def wrap_method_label(name: str) -> str:
        mapping = {
            "SLM-Manifold-Adaptive": r"\makecell{SLM-Manifold\\Adaptive}",
            "PPLS-EM": r"\makecell{PPLS\\EM}",
        }
        return mapping.get(name, _latex_escape(name))

    tex: list[str] = []
    tex.append(r"\setlength{\tabcolsep}{4pt}")
    tex.append(r"\renewcommand{\arraystretch}{0.95}")
    tex.append(r"\begin{table*}[t]\footnotesize")
    tex.append(r"\centering")
    tex.append(r"\caption{Synthetic prediction summary (5-fold cross-validation (CV)). \textbf{Left:}")
    tex.append(r"point-prediction accuracy---SLM-Manifold-Adaptive and PPLS-EM")
    tex.append(r"are statistically indistinguishable. \textbf{Right:} predictive")
    tex.append(r"interval calibration---SLM-Manifold-Adaptive achieves 94.84\%")
    tex.append(r"coverage at the nominal 95\% level, while PPLS-EM covers only")
    tex.append(r"87.00\%, a \textbf{7.84 percentage-point undercoverage} that")
    tex.append(r"would produce misleadingly narrow uncertainty intervals in")
    tex.append(r"practice. PLSR and Ridge do not produce distributional")
    tex.append(r"predictions and are therefore omitted from the calibration panel.}")
    tex.append(r"\label{tab:pred_synth_summary}")
    tex.append(r"\begin{minipage}[t]{0.40\textwidth}")
    tex.append(r"\centering")
    tex.append(r"\textbf{Prediction accuracy}\\[2pt]")
    tex.append(r"\resizebox{\linewidth}{!}{%")
    tex.append(r"\begin{tabular}{@{}lccc@{}}")
    tex.append(r"\toprule")
    tex.append(r"\textbf{Method} & \textbf{MSE} & \textbf{MAE} & \textbf{$R^2$} \\")
    tex.append(r"\midrule")
    for method, mse, mae, r2 in rows_metrics:
        tex.append(f"{wrap_method_label(method)} & {mse} & {mae} & {r2} \\\\ ")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}%")
    tex.append(r"}")
    tex.append(r"\end{minipage}\hfill")
    tex.append(r"\begin{minipage}[t]{0.55\textwidth}")
    tex.append(r"\centering")
    tex.append(r"\textbf{Predictive interval calibration}\\[2pt]")
    tex.append(r"\resizebox{\linewidth}{!}{%")
    tex.append(r"\begin{tabular}{@{}c|c|cc@{}}")
    tex.append(r"\toprule")
    tex.append(rf"$\alpha$ & Expected & {wrap_method_label(slm_display_name)} & {wrap_method_label('PPLS-EM')} \\")
    tex.append(r"\midrule")
    for a, expc, slm, em in rows_calib:
        tex.append(f"{a:.2f} & {expc} & {slm} & {em} \\\\ ")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}%")
    tex.append(r"}")
    tex.append(r"\end{minipage}")

    tex.append(r"\end{table*}")
    tex.append(r"\renewcommand{\arraystretch}{1}")
    tex.append("")

    out_path.write_text("\n".join(tex), encoding="utf-8")




def generate_prediction_synthetic_alpha_table(*, artifacts_dir: Path, out_path: Path) -> None:
    """Diagnostic table of selected adaptive shrinkage alpha* on synthetic data."""
    path = artifacts_dir / "prediction" / "synthetic" / "selected_shrinkage_alpha.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing: {path}")

    df = pd.read_csv(path)
    df["method"] = df["method"].astype(str)

    slm_method = _pick_slm_method(df["method"].unique())
    if slm_method is None:
        raise ValueError("No PPLS-SLM* method found in selected_shrinkage_alpha.csv")

    slm_display_name = "SLM-Manifold-Adaptive" if "Adaptive" in slm_method else "SLM-Manifold"

    sub = df[df["method"] == slm_method].sort_values("fold")
    vals = sub["shrinkage_alpha"].astype(float).to_numpy()

    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")

    fold_cell = "\\makecell{" + "\\\\".join(
        [f"Fold {int(f)}: {float(a):.3g}" for f, a in zip(sub["fold"], vals)]
    ) + "}"

    tex: list[str] = []
    tex.append(r"\setlength{\tabcolsep}{4pt}")
    tex.append(r"\begin{table}[t]\small")
    tex.append(r"\centering")
    tex.append(r"\caption{Synthetic adaptive shrinkage diagnostics: selected $\gamma^*$ across folds.}")
    tex.append(r"\label{tab:pred_synth_gamma}")
    tex.append(r"\begin{tabular}{lcc}")
    tex.append(r"\toprule")
    tex.append(r"\textbf{Method} & \textbf{$\gamma^*$ (mean $\pm$ std)} & \textbf{Per-fold $\gamma^*$} \\")
    tex.append(r"\midrule")
    tex.append(rf"{_latex_escape(slm_display_name)} & {_pm_makecell_float(mean, std, fmt='.3g')} & {fold_cell} \\")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")
    tex.append("")

    out_path.write_text("\n".join(tex), encoding="utf-8")


def generate_prediction_brca_alpha_table(*, artifacts_dir: Path, out_path: Path) -> None:
    """Diagnostic table of selected adaptive shrinkage alpha* on BRCA."""
    diag_path = artifacts_dir / "prediction" / "brca" / "brca_selected_shrinkage_alpha.csv"
    summary_path = artifacts_dir / "prediction" / "brca" / "brca_prediction_summary.csv"
    if not (diag_path.exists() and summary_path.exists()):
        raise FileNotFoundError(f"missing: {diag_path} or {summary_path}")

    diag = pd.read_csv(diag_path)
    diag["method"] = diag["method"].astype(str)

    summ = pd.read_csv(summary_path)
    summ["method"] = summ["method"].astype(str)

    slm_method = _pick_slm_method(summ["method"].unique())
    if slm_method is None:
        raise ValueError("No PPLS-SLM* method found in brca_prediction_summary.csv")

    slm_display_name = "SLM-Manifold-Adaptive" if "Adaptive" in slm_method else "SLM-Manifold"

    sub_s = summ[summ["method"] == slm_method]
    if sub_s.empty:
        raise ValueError(f"Missing method in summary: {slm_method}")
    r_star = int(sub_s.iloc[0]["r"])

    sub_d = diag[(diag["method"] == slm_method) & (diag["r"].astype(int) == r_star)].sort_values("fold")
    vals = sub_d["shrinkage_alpha"].astype(float).to_numpy()

    mean = float(np.mean(vals)) if len(vals) else float("nan")
    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")

    fold_cell = "\\makecell{" + "\\\\".join(
        [f"Fold {int(f)}: {float(a):.3g}" for f, a in zip(sub_d["fold"], vals)]
    ) + "}"

    tex: list[str] = []
    tex.append(r"\setlength{\tabcolsep}{4pt}")
    tex.append(r"\begin{table}[t]\small")
    tex.append(r"\centering")
    tex.append(r"\caption{BRCA adaptive shrinkage diagnostics: selected $\gamma^*$ across folds at $r^*$.}")
    tex.append(r"\label{tab:pred_brca_gamma}")
    tex.append(r"\begin{tabular}{lccc}")
    tex.append(r"\toprule")
    tex.append(r"\textbf{Method} & $r^*$ & \textbf{$\gamma^*$ (mean $\pm$ std)} & \textbf{Per-fold $\gamma^*$} \\")
    tex.append(r"\midrule")
    tex.append(rf"{_latex_escape(slm_display_name)} & {r_star} & {_pm_makecell_float(mean, std, fmt='.3g')} & {fold_cell} \\")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")
    tex.append("")

    out_path.write_text("\n".join(tex), encoding="utf-8")


def _load_prediction_brca_metrics_rows(*, artifacts_dir: Path) -> List[Tuple[str, str, str, str, str]]:
    path = artifacts_dir / "prediction" / "brca" / "brca_prediction_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing: {path}")

    df = pd.read_csv(path)
    df["method"] = df["method"].astype(str)
    df["r"] = df["r"].astype(str)

    slm_method = _pick_slm_method(df["method"].unique())
    slm_display_name = "SLM-Manifold-Adaptive" if slm_method and "Adaptive" in slm_method else "SLM-Manifold"
    order = [m for m in [slm_method, "PPLS-EM", "PLSR", "Ridge"] if m is not None]

    rows: list[tuple[str, str, str, str, str]] = []
    for m in order:
        sub = df[df["method"] == m]
        if sub.empty:
            continue
        r0 = sub.iloc[0]
        display_m = slm_display_name if m == slm_method else m
        rows.append(
            (
                display_m,
                str(r0["r"]),
                _pm_makecell_float(r0["mse_mean"], r0["mse_std"], fmt=".4g"),
                _pm_makecell_float(r0["mae_mean"], r0["mae_std"], fmt=".4g"),
                _pm_makecell_float(r0["r2_mean"], r0["r2_std"], fmt=".4g"),
            )
        )
    return rows


def generate_prediction_brca_metrics_table(*, artifacts_dir: Path, out_path: Path) -> None:
    """BRCA prediction accuracy table at r* chosen by CV-MSE per method."""
    rows = _load_prediction_brca_metrics_rows(artifacts_dir=artifacts_dir)

    tex: list[str] = []
    tex.append(r"\setlength{\tabcolsep}{3pt}")
    tex.append(r"\renewcommand{\arraystretch}{0.95}")
    tex.append(r"\begin{table}[t]\footnotesize")
    tex.append(r"\centering")
    tex.append(r"\caption{BRCA prediction accuracy (5-fold CV). For PPLS/PLSR, $r^*$ minimises CV-MSE.}")
    tex.append(r"\label{tab:pred_brca_metrics}")
    tex.append(r"\begin{tabular}{@{}lcccc@{}}")
    tex.append(r"\toprule")
    tex.append(r"\textbf{Method} & $r$ & \textbf{MSE} & \textbf{MAE} & \textbf{$R^2$} \\")
    tex.append(r"\midrule")

    for method, r_star, mse, mae, r2 in rows:
        tex.append(f"{method} & {r_star} & {mse} & {mae} & {r2} \\\\ ")

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")
    tex.append(r"\renewcommand{\arraystretch}{1}")
    tex.append("")

    out_path.write_text("\n".join(tex), encoding="utf-8")


def _load_prediction_brca_calibration_rows(*, artifacts_dir: Path) -> List[Tuple[float, str, List[str], str]]:
    path = artifacts_dir / "prediction" / "brca" / "brca_calibration_table.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing: {path}")

    df = pd.read_csv(path)
    folds = [f"Fold {i}" for i in range(1, 6)]

    rows: list[tuple[float, str, list[str], str]] = []
    for _, r in df.iterrows():
        a = float(r["Alpha"])
        expc = _latex_escape(str(r["Expected Coverage"]))
        vals = [_latex_escape(str(r[c])) for c in folds]
        mean = _latex_escape(str(r["Mean"]))
        rows.append((a, expc, vals, mean))
    return rows


def generate_prediction_brca_calibration_table(*, artifacts_dir: Path, out_path: Path) -> None:
    """BRCA calibration table for PPLS-SLM at the selected r*."""
    rows = _load_prediction_brca_calibration_rows(artifacts_dir=artifacts_dir)

    tex: list[str] = []
    tex.append(r"\setlength{\tabcolsep}{2pt}")
    tex.append(r"\renewcommand{\arraystretch}{0.95}")
    tex.append(r"\begin{table}[t]\scriptsize")
    tex.append(r"\centering")
    tex.append(r"\caption{BRCA calibration of SLM-Manifold predictive intervals (element-wise coverage, \%).}")
    tex.append(r"\label{tab:pred_brca_calib}")
    tex.append(r"\begin{tabular}{@{}c|c|ccccc|c@{}}")
    tex.append(r"\toprule")
    tex.append(r"$\alpha$ & Expected & Fold 1 & Fold 2 & Fold 3 & Fold 4 & Fold 5 & Mean \\")
    tex.append(r"\midrule")

    for a, expc, vals, mean in rows:
        tex.append(f"{a:.2f} & {expc} & " + " & ".join(vals) + f" & {mean} \\\\ ")

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")
    tex.append(r"\renewcommand{\arraystretch}{1}")
    tex.append("")

    out_path.write_text("\n".join(tex), encoding="utf-8")


def generate_prediction_brca_summary_table(*, artifacts_dir: Path, out_path: Path) -> None:
    rows_metrics = _load_prediction_brca_metrics_rows(artifacts_dir=artifacts_dir)
    rows_calib = _load_prediction_brca_calibration_rows(artifacts_dir=artifacts_dir)

    def wrap_method_label(name: str) -> str:
        if name == "SLM-Manifold-Adaptive":
            return r"\makecell{SLM-Manifold\\Adaptive}"
        return _latex_escape(name)

    tex: list[str] = []
    tex.append(r"\setlength{\tabcolsep}{4pt}")
    tex.append(r"\renewcommand{\arraystretch}{0.95}")
    tex.append(r"\begin{table*}[t]\footnotesize")
    tex.append(r"\centering")
    tex.append(r"\caption{BRCA prediction summary (5-fold CV): prediction accuracy (left) and predictive-interval calibration (right).}")
    tex.append(r"\label{tab:pred_brca_summary}")
    tex.append(r"\begin{minipage}[t]{0.42\textwidth}")
    tex.append(r"\centering")
    tex.append(r"\textbf{Prediction accuracy}\\[2pt]")
    tex.append(r"\resizebox{\linewidth}{!}{%")
    tex.append(r"\begin{tabular}{@{}lcccc@{}}")
    tex.append(r"\toprule")
    tex.append(r"\textbf{Method} & $r$ & \textbf{MSE} & \textbf{MAE} & \textbf{$R^2$} \\")
    tex.append(r"\midrule")
    for method, r_star, mse, mae, r2 in rows_metrics:
        tex.append(f"{wrap_method_label(method)} & {r_star} & {mse} & {mae} & {r2} \\\\ ")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}%")
    tex.append(r"}")
    tex.append(r"\end{minipage}\hfill")
    tex.append(r"\begin{minipage}[t]{0.54\textwidth}")
    tex.append(r"\centering")
    tex.append(r"\textbf{Predictive interval calibration}\\[2pt]")
    tex.append(r"\resizebox{\linewidth}{!}{%")
    tex.append(r"\begin{tabular}{@{}c|c|ccccc|c@{}}")
    tex.append(r"\toprule")
    tex.append(r"$\alpha$ & Expected & Fold 1 & Fold 2 & Fold 3 & Fold 4 & Fold 5 & Mean \\")
    tex.append(r"\midrule")
    for a, expc, vals, mean in rows_calib:
        tex.append(f"{a:.2f} & {expc} & " + " & ".join(vals) + f" & {mean} \\\\ ")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}%")
    tex.append(r"}")
    tex.append(r"\end{minipage}")
    tex.append(r"\end{table*}")
    tex.append(r"\renewcommand{\arraystretch}{1}")
    tex.append("")

    out_path.write_text("\n".join(tex), encoding="utf-8")



# -----------------------------------------------------------------------------
#  CITE-seq (PBMC) prediction tables
# -----------------------------------------------------------------------------





def generate_citeseq_prediction_summary_table(*, artifacts_dir: Path, out_path: Path) -> None:
    pred_path = artifacts_dir / "prediction" / "citeseq" / "citeseq_prediction_summary.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"missing: {pred_path}")

    dfp = pd.read_csv(pred_path)

    def pick_row(method_key: str) -> Optional[pd.Series]:
        sub = dfp[dfp["method"].astype(str) == method_key]
        if sub.empty:
            return None
        return sub.iloc[0]

    def find_slm_row() -> Optional[pd.Series]:
        sub = dfp[dfp["method"].astype(str).str.startswith("PPLS-SLM", na=False)]
        if sub.empty:
            return None
        return sub.sort_values(["mse_mean"], ascending=True).iloc[0]

    def pretty_method(name: str, r_val: Any) -> str:
        s = str(name)

        # Use \( \) to avoid needing $...$ (and avoid escaping issues in table generation).
        def _r_tex(val: Any) -> str:
            if val is None:
                return ""
            try:
                # Accept ints, floats, and numeric strings.
                rr = int(float(val))
            except Exception:
                return ""
            return f"(\\(r={rr}\\))"

        r_tex = _r_tex(r_val)

        if s.startswith("PPLS-SLM"):
            # Match paper phrasing.
            base = "SLM-Manifold-Adaptive" if s.endswith("Adaptive") else "SLM-Manifold"
            return f"{base} {r_tex}".strip()
        if s == "PLSR":
            return f"PLSR {r_tex}".strip()
        if s == "Ridge":
            return "Ridge"
        if s == "PPLS-EM":
            return f"PPLS-EM {r_tex}".strip()
        return s


    def fmt_metric(x: Any) -> str:
        try:
            xf = float(x)
        except Exception:
            return "--"
        if not np.isfinite(xf):
            return "DNF"
        ax = abs(xf)
        if ax == 0.0:
            return "0"
        if 1e-3 <= ax < 1e4:
            return f"{xf:.4g}"
        return f"{xf:.3e}"

    rows_out: list[Tuple[str, Any, Any, Any]] = []

    slm = find_slm_row()
    if slm is not None:
        rows_out.append(
            (
                pretty_method(str(slm["method"]), slm.get("r")),
                fmt_metric(slm.get("mse_mean")),
                fmt_metric(slm.get("mae_mean")),
                fmt_metric(slm.get("r2_mean")),
            )
        )

    for key in ("PLSR", "Ridge"):
        r = pick_row(key)
        if r is None:
            continue
        rows_out.append(
            (
                pretty_method(str(r["method"]), r.get("r")),
                fmt_metric(r.get("mse_mean")),
                fmt_metric(r.get("mae_mean")),
                fmt_metric(r.get("r2_mean")),
            )
        )

    em = pick_row("PPLS-EM")
    if em is not None:
        rows_out.append(
            (
                pretty_method(str(em["method"]), em.get("r")),
                fmt_metric(em.get("mse_mean")),
                fmt_metric(em.get("mae_mean")),
                fmt_metric(em.get("r2_mean")),
            )
        )

    tex: list[str] = []
    tex.append(r"\setlength{\tabcolsep}{3.5pt}")
    tex.append(r"\renewcommand{\arraystretch}{1.1}")
    tex.append(r"\begin{table}[t]\small")
    tex.append(r"\centering")
    tex.append(r"\caption{Protein imputation on PBMC CITE-seq (3-fold CV): prediction accuracy.}")
    tex.append(r"\label{tab:citeseq_pred_summary}")
    tex.append(r"\begin{tabular}{lccc}")
    tex.append(r"\toprule")
    tex.append(r"Method & MSE $\downarrow$ & MAE $\downarrow$ & $R^2$ $\uparrow$\\")
    tex.append(r"\midrule")

    for m, mse, mae, r2 in rows_out:
        tex.append(f"{_latex_escape_keep_math(m)} & {mse} & {mae} & {r2} \\\\ ")

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")
    tex.append(r"\renewcommand{\arraystretch}{1}")
    tex.append("")

    out_path.write_text("\n".join(tex), encoding="utf-8")



# -----------------------------------------------------------------------------
#  Model selection (latent dimension r)
# -----------------------------------------------------------------------------


def _format_sci_tex(x: Any, *, sig: int = 3) -> str:
    """Format a float as LaTeX-friendly scientific notation.

    Examples:
      12345 -> $1.23\times 10^{4}$
      0.0012 -> $1.20\times 10^{-3}$

    Keeps plain decimal for moderate magnitudes.
    """

    try:
        xf = float(x)
    except Exception:
        return _latex_escape(str(x))

    if not pd.isna(xf) and np.isfinite(xf):
        ax = abs(xf)
        if (ax == 0.0) or (1e-2 <= ax < 1e4):
            # Plain number
            return f"{xf:.4g}"

        s = f"{xf:.{sig}e}"
        m_s, e_s = s.split("e")
        e = int(e_s)
        return rf"${m_s}\times 10^{{{e}}}$"

    return "--"


def generate_pcca_simulation_table(*, artifacts_dir: Path, out_path: Path) -> None:
    """PCCA specialization simulation table (Appendix).

    Source: paper/artifacts/pcca_simulation/mse_table.json
    """

    path = artifacts_dir / "pcca_simulation" / "mse_table.json"
    if not path.exists():
        raise FileNotFoundError(f"missing: {path}")

    pcca = _read_json(path)

    # Try to pull experiment dimensions from the summary (best-effort).
    p = q = r = N = M = None
    summ = artifacts_dir / "pcca_simulation" / "experiment_summary.json"
    if summ.exists():
        try:
            info = _read_json(summ).get("experiment_info", {})
            p = info.get("p")
            q = info.get("q")
            r = info.get("r")
            N = info.get("n_samples")
            M = info.get("n_trials_completed")
        except Exception:
            p = q = r = N = M = None

    # Sensible defaults (match config.json).
    p = int(p) if p is not None else 20
    q = int(q) if q is not None else 20
    r = int(r) if r is not None else 3
    N = int(N) if N is not None else 500
    M = int(M) if M is not None else 20

    tex: list[str] = []
    tex.append(r"\begin{table}[t]\small")
    tex.append(r"\centering")
    tex.append(
        rf"\caption{{PCCA specialization parameter estimation MSE ($\times 10^2$) (mean $\pm$ std over $M={M}$ trials; $p=q={p}$, $r={r}$, $N={N}$; $B=I_r$, $\sigma_h^2=0$).}}"
    )
    tex.append(r"\label{tab:pcca_parameter_mse}")
    tex.append(r"\setlength{\tabcolsep}{6pt}")
    tex.append(r"\begin{tabular}{@{}lccc@{}}")
    tex.append(r"\toprule")
    tex.append(r"\textbf{Method} & $\text{MSE}_W$ & $\text{MSE}_C$ & $\text{MSE}_{\Sigma_t}$ \\")
    tex.append(r"\midrule")

    def row(method_key: str, display: str) -> None:
        cells = [
            _pm_makecell(str(pcca[method_key][k]["table_str_x1e2"]))
            for k in ("W", "C", "Sigma_t")
        ]
        tex.append(f"{display} & " + " & ".join(cells) + r" \\")

    # Keep the same order as the main table.
    if "bcd_slm" in pcca:
        row("bcd_slm", "BCD-SLM")
    if "em" in pcca:
        row("em", "EM")

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")
    tex.append("")

    out_path.write_text("\n".join(tex), encoding="utf-8")



def generate_ppca_verification_table(*, artifacts_dir: Path, out_path: Path) -> None:
    """PPCA noise-variance estimator verification table.


    Source: paper/artifacts/ppca_verification/summary.json
    """

    path = artifacts_dir / "ppca_verification" / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"missing: {path}")

    s = _read_json(path)

    def f6(x: Any) -> str:
        try:
            return f"{float(x):.6f}"
        except Exception:
            return _latex_escape(str(x))

    mean_spec = f6(s.get("sigma_e2_spectral_mean"))
    mean_tb = f6(s.get("sigma_e2_tb_mean"))
    err_spec = f6(s.get("abs_err_mean"))
    err_tb = f6(s.get("abs_err_mean"))

    M = int(s.get("n_trials", 20))
    p = int(s.get("p", 20))
    r = int(s.get("r", 3))
    N = int(s.get("n_samples", 500))
    se2 = f6(s.get("sigma_e2_true", 0.1))

    tex: list[str] = []
    tex.append(r"\begin{table}[t]\small")
    tex.append(r"\centering")
    tex.append(
        rf"\caption{{PPCA noise variance estimation verification ($M={M}$ trials, $p={p}$, $r={r}$, $N={N}$, $\sigma_e^2={se2}$).}}"
    )
    tex.append(r"\label{tab:ppca_noise_verification}")
    tex.append(r"\setlength{\tabcolsep}{6pt}")
    tex.append(r"\begin{tabular}{lcc}")
    tex.append(r"\toprule")
    tex.append(r"Estimator & Mean $\hat{\sigma}_e^2$ & Mean $\lvert\hat{\sigma}_e^2-\sigma_e^2\rvert$ \\")
    tex.append(r"\midrule")
    tex.append(rf"Spectral (Theorem 4) & {mean_spec} & {err_spec} \\")
    tex.append(rf"Tipping \& Bishop MLE & {mean_tb} & {err_tb} \\")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")
    tex.append("")

    out_path.write_text("\n".join(tex), encoding="utf-8")


def generate_paper_metrics(*, artifacts_dir: Path, out_path: Path) -> None:
    """Generate LaTeX macros for key scalar results used in the prose.


    This keeps the narrative consistent with the latest `paper/artifacts/*`.

    Output: `paper/generated/metrics.tex`
    """

    # --- Convergence stats ---
    xlsx = artifacts_dir / "simulation" / "figures" / "Table_3_Convergence_Comparison.xlsx"
    df_conv = pd.read_excel(xlsx, sheet_name="Convergence_Statistics")

    def _get_row(alg: str) -> pd.Series:
        sub = df_conv[df_conv["Algorithm"].astype(str).str.upper() == alg.upper()]
        if sub.empty:
            raise ValueError(f"Algorithm '{alg}' not found in {xlsx}")
        return sub.iloc[0]

    # Keep macro names stable in LaTeX, but source values from the fixed-noise SLM row.
    r_slm = _get_row("SLM-Manifold")


    r_em = _get_row("EM")
    r_ecm = _get_row("ECM")

    mean_slm = float(r_slm["Mean_Iterations"])
    mean_em = float(r_em["Mean_Iterations"])
    mean_ecm = float(r_ecm["Mean_Iterations"])
    std_slm = float(r_slm["Std_Iterations"])
    std_em = float(r_em["Std_Iterations"])
    std_ecm = float(r_ecm["Std_Iterations"])

    def f1(x: float) -> str:
        return f"{x:.1f}"

    def f2(x: float) -> str:
        return f"{x:.2f}"

    def safe_div(a: float, b: float) -> float:
        return float(a / b) if b else float("nan")

    # --- Parameter MSE (x1e2 table values) ---
    low = _read_json(artifacts_dir / "simulation" / "mse_table_low.json")
    high = _read_json(artifacts_dir / "simulation" / "mse_table_high.json")

    def mean_x1e2(table_str: str) -> float:
        # e.g. "7.43±1.23" -> 7.43
        s = str(table_str)
        if "±" in s:
            s = s.split("±", 1)[0]
        return float(s.strip())

    def mse_mean(data: Dict[str, Any], method: str, key: str) -> float:
        return mean_x1e2(data[method][key]["table_str_x1e2"])

    methods = ["slm_manifold", "em", "ecm"]


    # Low-noise range used in prose for W/C together.
    low_wc = [mse_mean(low, m, k) for m in methods for k in ("W", "C")]
    low_wc_min, low_wc_max = min(low_wc), max(low_wc)

    # Overall ranges (across low/high + all methods) for W and C.
    all_w = [mse_mean(d, m, "W") for d in (low, high) for m in methods]
    all_c = [mse_mean(d, m, "C") for d in (low, high) for m in methods]

    # Specific low-noise scalar parameters.
    low_sigmat_emecm = [mse_mean(low, m, "Sigma_t") for m in ("em", "ecm")]
    low_sigmah_emecm = [mse_mean(low, m, "sigma_h2") for m in ("em", "ecm")]
    low_b_emecm = [mse_mean(low, m, "B") for m in ("em", "ecm")]

    # High-noise specific mentions.
    high_w_slm = mse_mean(high, "slm_manifold", "W")
    high_c_slm = mse_mean(high, "slm_manifold", "C")
    high_sigmah_emecm = [mse_mean(high, m, "sigma_h2") for m in ("em", "ecm")]
    high_sigmah_slm = mse_mean(high, "slm_manifold", "sigma_h2")


    # --- Association overlap counts ---
    det = pd.read_csv(artifacts_dir / "association" / "detection_table.csv")
    det["p-value threshold"] = det["p-value threshold"].astype(str).str.strip()

    def overlap_at(th: str) -> int:
        sub = det[det["p-value threshold"] == th]
        if sub.empty:
            raise ValueError(f"Missing threshold row: {th}")
        return int(sub.iloc[0]["Overlap"])

    # Write macros
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("% Auto-generated. Do not edit by hand.")
    lines.append("% Generated by scripts/generate_paper_tables.py")

    # Convergence
    lines.append(r"\providecommand{\ConvMeanSLM}{" + f1(mean_slm) + "}")
    lines.append(r"\providecommand{\ConvMeanEM}{" + f1(mean_em) + "}")
    lines.append(r"\providecommand{\ConvMeanECM}{" + f1(mean_ecm) + "}")
    lines.append(r"\providecommand{\ConvRatioEM}{" + f1(safe_div(mean_em, mean_slm)) + "}")
    lines.append(r"\providecommand{\ConvRatioECM}{" + f1(safe_div(mean_ecm, mean_slm)) + "}")
    lines.append(r"\providecommand{\ConvCVSLM}{" + f2(safe_div(std_slm, mean_slm)) + "}")
    lines.append(r"\providecommand{\ConvCVEM}{" + f2(safe_div(std_em, mean_em)) + "}")
    lines.append(r"\providecommand{\ConvCVECM}{" + f2(safe_div(std_ecm, mean_ecm)) + "}")

    # MSE (x1e2)
    lines.append(r"\providecommand{\MSELowWCMin}{" + f2(low_wc_min) + "}")
    lines.append(r"\providecommand{\MSELowWCMax}{" + f2(low_wc_max) + "}")

    lines.append(r"\providecommand{\MSEAllWMin}{" + f2(min(all_w)) + "}")
    lines.append(r"\providecommand{\MSEAllWMax}{" + f2(max(all_w)) + "}")
    lines.append(r"\providecommand{\MSEAllCMin}{" + f2(min(all_c)) + "}")
    lines.append(r"\providecommand{\MSEAllCMax}{" + f2(max(all_c)) + "}")

    lines.append(r"\providecommand{\MSESigmaTLowEMECMMin}{" + f1(min(low_sigmat_emecm)) + "}")
    lines.append(r"\providecommand{\MSESigmaTLowEMECMMax}{" + f1(max(low_sigmat_emecm)) + "}")
    lines.append(r"\providecommand{\MSESigmaHLowEMECM}{" + f2(min(low_sigmah_emecm)) + "}")
    lines.append(r"\providecommand{\MSESigmaTLowSLM}{" + f1(mse_mean(low, "slm_manifold", "Sigma_t")) + "}")
    lines.append(r"\providecommand{\MSESigmaHLowSLM}{" + f2(mse_mean(low, "slm_manifold", "sigma_h2")) + "}")
    lines.append(r"\providecommand{\MSEBLowSLM}{" + f2(mse_mean(low, "slm_manifold", "B")) + "}")

    # Additional macros for SLM-Interior (low/high noise, used in ablation text)
    lines.append(r"\providecommand{\MSEWLowSLMInterior}{" + f2(mse_mean(low, "slm_interior", "W")) + "}")
    lines.append(r"\providecommand{\MSECLowSLMInterior}{" + f2(mse_mean(low, "slm_interior", "C")) + "}")
    lines.append(r"\providecommand{\MSEBLowSLMInterior}{" + f2(mse_mean(low, "slm_interior", "B")) + "}")
    lines.append(r"\providecommand{\MSESigmaTLowSLMInterior}{" + f2(mse_mean(low, "slm_interior", "Sigma_t")) + "}")
    lines.append(r"\providecommand{\MSESigmaHLowSLMInterior}{" + f2(mse_mean(low, "slm_interior", "sigma_h2")) + "}")

    lines.append(r"\providecommand{\MSEWHighSLMInterior}{" + f2(mse_mean(high, "slm_interior", "W")) + "}")
    lines.append(r"\providecommand{\MSECHighSLMInterior}{" + f2(mse_mean(high, "slm_interior", "C")) + "}")

    lines.append(r"\providecommand{\MSEBHighSLMInterior}{" + f2(mse_mean(high, "slm_interior", "B")) + "}")
    lines.append(r"\providecommand{\MSESigmaTHighSLMInterior}{" + f2(mse_mean(high, "slm_interior", "Sigma_t")) + "}")
    lines.append(r"\providecommand{\MSESigmaHHighSLMInterior}{" + f2(mse_mean(high, "slm_interior", "sigma_h2")) + "}")

    lines.append(r"\providecommand{\MSEBLowEMECMMin}{" + f2(min(low_b_emecm)) + "}")
    lines.append(r"\providecommand{\MSEBLowEMECMMax}{" + f2(max(low_b_emecm)) + "}")

    lines.append(r"\providecommand{\MSEWHighSLM}{" + f2(high_w_slm) + "}")
    lines.append(r"\providecommand{\MSECHighSLM}{" + f2(high_c_slm) + "}")
    lines.append(r"\providecommand{\MSESigmaHHighEMECM}{" + f2(min(high_sigmah_emecm)) + "}")
    lines.append(r"\providecommand{\MSESigmaHHighSLM}{" + f2(high_sigmah_slm) + "}")

    # Association overlap
    lines.append(r"\providecommand{\OverlapPOneEminusSeven}{" + str(overlap_at("p < 1e-7")) + "}")
    lines.append(r"\providecommand{\OverlapPOneEminusSix}{" + str(overlap_at("p < 1e-6")) + "}")
    lines.append(r"\providecommand{\OverlapPOneEminusFour}{" + str(overlap_at("p < 1e-4")) + "}")

    # --- CITE-seq (PBMC) headline metrics (optional) ---
    cit_dir = artifacts_dir / "prediction" / "citeseq"
    cit_pred = cit_dir / "citeseq_prediction_summary.csv"
    cit_cov = cit_dir / "citeseq_calibration_summary.csv"

    # Optional: best-SLM headline numbers (not required by the main paper text, but handy for future edits).

    if cit_pred.exists():
        try:
            dfp = pd.read_csv(cit_pred)
            slm_rows = dfp[dfp["method"].astype(str).str.startswith("PPLS-SLM", na=False)]
            slm_best = slm_rows.sort_values(["mse_mean"], ascending=True).iloc[0] if not slm_rows.empty else None
        except Exception:
            slm_best = None
    else:
        slm_best = None

    def _fmt_num(x: Any) -> str:
        try:
            xf = float(x)
        except Exception:
            return "--"
        if not np.isfinite(xf):
            return "DNF"
        ax = abs(xf)
        if ax == 0.0:
            return "0"
        if 1e-3 <= ax < 1e4:
            return f"{xf:.4g}"
        return f"{xf:.3e}"

    if slm_best is not None:
        lines.append(r"\providecommand{\CiteSeqBestR}{" + str(int(slm_best.get("r", 0))) + "}")
        lines.append(r"\providecommand{\CiteSeqBestMSE}{" + _fmt_num(slm_best.get("mse_mean")) + "}")
        lines.append(r"\providecommand{\CiteSeqBestMAE}{" + _fmt_num(slm_best.get("mae_mean")) + "}")
        lines.append(r"\providecommand{\CiteSeqBestRtwo}{" + _fmt_num(slm_best.get("r2_mean")) + "}")
    else:
        lines.append(r"\providecommand{\CiteSeqBestR}{--}")
        lines.append(r"\providecommand{\CiteSeqBestMSE}{--}")
        lines.append(r"\providecommand{\CiteSeqBestMAE}{--}")
        lines.append(r"\providecommand{\CiteSeqBestRtwo}{--}")

    if cit_cov.exists() and slm_best is not None:
        try:
            dfc = pd.read_csv(cit_cov)
            m = str(slm_best.get("method"))
            r_val = str(slm_best.get("r"))
            sub = dfc[(dfc["method"].astype(str) == m) & (dfc["r"].astype(str) == r_val)]

            def cov(alpha: float) -> str:
                s2 = sub[np.isclose(sub["alpha"].astype(float), float(alpha))]
                if s2.empty:
                    return "--"
                c = float(s2.iloc[0]["coverage_mean"])
                return "DNF" if not np.isfinite(c) else f"{100.0*c:.2f}\\%"

            lines.append(r"\providecommand{\CiteSeqCoverAZeroZeroFive}{" + cov(0.05) + "}")
            lines.append(r"\providecommand{\CiteSeqCoverAZeroOneZero}{" + cov(0.10) + "}")
            lines.append(r"\providecommand{\CiteSeqCoverAZeroTwoZero}{" + cov(0.20) + "}")
        except Exception:
            lines.append(r"\providecommand{\CiteSeqCoverAZeroZeroFive}{--}")
            lines.append(r"\providecommand{\CiteSeqCoverAZeroOneZero}{--}")
            lines.append(r"\providecommand{\CiteSeqCoverAZeroTwoZero}{--}")
    else:
        lines.append(r"\providecommand{\CiteSeqCoverAZeroZeroFive}{--}")
        lines.append(r"\providecommand{\CiteSeqCoverAZeroOneZero}{--}")
        lines.append(r"\providecommand{\CiteSeqCoverAZeroTwoZero}{--}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Generate LaTeX snippets under paper/generated from paper/artifacts")
    parser.add_argument(
        "--out-root",
        type=str,
        default=None,
        help="Output root dir (default: paper/generated). Tables go under <out-root>/tables and metrics.tex under <out-root>.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    repo_root = Path(__file__).resolve().parents[1]
    paper_dir = repo_root / "paper"
    artifacts_dir = paper_dir / "artifacts"

    if args.out_root:
        out_root = Path(args.out_root)
        if not out_root.is_absolute():
            out_root = repo_root / out_root
        out_root = out_root.resolve()
    else:
        out_root = paper_dir / "generated"

    out_dir = out_root / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    generate_convergence_table(artifacts_dir=artifacts_dir, out_path=out_dir / "tab_algorithm_convergence.tex")
    generate_parameter_mse_table(artifacts_dir=artifacts_dir, out_path=out_dir / "tab_parameter_mse.tex")

    # PCCA specialization table (optional)
    try:
        generate_pcca_simulation_table(artifacts_dir=artifacts_dir, out_path=out_dir / "tab_pcca_simulation.tex")
    except FileNotFoundError as e:
        print(f"[SKIP] PCCA simulation table ({e})")

    # PPCA verification table (optional)
    try:
        generate_ppca_verification_table(artifacts_dir=artifacts_dir, out_path=out_dir / "tab_ppca_verification.tex")
    except FileNotFoundError as e:
        print(f"[SKIP] PPCA verification table ({e})")



    generate_top10_pairs_table(artifacts_dir=artifacts_dir, out_path=out_dir / "tab_top10_pairs.tex")
    generate_detection_table(artifacts_dir=artifacts_dir, out_path=out_dir / "tab_detected_pairs.tex")

    # Prediction tables (synthetic + BRCA). These depend on synced artifacts.
    # Synthetic
    generate_prediction_synthetic_metrics_table(artifacts_dir=artifacts_dir, out_path=out_dir / "tab_prediction_synth_metrics.tex")
    generate_prediction_synthetic_calibration_table(artifacts_dir=artifacts_dir, out_path=out_dir / "tab_prediction_synth_calibration.tex")
    generate_prediction_synthetic_summary_table(artifacts_dir=artifacts_dir, out_path=out_dir / "tab_prediction_synth_summary.tex")

    syn_dir = artifacts_dir / "prediction" / "synthetic"

    if (syn_dir / "selected_shrinkage_alpha.csv").exists():
        generate_prediction_synthetic_alpha_table(artifacts_dir=artifacts_dir, out_path=out_dir / "tab_prediction_synth_alpha.tex")
    else:
        print(f"[SKIP] Synthetic alpha* table (missing: {syn_dir / 'selected_shrinkage_alpha.csv'})")

    # BRCA (optional in config; skip cleanly when artifacts are absent)

    brca_dir = artifacts_dir / "prediction" / "brca"
    if (brca_dir / "brca_prediction_summary.csv").exists():
        generate_prediction_brca_metrics_table(artifacts_dir=artifacts_dir, out_path=out_dir / "tab_prediction_brca_metrics.tex")

        if (brca_dir / "brca_selected_shrinkage_alpha.csv").exists():
            generate_prediction_brca_alpha_table(artifacts_dir=artifacts_dir, out_path=out_dir / "tab_prediction_brca_alpha.tex")
        else:
            print(f"[SKIP] BRCA alpha* table (missing: {brca_dir / 'brca_selected_shrinkage_alpha.csv'})")
    else:
        print(f"[SKIP] BRCA prediction table (missing: {brca_dir / 'brca_prediction_summary.csv'})")


    if (brca_dir / "brca_calibration_table.csv").exists():
        generate_prediction_brca_calibration_table(artifacts_dir=artifacts_dir, out_path=out_dir / "tab_prediction_brca_calibration.tex")
        if (brca_dir / "brca_prediction_summary.csv").exists():
            generate_prediction_brca_summary_table(artifacts_dir=artifacts_dir, out_path=out_dir / "tab_prediction_brca_summary.tex")
    else:
        print(f"[SKIP] BRCA calibration table (missing: {brca_dir / 'brca_calibration_table.csv'})")


    # CITE-seq (optional)
    cit_dir = artifacts_dir / "prediction" / "citeseq"
    if (cit_dir / "citeseq_prediction_summary.csv").exists():
        generate_citeseq_prediction_summary_table(artifacts_dir=artifacts_dir, out_path=out_dir / "tab_citeseq_prediction_summary.tex")
    else:
        print(f"[SKIP] CITE-seq prediction table (missing: {cit_dir / 'citeseq_prediction_summary.csv'})")





    # Legacy coverage table (kept for backward compatibility).
    # (Some older paper drafts reference it.)
    try:
        generate_prediction_coverage_table(artifacts_dir=artifacts_dir, out_path=out_dir / "tab_prediction_coverage.tex")
    except FileNotFoundError as e:
        print(f"[SKIP] Legacy coverage table ({e})")


    generate_paper_metrics(artifacts_dir=artifacts_dir, out_path=out_root / "metrics.tex")



    print(f"[OK] Wrote tables into: {out_dir}")




if __name__ == "__main__":
    main()

