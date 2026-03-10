"""Large-scale synthetic parameter recovery experiment for PPLS.

This entry point runs a reproducible parameter-recovery study over a grid of
larger \\(p, q, r\\) configurations using random orthogonal loading matrices.
It is designed for the manuscript's scale-validation experiment where
SLM-Fixed / SLM-Oracle are compared against EM / ECM.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import multiprocessing
import os
import platform
import socket
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from ppls_slm.algorithms import ECMAlgorithm, EMAlgorithm, InitialPointGenerator, ScalarLikelihoodMethod
from ppls_slm.bcd_slm import BCDScalarLikelihoodMethod

from ppls_slm.data_generator import RandomOrthogonalDataGenerator
from ppls_slm.experiment_config import ConfigError, load_config
from ppls_slm.utils import ensure_dir, repo_root, setup_logging


METHOD_ORDER = ["slm_fixed", "bcd_slm", "slm_oracle", "em", "ecm"]
METHOD_DISPLAY = {
    "slm_fixed": "SLM-Fixed",
    "bcd_slm": "BCD-SLM",
    "slm_oracle": "SLM-Oracle",
    "em": "EM",
    "ecm": "ECM",
}

METRIC_KEYS = ["mse_W", "mse_C", "mse_B", "mse_Sigma_t", "mse_sigma_h2"]
NOISE_ORDER = ["low", "high"]
_WORKER_CTX: Dict[str, Any] = {}


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _safe_corr_sign(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 0:
        return 1.0
    val = float(np.dot(x, y) / denom)
    return -1.0 if val < 0 else 1.0


def align_estimates(params_est: Dict[str, Any], params_true: Dict[str, Any]) -> Dict[str, Any]:
    aligned = {
        key: (np.array(value, copy=True) if isinstance(value, np.ndarray) else copy.deepcopy(value))
        for key, value in params_est.items()
    }

    W_est = np.array(aligned["W"], copy=True)
    C_est = np.array(aligned["C"], copy=True)
    B_diag = np.diag(np.array(aligned["B"], copy=True)).astype(float)

    W_true = np.asarray(params_true["W"], dtype=float)
    C_true = np.asarray(params_true["C"], dtype=float)
    r = W_true.shape[1]

    for idx in range(r):
        W_est[:, idx] *= _safe_corr_sign(W_est[:, idx], W_true[:, idx])
        C_est[:, idx] *= _safe_corr_sign(C_est[:, idx], C_true[:, idx])

    signs = np.sign(B_diag)
    signs[signs == 0] = 1.0
    for idx in range(r):
        if signs[idx] < 0:
            C_est[:, idx] *= -1.0
    aligned["W"] = W_est
    aligned["C"] = C_est
    aligned["B"] = np.diag(np.abs(B_diag))
    return aligned


def compute_parameter_mse(params_est: Dict[str, Any], params_true: Dict[str, Any]) -> Dict[str, float]:
    aligned = align_estimates(params_est, params_true)
    p, q, r = int(params_true["p"]), int(params_true["q"]), int(params_true["r"])

    return {
        "mse_W": float(np.linalg.norm(aligned["W"] - params_true["W"], ord="fro") ** 2 / (p * r)),
        "mse_C": float(np.linalg.norm(aligned["C"] - params_true["C"], ord="fro") ** 2 / (q * r)),
        "mse_B": float(np.linalg.norm(np.diag(aligned["B"]) - np.diag(params_true["B"])) ** 2 / r),
        "mse_Sigma_t": float(
            np.linalg.norm(np.diag(aligned["Sigma_t"]) - np.diag(params_true["Sigma_t"])) ** 2 / r
        ),
        "mse_sigma_h2": float((float(aligned["sigma_h2"]) - float(params_true["sigma_h2"])) ** 2),
    }


def _slm_kwargs(exp_cfg: Dict[str, Any], *, oracle: bool, true_params: Dict[str, Any]) -> Dict[str, Any]:
    kwargs = {
        "optimizer": str(exp_cfg.get("optimizer", "manifold")),
        "max_iter": int(exp_cfg["max_iter"]),
        "use_noise_preestimation": not oracle,
        "gtol": float(exp_cfg.get("slm_gtol", 5e-3)),
        "xtol": float(exp_cfg.get("slm_xtol", 5e-3)),
        "barrier_tol": float(exp_cfg.get("slm_barrier_tol", 5e-3)),
        "constraint_slack": float(exp_cfg.get("slm_constraint_slack", 5e-3)),
        "verbose": bool(exp_cfg.get("slm_verbose", False)),
        "progress_every": int(exp_cfg.get("slm_progress_every", 1)),
        "early_stop_patience": exp_cfg.get("slm_early_stop_patience"),
        "early_stop_rel_improvement": exp_cfg.get("slm_early_stop_rel_improvement"),
    }
    if oracle:
        kwargs["fixed_sigma_e2"] = float(true_params["sigma_e2"])
        kwargs["fixed_sigma_f2"] = float(true_params["sigma_f2"])
    return kwargs


def _instantiate_methods(condition: Dict[str, Any], true_params: Dict[str, Any]) -> Dict[str, Any]:
    common = {"p": int(condition["p"]), "q": int(condition["q"]), "r": int(condition["r"])}
    exp_cfg = condition["experiment_cfg"]
    return {
        "slm_fixed": ScalarLikelihoodMethod(**common, **_slm_kwargs(exp_cfg, oracle=False, true_params=true_params)),
        "bcd_slm": BCDScalarLikelihoodMethod(
            **common,
            max_outer_iter=int(exp_cfg["max_iter"]),
            n_cg_steps_W=int(exp_cfg.get("bcd_n_cg_steps_W", 5)),
            n_cg_steps_C=int(exp_cfg.get("bcd_n_cg_steps_C", 5)),
            tolerance=float(exp_cfg.get("bcd_tolerance", exp_cfg.get("slm_gtol", 5e-3))),
            use_noise_preestimation=True,
        ),
        "slm_oracle": ScalarLikelihoodMethod(**common, **_slm_kwargs(exp_cfg, oracle=True, true_params=true_params)),
        "em": EMAlgorithm(

            **common,
            max_iter=int(exp_cfg["max_iter"]),
            tolerance=float(exp_cfg.get("em_tolerance", 5e-3)),
        ),
        "ecm": ECMAlgorithm(
            **common,
            max_iter=int(exp_cfg["max_iter"]),
            tolerance=float(exp_cfg.get("ecm_tolerance", 5e-3)),
        ),
    }


def _method_success(method_name: str, result: Dict[str, Any]) -> bool:
    if method_name.startswith("slm") or method_name.startswith("bcd"):
        return bool(result.get("success", False))
    return np.isfinite(float(result.get("log_likelihood", np.nan)))



def _init_worker(condition: Dict[str, Any], true_params: Dict[str, Any], starting_points: List[np.ndarray]) -> None:
    global _WORKER_CTX
    _WORKER_CTX = {
        "condition": condition,
        "true_params": true_params,
        "starting_points": starting_points,
    }


def _run_trial_worker(trial_id: int, trial_seed: int) -> Dict[str, Any]:
    ctx = _WORKER_CTX
    condition = ctx["condition"]
    true_params = ctx["true_params"]
    starting_points = ctx["starting_points"]

    generator = RandomOrthogonalDataGenerator(
        p=int(condition["p"]),
        q=int(condition["q"]),
        r=int(condition["r"]),
        n_samples=int(condition["n_samples"]),
        random_seed=int(condition["true_param_seed"]),
        output_dir=str(condition["condition_dir"]),
    )
    X, Y = generator.generate_samples(true_params, seed=trial_seed)
    methods = _instantiate_methods(condition, true_params)

    trial_output: Dict[str, Any] = {
        "trial_id": int(trial_id),
        "trial_seed": int(trial_seed),
        "methods": {},
    }

    for method_name in METHOD_ORDER:
        estimator = methods[method_name]
        t0 = time.perf_counter()
        record: Dict[str, Any] = {
            "method": method_name,
            "success": False,
            "runtime_sec": np.nan,
            "n_iterations": np.nan,
            "error": None,
        }
        for metric_key in METRIC_KEYS:
            record[metric_key] = np.nan

        try:
            result = estimator.fit(X, Y, starting_points)
            record["success"] = _method_success(method_name, result)
            record["n_iterations"] = float(result.get("n_iterations", np.nan))
            if method_name.startswith("slm") or method_name.startswith("bcd"):
                record["objective_or_ll"] = float(result.get("objective_value", np.nan))
            else:
                record["objective_or_ll"] = float(result.get("log_likelihood", np.nan))

            metrics = compute_parameter_mse(result, true_params)
            record.update(metrics)
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["objective_or_ll"] = np.nan
        finally:
            record["runtime_sec"] = float(time.perf_counter() - t0)

        trial_output["methods"][method_name] = record

    return trial_output


def _flatten_trial_outputs(condition: Dict[str, Any], trial_outputs: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for trial in trial_outputs:
        for method_name in METHOD_ORDER:
            rec = dict(trial["methods"][method_name])
            rec.update(
                {
                    "trial_id": int(trial["trial_id"]),
                    "trial_seed": int(trial["trial_seed"]),
                    "config_id": int(condition["config_id"]),
                    "config_name": condition["config_name"],
                    "config_label": condition["config_label"],
                    "p": int(condition["p"]),
                    "q": int(condition["q"]),
                    "r": int(condition["r"]),
                    "n_samples": int(condition["n_samples"]),
                    "noise": condition["noise_name"],
                    "method_display": METHOD_DISPLAY[method_name],
                }
            )
            rows.append(rec)
    return pd.DataFrame(rows)


def _summarize_method_rows(condition: Dict[str, Any], df: pd.DataFrame, condition_runtime_sec: float) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    summary_rows: List[Dict[str, Any]] = []

    for method_name in METHOD_ORDER:
        sub = df[df["method"] == method_name].copy()
        summary: Dict[str, Any] = {
            "config_id": int(condition["config_id"]),
            "config_name": condition["config_name"],
            "config_label": condition["config_label"],
            "p": int(condition["p"]),
            "q": int(condition["q"]),
            "r": int(condition["r"]),
            "n_samples": int(condition["n_samples"]),
            "noise": condition["noise_name"],
            "method": method_name,
            "method_display": METHOD_DISPLAY[method_name],
            "n_trials": int(len(sub)),
            "n_success": int(sub["success"].fillna(False).astype(bool).sum()),
            "success_rate": float(sub["success"].fillna(False).astype(bool).mean()) if len(sub) else float("nan"),
            "avg_runtime_sec": float(sub["runtime_sec"].mean()) if len(sub) else float("nan"),
            "std_runtime_sec": float(sub["runtime_sec"].std(ddof=0)) if len(sub) else float("nan"),
            "avg_iterations": float(sub["n_iterations"].mean()) if len(sub) else float("nan"),
            "condition_runtime_sec": float(condition_runtime_sec),
        }
        for metric in METRIC_KEYS:
            values = sub[metric].astype(float)
            mean = float(values.mean()) if len(values) else float("nan")
            std = float(values.std(ddof=0)) if len(values) else float("nan")
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = std
            summary[f"{metric}_mean_x1e2"] = 100.0 * mean
            summary[f"{metric}_std_x1e2"] = 100.0 * std
            summary[f"{metric}_table_str_x1e2"] = f"{100.0 * mean:.2f}±{100.0 * std:.2f}"
        summary_rows.append(summary)

    summary_df = pd.DataFrame(summary_rows)
    condition_summary = {
        "condition": {
            "config_id": int(condition["config_id"]),
            "config_name": condition["config_name"],
            "config_label": condition["config_label"],
            "p": int(condition["p"]),
            "q": int(condition["q"]),
            "r": int(condition["r"]),
            "n_samples": int(condition["n_samples"]),
            "noise": condition["noise_name"],
            "noise_parameters": condition["noise_config"],
            "generator": "random_orthogonal",
            "true_param_seed": int(condition["true_param_seed"]),
            "trial_seed_base": int(condition["trial_seed_base"]),
        },
        "runtime": {
            "condition_runtime_sec": float(condition_runtime_sec),
            "condition_runtime_min": float(condition_runtime_sec / 60.0),
        },
        "methods": {
            row["method"]: {
                key: value
                for key, value in row.items()
                if key
                not in {
                    "config_id",
                    "config_name",
                    "config_label",
                    "p",
                    "q",
                    "r",
                    "n_samples",
                    "noise",
                    "method",
                }
            }
            for row in summary_rows
        },
    }
    return summary_df, condition_summary


def _build_true_params(base_params: Dict[str, Any], noise_cfg: Dict[str, Any], n_samples: int) -> Dict[str, Any]:
    params = {
        key: (np.array(value, copy=True) if isinstance(value, np.ndarray) else copy.deepcopy(value))
        for key, value in base_params.items()
    }
    params["sigma_e2"] = float(noise_cfg["sigma_e2"])
    params["sigma_f2"] = float(noise_cfg["sigma_f2"])
    params["sigma_h2"] = float(noise_cfg["sigma_h2"])
    params["n_samples"] = int(n_samples)
    return params


def _condition_complete(condition_dir: Path, expected_trials: int) -> bool:
    summary_csv = condition_dir / "condition_summary.csv"
    trial_csv = condition_dir / "trial_method_metrics.csv"
    if not (summary_csv.exists() and trial_csv.exists()):
        return False
    try:
        df = pd.read_csv(summary_csv)
    except Exception:
        return False
    return bool(not df.empty and int(df["n_trials"].iloc[0]) == int(expected_trials))


def run_condition(condition: Dict[str, Any], true_params: Dict[str, Any], starting_points: List[np.ndarray], overwrite: bool) -> Dict[str, Any]:
    condition_dir = Path(condition["condition_dir"])
    ensure_dir(condition_dir)

    if (not overwrite) and _condition_complete(condition_dir, int(condition["n_trials"])):
        logging.info("Skipping completed condition %s", condition["config_label"])
        summary_df = pd.read_csv(condition_dir / "condition_summary.csv")
        condition_summary = json.loads((condition_dir / "condition_summary.json").read_text(encoding="utf-8"))
        return {
            "trial_df": pd.read_csv(condition_dir / "trial_method_metrics.csv"),
            "summary_df": summary_df,
            "condition_summary": condition_summary,
            "condition_dir": str(condition_dir),
        }

    _write_json(
        condition_dir / "condition_design.json",
        {
            "condition": condition,
            "true_parameters": {
                "B_diagonal": np.diag(true_params["B"]).tolist(),
                "Sigma_t_diagonal": np.diag(true_params["Sigma_t"]).tolist(),
                "identifiability_products": (np.diag(true_params["Sigma_t"]) * np.diag(true_params["B"])).tolist(),
                "sigma_e2": float(true_params["sigma_e2"]),
                "sigma_f2": float(true_params["sigma_f2"]),
                "sigma_h2": float(true_params["sigma_h2"]),
                "random_seed": int(true_params["random_seed"]),
            },
        },
    )

    t0 = time.perf_counter()
    trial_outputs: List[Dict[str, Any]] = []
    trial_ids = list(range(int(condition["n_trials"])))
    trial_seeds = [int(condition["trial_seed_base"]) + trial_id for trial_id in trial_ids]

    if bool(condition["parallel_trials"]) and int(condition["n_jobs"]) > 1:
        workers = min(int(condition["n_jobs"]), len(trial_ids))
        mp_ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=max(1, workers),
            mp_context=mp_ctx,
            initializer=_init_worker,
            initargs=(condition, true_params, starting_points),
        ) as executor:
            future_map = {
                executor.submit(_run_trial_worker, trial_id, trial_seed): trial_id
                for trial_id, trial_seed in zip(trial_ids, trial_seeds)
            }
            for future in as_completed(future_map):
                trial_outputs.append(future.result())
    else:
        _init_worker(condition, true_params, starting_points)
        for trial_id, trial_seed in zip(trial_ids, trial_seeds):
            trial_outputs.append(_run_trial_worker(trial_id, trial_seed))

    trial_outputs.sort(key=lambda item: int(item["trial_id"]))
    condition_runtime_sec = float(time.perf_counter() - t0)

    trial_df = _flatten_trial_outputs(condition, trial_outputs)
    summary_df, condition_summary = _summarize_method_rows(condition, trial_df, condition_runtime_sec)

    trial_df.to_csv(condition_dir / "trial_method_metrics.csv", index=False)
    summary_df.to_csv(condition_dir / "condition_summary.csv", index=False)
    _write_json(condition_dir / "condition_summary.json", condition_summary)

    return {
        "trial_df": trial_df,
        "summary_df": summary_df,
        "condition_summary": condition_summary,
        "condition_dir": str(condition_dir),
    }


def collect_hardware_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "logical_cpus": os.cpu_count(),
    }
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        info["memory_total_gb"] = round(float(vm.total) / (1024 ** 3), 2)
    except Exception:
        pass
    return info


def _load_scale_config(config_path: Path) -> Dict[str, Any]:
    cfg = load_config(config_path)
    experiments = cfg.get("experiments", {})
    if not isinstance(experiments, dict):
        raise ConfigError("config.experiments must be an object")
    scale_cfg = experiments.get("parameter_recovery_scale")
    if not isinstance(scale_cfg, dict):
        raise ConfigError("Missing experiments.parameter_recovery_scale in config.json")
    return scale_cfg


def _dimension_grid(scale_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    grid = scale_cfg.get("dimension_grid")
    if not isinstance(grid, list) or not grid:
        raise ConfigError("experiments.parameter_recovery_scale.dimension_grid must be a non-empty list")
    out: List[Dict[str, Any]] = []
    for idx, item in enumerate(grid, start=1):
        if not isinstance(item, dict):
            raise ConfigError(f"dimension_grid[{idx - 1}] must be an object")
        p, q, r = int(item["p"]), int(item["q"]), int(item["r"])
        n_samples = int(item.get("n_samples", max(3 * (p + q), 500)))
        name = str(item.get("name", f"config_{idx:02d}"))
        out.append(
            {
                "config_id": idx,
                "config_name": name,
                "config_label": f"C{idx}: (p={p}, q={q}, r={r}, N={n_samples})",
                "p": p,
                "q": q,
                "r": r,
                "n_samples": n_samples,
            }
        )
    return out


def _noise_levels(scale_cfg: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    noise_levels = scale_cfg.get("noise_levels")
    if not isinstance(noise_levels, dict) or not noise_levels:
        raise ConfigError("experiments.parameter_recovery_scale.noise_levels must be an object")
    out: Dict[str, Dict[str, float]] = {}
    for name in NOISE_ORDER:
        if name not in noise_levels:
            raise ConfigError(f"Missing noise level '{name}' in experiments.parameter_recovery_scale.noise_levels")
        vals = noise_levels[name]
        out[name] = {
            "sigma_e2": float(vals["sigma_e2"]),
            "sigma_f2": float(vals["sigma_f2"]),
            "sigma_h2": float(vals["sigma_h2"]),
        }
    return out


def _build_condition(
    config_entry: Dict[str, Any],
    noise_name: str,
    noise_cfg: Dict[str, float],
    scale_cfg: Dict[str, Any],
    stage_dir: Path,
    n_trials: int,
    trial_seed_base: int,
    true_param_seed: int,
    n_jobs_override: int | None,
) -> Dict[str, Any]:
    condition_dir = stage_dir / config_entry["config_name"] / noise_name
    n_jobs = int(n_jobs_override) if n_jobs_override is not None else int(scale_cfg.get("n_jobs", 1))
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1
    return {
        **config_entry,
        "noise_name": noise_name,
        "noise_config": noise_cfg,
        "n_trials": int(n_trials),
        "n_starts": int(scale_cfg.get("n_starts", 8)),
        "parallel_trials": bool(scale_cfg.get("parallel_trials", True)),
        "n_jobs": max(1, int(n_jobs)),
        "optimizer": str(scale_cfg.get("optimizer", "manifold")),
        "max_iter": int(scale_cfg.get("max_iter", 2000)),
        "condition_dir": str(condition_dir),
        "true_param_seed": int(true_param_seed),
        "trial_seed_base": int(trial_seed_base),
        "experiment_cfg": {
            "optimizer": str(scale_cfg.get("optimizer", "manifold")),
            "max_iter": int(scale_cfg.get("max_iter", 2000)),
            "slm_gtol": float(scale_cfg.get("slm_gtol", 5e-3)),
            "slm_xtol": float(scale_cfg.get("slm_xtol", 5e-3)),
            "slm_barrier_tol": float(scale_cfg.get("slm_barrier_tol", 5e-3)),
            "slm_constraint_slack": float(scale_cfg.get("slm_constraint_slack", 5e-3)),
            "slm_verbose": bool(scale_cfg.get("slm_verbose", False)),
            "slm_progress_every": int(scale_cfg.get("slm_progress_every", 1)),
            "slm_early_stop_patience": scale_cfg.get("slm_early_stop_patience"),
            "slm_early_stop_rel_improvement": scale_cfg.get("slm_early_stop_rel_improvement"),
            "em_tolerance": float(scale_cfg.get("em_tolerance", 5e-3)),
            "ecm_tolerance": float(scale_cfg.get("ecm_tolerance", 5e-3)),
        },
    }


def _aggregate_stage_outputs(stage_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_paths = sorted(stage_dir.glob("*/*/condition_summary.csv"))
    runtime_rows: List[Dict[str, Any]] = []
    summary_frames: List[pd.DataFrame] = []
    for path in summary_paths:
        df = pd.read_csv(path)
        summary_frames.append(df)
        if not df.empty:
            head = df.iloc[0]
            runtime_rows.append(
                {
                    "config_id": int(head["config_id"]),
                    "config_name": head["config_name"],
                    "config_label": head["config_label"],
                    "p": int(head["p"]),
                    "q": int(head["q"]),
                    "r": int(head["r"]),
                    "n_samples": int(head["n_samples"]),
                    "noise": head["noise"],
                    "condition_runtime_sec": float(head["condition_runtime_sec"]),
                    "condition_runtime_min": float(head["condition_runtime_sec"] / 60.0),
                }
            )
    summary_df = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    runtime_df = pd.DataFrame(runtime_rows)
    if not runtime_df.empty:
        runtime_df = runtime_df.sort_values(["config_id", "noise"]).reset_index(drop=True)
    return summary_df, runtime_df


def _save_stage_aggregates(stage_dir: Path, manifest: Dict[str, Any]) -> None:
    summary_df, runtime_df = _aggregate_stage_outputs(stage_dir)
    if not summary_df.empty:
        summary_df.to_csv(stage_dir / "parameter_recovery_scale_summary.csv", index=False)
    if not runtime_df.empty:
        runtime_df.to_csv(stage_dir / "parameter_recovery_scale_runtime.csv", index=False)
    _write_json(stage_dir / "run_manifest.json", manifest)


def _run_stage(
    *,
    stage_name: str,
    stage_dir: Path,
    scale_cfg: Dict[str, Any],
    hardware_info: Dict[str, Any],
    overwrite: bool,
    smoke_only: bool,
    n_jobs_override: int | None,
) -> Dict[str, Any]:
    ensure_dir(stage_dir)
    config_grid = _dimension_grid(scale_cfg)
    noise_levels = _noise_levels(scale_cfg)
    base_seed = int(scale_cfg.get("seed", 42))
    n_trials = int(scale_cfg["smoke_trials"] if smoke_only else scale_cfg["n_trials"])
    start_time = time.perf_counter()
    completed_conditions: List[Dict[str, Any]] = []

    target_configs = config_grid[:1] if smoke_only else config_grid
    for config_entry in target_configs:
        logging.info("Starting %s for %s", stage_name, config_entry["config_label"])
        generator = RandomOrthogonalDataGenerator(
            p=int(config_entry["p"]),
            q=int(config_entry["q"]),
            r=int(config_entry["r"]),
            n_samples=int(config_entry["n_samples"]),
            random_seed=base_seed + 100 * int(config_entry["config_id"]),
            output_dir=str(stage_dir / config_entry["config_name"]),
        )
        base_true = generator.generate_true_parameters()
        starting_points = InitialPointGenerator(
            p=int(config_entry["p"]),
            q=int(config_entry["q"]),
            r=int(config_entry["r"]),
            n_starts=int(scale_cfg.get("n_starts", 8)),
            random_seed=base_seed + 1000 + int(config_entry["config_id"]),
        ).generate_starting_points()

        config_summaries: List[pd.DataFrame] = []
        for noise_idx, noise_name in enumerate(NOISE_ORDER):
            noise_cfg = noise_levels[noise_name]
            trial_seed_base = base_seed + 10000 * int(config_entry["config_id"]) + 1000 * noise_idx
            condition = _build_condition(
                config_entry,
                noise_name,
                noise_cfg,
                scale_cfg,
                stage_dir,
                n_trials,
                trial_seed_base,
                base_seed + 100 * int(config_entry["config_id"]),
                n_jobs_override,
            )
            true_params = _build_true_params(base_true, noise_cfg, int(config_entry["n_samples"]))
            result = run_condition(condition, true_params, starting_points, overwrite=overwrite)
            config_summaries.append(result["summary_df"])
            completed_conditions.append(result["condition_summary"])  # type: ignore[arg-type]

        config_df = pd.concat(config_summaries, ignore_index=True)
        config_dir = stage_dir / config_entry["config_name"]
        config_df.to_csv(config_dir / "config_summary.csv", index=False)
        _save_stage_aggregates(
            stage_dir,
            {
                "stage": stage_name,
                "hardware": hardware_info,
                "n_trials": n_trials,
                "completed_conditions": completed_conditions,
                "elapsed_sec": float(time.perf_counter() - start_time),
            },
        )

    manifest = {
        "stage": stage_name,
        "hardware": hardware_info,
        "n_trials": n_trials,
        "completed_conditions": completed_conditions,
        "elapsed_sec": float(time.perf_counter() - start_time),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _save_stage_aggregates(stage_dir, manifest)
    return manifest


def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Large-scale parameter recovery experiment")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config.json")
    parser.add_argument(
        "--mode",
        choices=["all", "smoke", "full"],
        default="all",
        help="Run smoke validation only, full grid only, or smoke followed by full",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite completed condition outputs")
    parser.add_argument("--n-jobs", type=int, default=None, help="Override parallel trial workers")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(argv)
    root = repo_root()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (root / config_path).resolve()

    scale_cfg = _load_scale_config(config_path)
    output_dir = Path(scale_cfg.get("output_dir", "output/parameter_recovery_scale"))
    if not output_dir.is_absolute():
        output_dir = (root / output_dir).resolve()
    ensure_dir(output_dir)
    logs_dir = ensure_dir(output_dir / "logs")
    log_path = setup_logging(logs_dir, filename="parameter_recovery_scale.log")

    hardware_info = collect_hardware_info()
    logging.info("Parameter recovery at scale started")
    logging.info("Config path: %s", config_path)
    logging.info("Output dir: %s", output_dir)
    logging.info("Log path: %s", log_path)
    logging.info("Hardware: %s", hardware_info)

    print("=" * 72)
    print("Large-scale parameter recovery experiment")
    print("=" * 72)
    print(f"Config      : {config_path}")
    print(f"Output dir  : {output_dir}")
    print(f"Log file    : {log_path}")
    print(f"Optimizer   : {scale_cfg.get('optimizer', 'manifold')}")
    print(f"Base seed   : {int(scale_cfg.get('seed', 42))}")
    print(f"Hardware    : {hardware_info.get('machine', 'unknown')} / {hardware_info.get('processor', 'unknown')} / logical_cpus={hardware_info.get('logical_cpus')}")
    print("=" * 72)

    if args.mode in {"all", "smoke"}:
        smoke_dir = ensure_dir(output_dir / "smoke")
        smoke_manifest = _run_stage(
            stage_name="smoke",
            stage_dir=smoke_dir,
            scale_cfg=scale_cfg,
            hardware_info=hardware_info,
            overwrite=args.overwrite,
            smoke_only=True,
            n_jobs_override=args.n_jobs,
        )
        print(f"[OK] Smoke validation completed in {smoke_manifest['elapsed_sec'] / 60.0:.2f} min")
        if args.mode == "smoke":
            return 0

    full_dir = ensure_dir(output_dir / "full")
    full_manifest = _run_stage(
        stage_name="full",
        stage_dir=full_dir,
        scale_cfg=scale_cfg,
        hardware_info=hardware_info,
        overwrite=args.overwrite,
        smoke_only=False,
        n_jobs_override=args.n_jobs,
    )
    print(f"[OK] Full experiment completed in {full_manifest['elapsed_sec'] / 3600.0:.2f} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
