"""Monte Carlo pipeline entry point.

This module orchestrates a three-stage pipeline:
1) synthetic data generation
2) parameter estimation (Monte Carlo)
3) visualization

The implementation is intentionally "thin": core logic lives in `ppls_slm.*` modules.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict


import numpy as np

from ppls_slm.data_generator import SineDataGenerator
from ppls_slm.experiment import PPLSExperiment
from ppls_slm.experiment_config import load_config_with_defaults
from ppls_slm.visualization import PPLSVisualizer
from ppls_slm.utils import repo_root, setup_logging


_DEFAULTS: Dict = {
    # Paper default simulation setting (Section 8.1)
    "model": {"p": 20, "q": 20, "r": 3, "n_samples": 500},
    "data_generation": {
        "noise_levels": {
            "low": {"sigma_e2": 0.1, "sigma_f2": 0.1, "sigma_h2": 0.05},
            "high": {"sigma_e2": 0.5, "sigma_f2": 0.5, "sigma_h2": 0.25},
        },
        "sine_parameters": {"frequency": 0.7, "magnitude": 0.7},
    },
    "algorithms": {
        "common": {"n_starts": 8, "random_seed": 42},
        "slm_manifold": {
            "optimizer": "manifold",
            "max_iter": 2000,
            "use_noise_preestimation": True,
        },
        "slm_interior": {
            "optimizer": "trust-constr",
            "max_iter": 2000,
            "use_noise_preestimation": True,
            "gtol": 0.005,
            "xtol": 0.005,
            "barrier_tol": 0.005,
            "constraint_slack": 0.005,
        },
        "bcd_slm": {
            "max_outer_iter": 200,
            "n_cg_steps_W": 5,
            "n_cg_steps_C": 5,
            "tolerance": 1e-4,
            "use_noise_preestimation": True,
        },
        "em": {"max_iter": 2000, "tolerance": 0.005},
        "ecm": {"max_iter": 2000, "tolerance": 0.005},
    },
    "experiment": {"n_trials": 20, "random_seed": 42, "parallel_trials": False, "n_jobs": 1},
    "output": {
        "save_intermediate": True,
        "base_dir": None,
        "figure_format": "png",
        "force_data_generation": False,
        "force_parameter_estimation": False,
        "force_visualization": False,
    },
}



def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPLS Monte Carlo pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to config JSON (default: config.json)",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Override output.base_dir (optional)",
    )
    parser.add_argument(
        "--noise-level",
        type=str,
        action="append",
        default=None,
        help=(
            "Run only selected noise level(s). Example: --noise-level low  or  --noise-level high. "
            "You can repeat the flag or pass a comma-separated list."
        ),
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Override experiment.n_trials (useful for smoke tests)",
    )
    return parser.parse_args(argv)



def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # Resolve config path: if relative, interpret it from repo root.
    root = repo_root()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (root / config_path).resolve()

    print("Loading configuration...")
    config = load_config_with_defaults(config_path, defaults=_DEFAULTS)

    # Optional override for quick smoke tests.
    if args.n_trials is not None:
        config = copy.deepcopy(config)
        config.setdefault("experiment", {})["n_trials"] = int(args.n_trials)

    # Optionally restrict to selected noise levels (so low/high can be run separately).
    if args.noise_level:
        requested: list[str] = []
        for item in args.noise_level:
            for part in str(item).split(","):
                name = part.strip()
                if name:
                    requested.append(name)

        # stable de-dup
        requested = list(dict.fromkeys(requested))

        noise_levels = config.get("data_generation", {}).get("noise_levels", {})
        if not isinstance(noise_levels, dict) or len(noise_levels) == 0:
            noise_levels = _DEFAULTS.get("data_generation", {}).get("noise_levels", {})

        unknown = [n for n in requested if n not in noise_levels]
        if unknown:
            raise ValueError(
                f"Unknown noise level(s): {unknown}. Available: {list(noise_levels.keys())}"
            )

        config = copy.deepcopy(config)
        config.setdefault("data_generation", {})["noise_levels"] = {k: noise_levels[k] for k in requested}
        print(f"Restricting run to noise level(s): {', '.join(requested)}")

    if not validate_configuration(config):
        print("Configuration validation failed. Exiting.")
        return 1


    # Determine base directory.
    base_dir = args.base_dir or config.get("output", {}).get("base_dir") or os.getcwd()
    base_dir = os.path.abspath(str(base_dir))

    os.makedirs(base_dir, exist_ok=True)

    # Logging
    log_dir = os.path.join(base_dir, "logs")
    log_path = setup_logging(log_dir, filename="experiment.log")
    logging.info("=" * 60)
    logging.info("PPLS Parameter Estimation Experiment Started")
    logging.info(f"Config    : {config_path}")
    logging.info(f"Log file  : {log_path}")
    logging.info(f"Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("=" * 60)

    print("\n" + "=" * 60)
    print("PPLS Parameter Estimation Comparison Experiment")
    print("=" * 60)
    print(f"Model dimensions: p={config['model']['p']}, q={config['model']['q']}, r={config['model']['r']}")
    print(f"Number of trials: {config['experiment']['n_trials']}")
    print(f"Base directory: {base_dir}")

    if not os.access(base_dir, os.W_OK):
        print(f"Error: Base directory is not writable: {base_dir}")
        return 1

    print("=" * 60 + "\n")

    stages_run: list[str] = []

    try:
        if run_data_generation_stage(config, base_dir):
            stages_run.append("Data Generation")

        if run_parameter_estimation_stage(config, base_dir):
            stages_run.append("Parameter Estimation")

        if run_visualization_stage(config, base_dir):
            stages_run.append("Visualization")

        print_experiment_summary(config, stages_run, base_dir)

        print("\nExperiment completed successfully!")
        print(f"Stages executed: {', '.join(stages_run) if stages_run else 'None (all outputs existed)'}")
        return 0

    except Exception as e:
        logging.error(f"Experiment failed: {str(e)}", exc_info=True)
        print(f"\nExperiment failed with error: {str(e)}")
        return 1


def run_data_generation_stage(config: Dict, base_dir: str) -> bool:
    """Execute data generation stage for each configured noise level.

    For backward compatibility, keep directory names for the low-noise setting:
    - low  -> base_dir/data
    - other levels -> base_dir/data_<level>

    Returns True if any data was generated.
    """

    force = config.get("output", {}).get("force_data_generation", False)
    noise_levels = config.get("data_generation", {}).get("noise_levels", {})
    if not isinstance(noise_levels, dict) or len(noise_levels) == 0:
        noise_levels = {"low": {"sigma_e2": 0.1, "sigma_f2": 0.1, "sigma_h2": 0.05}}

    generated_any = False

    for level_name, noise_config in noise_levels.items():
        data_dir = os.path.join(base_dir, "data" if level_name == "low" else f"data_{level_name}")

        if not force and check_directory_status(data_dir):
            print(f"Data directory for noise='{level_name}' exists and is non-empty. Skipping data generation.")
            logging.info(f"Skipping data generation - data already exists (noise='{level_name}')")
            continue

        print(f"Stage 1: Generating experimental data (noise='{level_name}')...")
        logging.info(f"Starting data generation stage (noise='{level_name}')")

        os.makedirs(data_dir, exist_ok=True)

        generator = SineDataGenerator(
            p=config["model"]["p"],
            q=config["model"]["q"],
            r=config["model"]["r"],
            n_samples=config["model"]["n_samples"],
            random_seed=config["experiment"]["random_seed"],
            output_dir=data_dir,
        )

        sigma_e2 = noise_config.get("sigma_e2", 0.1)
        sigma_f2 = noise_config.get("sigma_f2", 0.1)
        sigma_h2 = noise_config.get("sigma_h2", 0.05)

        true_params = generator.generate_true_parameters(sigma_e2=sigma_e2, sigma_f2=sigma_f2, sigma_h2=sigma_h2)

        all_X = []
        all_Y = []

        for trial_id in range(config["experiment"]["n_trials"]):
            # Keep the original per-trial data seed scheme.
            data_seed = config["experiment"]["random_seed"] + 1000 + trial_id
            np.random.seed(data_seed)

            from ppls_slm.ppls_model import PPLSModel

            model = PPLSModel(config["model"]["p"], config["model"]["q"], config["model"]["r"])

            X, Y = model.sample(
                n_samples=config["model"]["n_samples"],
                W=true_params["W"],
                C=true_params["C"],
                B=true_params["B"],
                Sigma_t=true_params["Sigma_t"],
                sigma_e2=true_params["sigma_e2"],
                sigma_f2=true_params["sigma_f2"],
                sigma_h2=true_params["sigma_h2"],
            )

            all_X.append(X)
            all_Y.append(Y)

        np.save(os.path.join(data_dir, "X_trials.npy"), np.array(all_X))
        np.save(os.path.join(data_dir, "Y_trials.npy"), np.array(all_Y))

        import pickle

        with open(os.path.join(data_dir, "ground_truth.pkl"), "wb") as f:
            pickle.dump(true_params, f)

        param_summary = {
            "model_info": {
                "dimensions": {"p": config["model"]["p"], "q": config["model"]["q"], "r": config["model"]["r"]},
                "n_trials": config["experiment"]["n_trials"],
                "n_samples": config["model"]["n_samples"],
            },
            "noise_level": level_name,
            "noise_parameters": {"sigma_e2": float(sigma_e2), "sigma_f2": float(sigma_f2), "sigma_h2": float(sigma_h2)},
            "loading_matrices": {
                "W_shape": list(true_params["W"].shape),
                "C_shape": list(true_params["C"].shape),
                "W_norm_by_component": [float(np.linalg.norm(true_params["W"][:, i])) for i in range(config["model"]["r"])],
                "C_norm_by_component": [float(np.linalg.norm(true_params["C"][:, i])) for i in range(config["model"]["r"])],
            },
            "diagonal_parameters": {
                "B_diagonal": [float(x) for x in np.diag(true_params["B"])],
                "Sigma_t_diagonal": [float(x) for x in np.diag(true_params["Sigma_t"])],
                "identifiability_products": [float(x) for x in (np.diag(true_params["Sigma_t"]) * np.diag(true_params["B"]))],
            },
            "generation_info": {"random_seed": config["experiment"]["random_seed"], "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
        }

        with open(os.path.join(data_dir, "data_summary.json"), "w", encoding="utf-8") as f:
            json.dump(param_summary, f, indent=2)

        print(f"[OK] Data generated for {config['experiment']['n_trials']} trials (noise='{level_name}')")
        print(f"[OK] Data saved to: {data_dir}")
        logging.info(f"Data generation completed - {config['experiment']['n_trials']} trials (noise='{level_name}')")

        generated_any = True

    return generated_any


def run_parameter_estimation_stage(config: Dict, base_dir: str) -> bool:
    """Execute parameter estimation stage for each configured noise level."""

    force = config.get("output", {}).get("force_parameter_estimation", False)
    noise_levels = config.get("data_generation", {}).get("noise_levels", {})
    if not isinstance(noise_levels, dict) or len(noise_levels) == 0:
        noise_levels = {"low": {"sigma_e2": 0.1, "sigma_f2": 0.1, "sigma_h2": 0.05}}

    def _format_mean_pm_std(mean: float, std: float, scale: float = 100.0, decimals: int = 2) -> str:
        return f"{mean*scale:.{decimals}f}±{std*scale:.{decimals}f}"

    def _extract_mse_table_from_results(experiment_results: Dict) -> Dict:
        analysis = experiment_results.get("analysis", {})
        params = ["W", "C", "B", "Sigma_t", "sigma_h2", "sigma_e2", "sigma_f2"]
        methods = ["slm_manifold", "bcd_slm", "slm_interior", "slm_oracle", "em", "ecm"]


        table: Dict = {}
        for method in methods:
            if method not in analysis:
                continue
            table[method] = {}
            for param in params:
                key = f"mse_{param}"
                if key in analysis[method]:
                    mean = float(analysis[method][key].get("mean", float("nan")))
                    std = float(analysis[method][key].get("std", float("nan")))
                    table[method][param] = {
                        "mean": mean,
                        "std": std,
                        "table_str_x1e2": _format_mean_pm_std(mean, std, scale=100.0, decimals=2),
                    }
        return table

    robustness_summary = {
        "model": config.get("model", {}),
        "experiment": config.get("experiment", {}),
        "noise_levels": {},
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    estimated_any = False

    for level_name, noise_cfg in noise_levels.items():
        if level_name == "low":
            data_dir = os.path.join(base_dir, "data")
            results_dir = os.path.join(base_dir, "results")
        else:
            data_dir = os.path.join(base_dir, f"data_{level_name}")
            results_dir = os.path.join(base_dir, f"results_{level_name}")

        results_pkl = os.path.join(results_dir, "experiment_results.pkl")
        if not force and check_directory_status(results_dir) and os.path.exists(results_pkl):
            print(
                f"Results directory for noise='{level_name}' exists and contains experiment_results.pkl. Skipping estimation."
            )
            logging.info(f"Skipping parameter estimation - results already exist (noise='{level_name}')")

            try:
                import pickle

                with open(results_pkl, "rb") as f:
                    existing_results = pickle.load(f)

                mse_table = _extract_mse_table_from_results(existing_results)
                with open(os.path.join(results_dir, "mse_table.json"), "w", encoding="utf-8") as f:
                    json.dump(mse_table, f, indent=2)

                robustness_summary["noise_levels"][level_name] = {
                    "noise_parameters": {
                        "sigma_e2": float(noise_cfg.get("sigma_e2", 0.1)),
                        "sigma_f2": float(noise_cfg.get("sigma_f2", 0.1)),
                        "sigma_h2": float(noise_cfg.get("sigma_h2", 0.05)),
                    },
                    "data_dir": data_dir,
                    "results_dir": results_dir,
                    "mse_table": mse_table,
                    "n_trials_completed": int(existing_results.get("n_trials_completed", 0)),
                }
            except Exception as e:
                robustness_summary["noise_levels"][level_name] = {
                    "noise_parameters": {
                        "sigma_e2": float(noise_cfg.get("sigma_e2", 0.1)),
                        "sigma_f2": float(noise_cfg.get("sigma_f2", 0.1)),
                        "sigma_h2": float(noise_cfg.get("sigma_h2", 0.05)),
                    },
                    "data_dir": data_dir,
                    "results_dir": results_dir,
                    "warning": f"Could not load existing experiment_results.pkl: {str(e)}",
                }
            continue

        if not force and check_directory_status(results_dir) and not os.path.exists(results_pkl):
            print(
                f"Results directory for noise='{level_name}' looks incomplete (missing experiment_results.pkl). Re-running estimation."
            )
            logging.info(f"Found incomplete results directory, re-running estimation (noise='{level_name}')")

        if not check_directory_status(data_dir):
            raise FileNotFoundError(
                f"Data directory for noise='{level_name}' is empty or doesn't exist: {data_dir}. Run data generation first."
            )

        print(f"Stage 2: Running parameter estimation (noise='{level_name}')...")
        logging.info(f"Starting parameter estimation stage (noise='{level_name}')")

        os.makedirs(results_dir, exist_ok=True)

        experiment = PPLSExperiment(config, base_dir, results_dir, data_dir=data_dir)
        results = experiment.run_monte_carlo()

        import pickle

        with open(os.path.join(results_dir, "experiment_results.pkl"), "wb") as f:
            pickle.dump(results, f)

        mse_table = _extract_mse_table_from_results(results)
        with open(os.path.join(results_dir, "mse_table.json"), "w", encoding="utf-8") as f:
            json.dump(mse_table, f, indent=2)

        readable_summary = {
            "noise_level": level_name,
            "noise_parameters": {
                "sigma_e2": float(noise_cfg.get("sigma_e2", 0.1)),
                "sigma_f2": float(noise_cfg.get("sigma_f2", 0.1)),
                "sigma_h2": float(noise_cfg.get("sigma_h2", 0.05)),
            },
            "experiment_overview": {
                "n_trials_completed": results.get("n_trials_completed", 0),
                "success_rate_percent": round(results.get("n_trials_completed", 0) / config["experiment"]["n_trials"] * 100, 1),
                "total_runtime_minutes": round(results.get("timing", {}).get("total_time", 0) / 60, 2),
                "avg_time_per_trial_seconds": round(results.get("timing", {}).get("avg_time_per_trial", 0), 2),
            },
            "algorithm_performance": {},
            "parameter_estimation_quality": {},
            "mse_table": mse_table,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        if "analysis" in results and "runtime_statistics" in results["analysis"]:
            runtime_stats = results["analysis"]["runtime_statistics"]
            readable_summary["algorithm_performance"] = {
                "slm_manifold": {
                    "avg_runtime_seconds": round(runtime_stats.get("slm_manifold", {}).get("avg_runtime", 0), 2),
                    "avg_convergence_rate_percent": round(runtime_stats.get("slm_manifold", {}).get("avg_convergence_rate", 0) * 100, 1),
                },
                "bcd_slm": {
                    "avg_runtime_seconds": round(runtime_stats.get("bcd_slm", {}).get("avg_runtime", 0), 2),
                    "avg_convergence_rate_percent": round(runtime_stats.get("bcd_slm", {}).get("avg_convergence_rate", 0) * 100, 1),
                },
                "slm_interior": {
                    "avg_runtime_seconds": round(runtime_stats.get("slm_interior", {}).get("avg_runtime", 0), 2),
                    "avg_convergence_rate_percent": round(runtime_stats.get("slm_interior", {}).get("avg_convergence_rate", 0) * 100, 1),
                },
                "slm_oracle": {
                    "avg_runtime_seconds": round(runtime_stats.get("slm_oracle", {}).get("avg_runtime", 0), 2),
                    "avg_convergence_rate_percent": round(runtime_stats.get("slm_oracle", {}).get("avg_convergence_rate", 0) * 100, 1),
                },
                "em": {
                    "avg_runtime_seconds": round(runtime_stats.get("em", {}).get("avg_runtime", 0), 2),
                    "avg_convergence_rate_percent": round(runtime_stats.get("em", {}).get("avg_convergence_rate", 0) * 100, 1),
                },
                "ecm": {
                    "avg_runtime_seconds": round(runtime_stats.get("ecm", {}).get("avg_runtime", 0), 2),
                    "avg_convergence_rate_percent": round(runtime_stats.get("ecm", {}).get("avg_convergence_rate", 0) * 100, 1),
                },
            }

        if "analysis" in results:
            for method in ["slm_manifold", "bcd_slm", "slm_interior", "slm_oracle", "em", "ecm"]:

                if method in results["analysis"]:
                    method_quality: Dict[str, float] = {}
                    for param in ["W", "C", "B", "Sigma_t", "sigma_h2"]:
                        key = f"mse_{param}"
                        if key in results["analysis"][method]:
                            method_quality[f"{param}_mse_mean"] = round(results["analysis"][method][key]["mean"], 6)
                            method_quality[f"{param}_mse_std"] = round(results["analysis"][method][key]["std"], 6)
                    readable_summary["parameter_estimation_quality"][method] = method_quality

        with open(os.path.join(results_dir, "results_summary.json"), "w", encoding="utf-8") as f:
            json.dump(readable_summary, f, indent=2)

        print(
            f"[OK] Parameter estimation completed for {results.get('n_trials_completed', 0)} trials (noise='{level_name}')"
        )
        print(f"[OK] Results saved to: {results_dir}")
        logging.info(
            f"Parameter estimation completed - {results.get('n_trials_completed', 0)} trials (noise='{level_name}')"
        )

        robustness_summary["noise_levels"][level_name] = {
            "noise_parameters": readable_summary["noise_parameters"],
            "data_dir": data_dir,
            "results_dir": results_dir,
            "mse_table": mse_table,
            "n_trials_completed": int(results.get("n_trials_completed", 0)),
        }

        estimated_any = True

    with open(os.path.join(base_dir, "robustness_summary.json"), "w", encoding="utf-8") as f:
        json.dump(robustness_summary, f, indent=2)

    return estimated_any


def run_visualization_stage(config: Dict, base_dir: str) -> bool:
    """Generate visualizations for each configured noise level.

    For backward compatibility, the low-noise setting uses:
      - figures -> base_dir/figures
      - results -> base_dir/results
      - data    -> base_dir/data

    Other levels use:
      - figures -> base_dir/figures_<level>
      - results -> base_dir/results_<level>
      - data    -> base_dir/data_<level>
    """

    force = config.get("output", {}).get("force_visualization", False)
    figure_format = config.get("output", {}).get("figure_format", "pdf")

    noise_levels = config.get("data_generation", {}).get("noise_levels", {})
    if not isinstance(noise_levels, dict) or len(noise_levels) == 0:
        noise_levels = {"low": {"sigma_e2": 0.1, "sigma_f2": 0.1, "sigma_h2": 0.05}}

    generated_any = False

    for level_name in noise_levels.keys():
        if level_name == "low":
            figures_dir = os.path.join(base_dir, "figures")
            results_dir = os.path.join(base_dir, "results")
            data_dir = os.path.join(base_dir, "data")
        else:
            figures_dir = os.path.join(base_dir, f"figures_{level_name}")
            results_dir = os.path.join(base_dir, f"results_{level_name}")
            data_dir = os.path.join(base_dir, f"data_{level_name}")

        if not force and check_directory_status(figures_dir):
            print(f"Figures for noise='{level_name}' exist and are non-empty. Skipping visualization.")
            logging.info(f"Skipping visualization - figures already exist (noise='{level_name}')")
            continue

        if not check_directory_status(results_dir):
            raise FileNotFoundError(
                f"Results directory for noise='{level_name}' is empty or doesn't exist: {results_dir}. "
                "Run parameter estimation first."
            )
        if not check_directory_status(data_dir):
            raise FileNotFoundError(
                f"Data directory for noise='{level_name}' is empty or doesn't exist: {data_dir}. "
                "Run data generation first."
            )

        print(f"Stage 3: Generating visualizations (noise='{level_name}')...")
        logging.info(f"Starting visualization stage (noise='{level_name}')")

        os.makedirs(figures_dir, exist_ok=True)
        visualizer = PPLSVisualizer(
            base_dir,
            figure_format=figure_format,
            figure_dir=figures_dir,
            data_dir=data_dir,
            results_dir=results_dir,
        )

        import pickle

        results_path = os.path.join(results_dir, "experiment_results.pkl")
        with open(results_path, "rb") as f:
            experiment_results = pickle.load(f)

        visualizer.create_results_summary(experiment_results)

        if experiment_results.get("trial_results"):
            first_trial = experiment_results["trial_results"][0]
            for component_idx in range(config["model"]["r"]):
                visualizer.plot_loading_comparison(first_trial, component_idx)

        if "analysis" in experiment_results:
            visualizer.plot_parameter_recovery(experiment_results["analysis"])

        if "trial_results" in experiment_results:
            visualizer.plot_convergence_history(experiment_results["trial_results"])

        visualizer.save_all_figures()

        print(f"[OK] Visualizations generated (noise='{level_name}')")
        print(f"[OK] Figures saved to: {figures_dir}")
        logging.info(f"Visualization completed (noise='{level_name}')")

        generated_any = True

    return generated_any


def check_directory_status(directory: str) -> bool:
    if not os.path.exists(directory):
        return False
    try:
        return len(os.listdir(directory)) > 0
    except OSError:
        return False


def validate_configuration(config: Dict) -> bool:
    try:
        required_sections = ["model", "algorithms", "experiment"]
        for section in required_sections:
            if section not in config:
                print(f"Missing required section: {section}")
                return False

        p = config["model"]["p"]
        q = config["model"]["q"]
        r = config["model"]["r"]

        if r > min(p, q):
            print(f"Invalid: r ({r}) must be less than min(p, q) = {min(p, q)}")
            return False

        if config["model"]["n_samples"] < 10:
            print("Invalid: n_samples must be at least 10")
            return False

        if config["algorithms"]["common"]["n_starts"] < 1:
            print("Invalid: n_starts must be at least 1")
            return False

        if config["experiment"]["n_trials"] < 1:
            print("Invalid: n_trials must be at least 1")
            return False

        return True

    except Exception as e:
        print(f"Configuration validation error: {str(e)}")
        return False


def print_experiment_summary(config: Dict, stages_run: list, base_dir: str) -> None:
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    print("\nConfiguration:")
    print(f"  Model dimensions: p={config['model']['p']}, q={config['model']['q']}, r={config['model']['r']}")
    print(f"  Sample size: {config['model']['n_samples']}")
    print(f"  Number of trials: {config['experiment']['n_trials']}")
    print(f"  Starting points: {config['algorithms']['common']['n_starts']}")

    print(f"\nStages executed: {', '.join(stages_run) if stages_run else 'None (all outputs existed)'}")

    print("\nOutput directories:")
    print(f"  Data (low): {os.path.join(base_dir, 'data')}")
    print(f"  Results (low): {os.path.join(base_dir, 'results')}")
    print(f"  Figures: {os.path.join(base_dir, 'figures')}")
    print(f"  Logs: {os.path.join(base_dir, 'logs')}")

    noise_levels = config.get("data_generation", {}).get("noise_levels", {})
    if isinstance(noise_levels, dict):
        extra_levels = [k for k in noise_levels.keys() if k != "low"]
        for level in extra_levels:
            print(f"  Data ({level}): {os.path.join(base_dir, f'data_{level}')} ")
            print(f"  Results ({level}): {os.path.join(base_dir, f'results_{level}')} ")

    print(f"  Robustness summary: {os.path.join(base_dir, 'robustness_summary.json')}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    raise SystemExit(main())
