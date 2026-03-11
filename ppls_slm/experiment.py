"""
Experiment Runner and Analysis for PPLS Parameter Estimation
===========================================================

This module orchestrates Monte Carlo experiments ensuring fair comparison between SLM, EM, 
and ECM algorithms with identical starting points and fixed ground truth parameters. Uses a 
simplified directory structure with all data loaded from the data directory and results saved 
to the results directory.

Architecture Overview:
---------------------
The module provides:
1. PPLSExperiment: Main experiment coordinator for running comparative studies
2. PerformanceMetrics: Calculate estimation quality metrics (MSE, bias, variance)
3. ParameterRecovery: Assess parameter recovery quality and identifiability

Function List:
--------------
PPLSExperiment:
    - __init__(config, base_dir, results_dir): Initialize experiment with simplified structure
    - run_monte_carlo(): Execute multiple trials with parallel processing
    - _run_single_trial(trial_id, X, Y, true_params): Execute one complete trial with all algorithms
    - _compare_algorithms(X, Y, starting_points, true_params): Compare SLM, EM, and ECM
    - _run_algorithm_with_stats(algorithm, X, Y, starting_points, algorithm_name): Run algorithm and collect stats
    - analyze_results(trial_results): Compute summary statistics across trials
    - save_experiment_summary(results): Save overall experiment results

PerformanceMetrics:
    - __init__(true_params): Initialize with ground truth
    - compute_mse(estimated_params): Mean squared error calculation
    - compute_bias(estimated_params_list): Bias of parameter estimates
    - compute_variance(estimated_params_list): Variance of estimates
    - compute_correlation(W_est, C_est, W_true, C_true): Loading correlations
    - generate_summary_table(slm_metrics_list, em_metrics_list, ecm_metrics_list): Create results table
    - save_metrics(metrics, filename): Save computed metrics

ParameterRecovery:
    - align_estimated_params(params_est, params_true): Handle sign indeterminacy
    - compute_loading_errors(W_est, C_est, W_true, C_true): W and C errors
    - compute_diagonal_errors(B_est, Sigma_t_est, B_true, Sigma_t_true): B and Σt errors
    - assess_identifiability(params): Check identifiability constraints
    - compute_recovery_rate(estimated_list, true_params, threshold): Recovery success rate

Call Relationships:
------------------
PPLSExperiment.run_monte_carlo() → PPLSExperiment._run_single_trial()
PPLSExperiment._run_single_trial() → PPLSExperiment._compare_algorithms()
PPLSExperiment._compare_algorithms() → ScalarLikelihoodMethod.fit()
PPLSExperiment._compare_algorithms() → EMAlgorithm.fit()
PPLSExperiment._compare_algorithms() → ECMAlgorithm.fit()
PPLSExperiment.analyze_results() → PerformanceMetrics.compute_mse()
PPLSExperiment.analyze_results() → PerformanceMetrics.generate_summary_table()
PerformanceMetrics.compute_mse() → ParameterRecovery.align_estimated_params()
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple, Optional
import json
import pickle
import os
import time
from datetime import datetime
import warnings
import traceback
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed



_WORKER_CTX: Dict[str, Any] = {}


def _init_trial_worker(
    config: Dict,
    true_params: Dict,
    starting_points: List[np.ndarray],
    p: int,
    q: int,
    r: int,
    results_dir: str,
    save_intermediate: bool,
):
    """Initializer for worker processes (Windows spawn-safe).

    Keeps large shared objects (config / true_params / starting_points) in a module-level
    context to avoid re-sending them for every single trial.
    """
    global _WORKER_CTX
    _WORKER_CTX = {
        "config": config,
        "true_params": true_params,
        "starting_points": starting_points,
        "p": p,
        "q": q,
        "r": r,
        "results_dir": results_dir,
        "save_intermediate": save_intermediate,
    }


def _run_trial_worker(trial_id: int, X: np.ndarray, Y: np.ndarray, seed: int) -> Optional[Dict]:
    """Run one trial in a worker process and return the same trial_result structure."""
    ctx = _WORKER_CTX
    try:
        # Ensure deterministic per-trial randomness in each process.
        np.random.seed(seed)

        config = ctx["config"]
        true_params = ctx["true_params"]
        starting_points = ctx["starting_points"]
        p, q, r = ctx["p"], ctx["q"], ctx["r"]
        results_dir = ctx["results_dir"]
        save_intermediate = bool(ctx.get("save_intermediate", True))

        # Instantiate algorithms inside the process.
        slm_cfg = config.get("algorithms", {}).get("slm_manifold", {})
        slm_manifold = ScalarLikelihoodMethod(
            p=p,
            q=q,
            r=r,
            optimizer=str(slm_cfg.get("optimizer", "manifold")),
            max_iter=int(slm_cfg.get("max_iter", 2000)),
            use_noise_preestimation=bool(slm_cfg.get("use_noise_preestimation", True)),
        )


        slm_interior_cfg = config.get("algorithms", {}).get("slm_interior", {})
        slm_interior = ScalarLikelihoodMethod(
            p=p,
            q=q,
            r=r,
            optimizer="trust-constr",
            max_iter=int(slm_interior_cfg.get("max_iter", 2000)),
            use_noise_preestimation=bool(slm_interior_cfg.get("use_noise_preestimation", True)),
            gtol=float(slm_interior_cfg.get("gtol", 0.005)),
            xtol=float(slm_interior_cfg.get("xtol", 0.005)),
            barrier_tol=float(slm_interior_cfg.get("barrier_tol", 0.005)),
            constraint_slack=float(slm_interior_cfg.get("constraint_slack", 0.005)),
            initial_constr_penalty=float(slm_interior_cfg.get("initial_constr_penalty", 1.0)),
        )

        # Oracle-noise variant: skip closed-form noise pre-estimation and use true (sigma_e^2, sigma_f^2).
        slm_oracle = ScalarLikelihoodMethod(
            p=p,
            q=q,
            r=r,
            optimizer=str(slm_cfg.get("optimizer", "manifold")),
            max_iter=int(slm_cfg.get("max_iter", 2000)),
            use_noise_preestimation=False,
            fixed_sigma_e2=float(true_params.get("sigma_e2")),
            fixed_sigma_f2=float(true_params.get("sigma_f2")),
        )


        # BCD-SLM (paper Algorithm 1): block-coordinate descent with closed-form diagonal updates.
        bcd_cfg = config.get("algorithms", {}).get("bcd_slm", {})
        if not isinstance(bcd_cfg, dict):
            bcd_cfg = {}

        bcd_slm = BCDScalarLikelihoodMethod(
            p=p,
            q=q,
            r=r,
            max_outer_iter=int(bcd_cfg.get("max_outer_iter", 200)),
            n_cg_steps_W=int(bcd_cfg.get("n_cg_steps_W", 5)),
            n_cg_steps_C=int(bcd_cfg.get("n_cg_steps_C", 5)),
            tolerance=float(bcd_cfg.get("tolerance", 1e-4)),
            use_noise_preestimation=bool(bcd_cfg.get("use_noise_preestimation", True)),
        )

        em = EMAlgorithm(
            p=p,
            q=q,
            r=r,
            max_iter=config["algorithms"]["em"]["max_iter"],
            tolerance=config["algorithms"]["em"]["tolerance"],
        )
        ecm = ECMAlgorithm(
            p=p,
            q=q,
            r=r,
            max_iter=config["algorithms"]["ecm"]["max_iter"],
            tolerance=config["algorithms"]["ecm"]["tolerance"],
        )

        def _fallback_estimate() -> Dict[str, Any]:
            return {
                "W": np.zeros((p, r)),
                "C": np.zeros((q, r)),
                "B": np.eye(r),
                "Sigma_t": np.eye(r),
                "sigma_e2": 0.0,
                "sigma_f2": 0.0,
                "sigma_h2": 0.0,
                "objective_value": float("inf"),
                "n_iterations": 0,
                "success": False,
            }

        # Run algorithms and collect timing.
        slm_manifold_start_time = time.time()
        try:
            slm_manifold_results = slm_manifold.fit(X, Y, starting_points)
        except Exception as e:
            warnings.warn(f"SLM-Manifold failed: {e}")
            slm_manifold_results = _fallback_estimate()
        slm_manifold_time = time.time() - slm_manifold_start_time

        bcd_start_time = time.time()
        try:
            bcd_slm_results = bcd_slm.fit(X, Y, starting_points)
        except Exception as e:
            warnings.warn(f"BCD-SLM failed: {e}")
            bcd_slm_results = _fallback_estimate()
        bcd_slm_time = time.time() - bcd_start_time

        slm_interior_start_time = time.time()
        try:
            slm_interior_results = slm_interior.fit(X, Y, starting_points)
        except Exception as e:
            warnings.warn(f"SLM-Interior failed: {e}")
            slm_interior_results = _fallback_estimate()
        slm_interior_time = time.time() - slm_interior_start_time

        slm_oracle_start_time = time.time()
        try:
            slm_oracle_results = slm_oracle.fit(X, Y, starting_points)
        except Exception as e:
            warnings.warn(f"SLM-Oracle failed: {e}")
            slm_oracle_results = _fallback_estimate()
        slm_oracle_time = time.time() - slm_oracle_start_time

        em_start_time = time.time()
        try:
            em_results = em.fit(X, Y, starting_points)
        except Exception as e:
            warnings.warn(f"EM failed: {e}")
            em_results = _fallback_estimate()
            em_results["log_likelihood"] = -np.inf
        em_time = time.time() - em_start_time

        ecm_start_time = time.time()
        try:
            ecm_results = ecm.fit(X, Y, starting_points)
        except Exception as e:
            warnings.warn(f"ECM failed: {e}")
            ecm_results = _fallback_estimate()
            ecm_results["log_likelihood"] = -np.inf
        ecm_time = time.time() - ecm_start_time

        slm_manifold_converged = 1 if slm_manifold_results.get("success", False) else 0
        bcd_slm_converged = 1 if bcd_slm_results.get("success", False) else 0
        slm_interior_converged = 1 if slm_interior_results.get("success", False) else 0
        slm_oracle_converged = 1 if slm_oracle_results.get("success", False) else 0

        em_converged = 1 if em_results.get("log_likelihood", -np.inf) > -np.inf else 0
        ecm_converged = 1 if ecm_results.get("log_likelihood", -np.inf) > -np.inf else 0

        stats_summary = {
            "trial_id": trial_id,
            "slm_manifold": {
                "runtime": slm_manifold_time,
                "avg_time_per_start": slm_manifold_time / len(starting_points),
                "converged": slm_manifold_converged,
                "failed": 1 - slm_manifold_converged,
                "convergence_rate": float(slm_manifold_converged),
                "best_objective": slm_manifold_results.get("objective_value", np.inf),
                "avg_iterations": slm_manifold_results.get("n_iterations", 0),
            },
            "bcd_slm": {
                "runtime": bcd_slm_time,
                "avg_time_per_start": bcd_slm_time / len(starting_points),
                "converged": bcd_slm_converged,
                "failed": 1 - bcd_slm_converged,
                "convergence_rate": float(bcd_slm_converged),
                "best_objective": bcd_slm_results.get("objective_value", np.inf),
                "avg_iterations": bcd_slm_results.get("n_iterations", 0),
            },
            "slm_interior": {
                "runtime": slm_interior_time,
                "avg_time_per_start": slm_interior_time / len(starting_points),
                "converged": slm_interior_converged,
                "failed": 1 - slm_interior_converged,
                "convergence_rate": float(slm_interior_converged),
                "best_objective": slm_interior_results.get("objective_value", np.inf),
                "avg_iterations": slm_interior_results.get("n_iterations", 0),
            },
            "slm_oracle": {
                "runtime": slm_oracle_time,
                "avg_time_per_start": slm_oracle_time / len(starting_points),
                "converged": slm_oracle_converged,
                "failed": 1 - slm_oracle_converged,
                "convergence_rate": float(slm_oracle_converged),
                "best_objective": slm_oracle_results.get("objective_value", np.inf),
                "avg_iterations": slm_oracle_results.get("n_iterations", 0),
            },
            "em": {
                "runtime": em_time,
                "avg_time_per_start": em_time / len(starting_points),
                "converged": em_converged,
                "failed": 1 - em_converged,
                "convergence_rate": float(em_converged),
                "best_likelihood": em_results.get("log_likelihood", -np.inf),
                "avg_iterations": em_results.get("n_iterations", 0),
            },
            "ecm": {
                "runtime": ecm_time,
                "avg_time_per_start": ecm_time / len(starting_points),
                "converged": ecm_converged,
                "failed": 1 - ecm_converged,
                "convergence_rate": float(ecm_converged),
                "best_likelihood": ecm_results.get("log_likelihood", -np.inf),
                "avg_iterations": ecm_results.get("n_iterations", 0),
            },
        }

        if save_intermediate:
            stats_file = os.path.join(results_dir, f"trial_{trial_id:03d}_statistics.json")
            with open(stats_file, "w") as f:
                json.dump(stats_summary, f, indent=4)

        return {
            "trial_id": trial_id,
            "true_params": true_params,
            "slm_manifold_results": slm_manifold_results,
            "bcd_slm_results": bcd_slm_results,
            "slm_interior_results": slm_interior_results,
            "slm_oracle_results": slm_oracle_results,
            "em_results": em_results,
            "ecm_results": ecm_results,
            "data_shape": {"X": X.shape, "Y": Y.shape},
            "statistics": stats_summary,
        }





    except Exception as e:
        # Mirror the sequential behavior: record an error file and return None.
        error_info = {
            "trial_id": trial_id,
            "error_type": type(e).__name__,
            "error_message": str(e),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        try:
            error_file = os.path.join(ctx.get("results_dir", "."), f"trial_{trial_id:03d}_error.json")
            with open(error_file, "w") as f:
                json.dump(error_info, f, indent=4)
        except Exception:
            pass
        return None


from .algorithms import InitialPointGenerator, ScalarLikelihoodMethod, EMAlgorithm, ECMAlgorithm, NoiseEstimator
from .bcd_slm import BCDScalarLikelihoodMethod
from .ppls_model import PPLSModel



class PPLSExperiment:
    """
    Main experiment coordinator for comparing SLM, EM, and ECM algorithms.
    Ensures fair comparison by using identical datasets and starting points.
    """
    
    def __init__(self, config: Dict, base_dir: str, results_dir: str, data_dir: Optional[str] = None):
        """
        Initialize experiment with configuration and simplified directory structure.
        
        Parameters:
        -----------
        config : dict
            Experiment configuration
        base_dir : str
            Base directory containing data
        results_dir : str
            Directory to save results
        data_dir : Optional[str]
            Directory containing generated data files (defaults to base_dir/data)
        """
        self.config = config
        self.base_dir = base_dir
        self.results_dir = results_dir
        self.data_dir = data_dir if data_dir is not None else os.path.join(self.base_dir, 'data')
        
        # Model parameters
        self.p = config['model']['p']
        self.q = config['model']['q']
        self.r = config['model']['r']
        self.n_samples = config['model']['n_samples']
        
        # Algorithm settings
        self.n_starts = config['algorithms']['common']['n_starts']
        self.algorithm_seed = config['algorithms']['common']['random_seed']
        
        # Experiment settings
        self.n_trials = config['experiment']['n_trials']
        self.experiment_seed = config['experiment']['random_seed']
        
        # Load data and ground truth
        self._load_data_and_ground_truth()
        
    def _load_data_and_ground_truth(self):
        """Load experimental data and ground truth parameters from data directory."""
        data_dir = self.data_dir
        
        # Load data arrays.
        # Use memory mapping to avoid loading the full tensor into RAM for high-dimensional runs.
        self.X_trials = np.load(os.path.join(data_dir, 'X_trials.npy'), mmap_mode='r')
        self.Y_trials = np.load(os.path.join(data_dir, 'Y_trials.npy'), mmap_mode='r')
        
        # Load ground truth parameters
        with open(os.path.join(data_dir, 'ground_truth.pkl'), 'rb') as f:
            self.true_params = pickle.load(f)
            
        # Validate data dimensions
        expected_shape = (self.n_trials, self.n_samples)
        if self.X_trials.shape[:2] != expected_shape:
            raise ValueError(f"X data shape mismatch: expected {expected_shape}, got {self.X_trials.shape[:2]}")
        if self.Y_trials.shape[:2] != expected_shape:
            raise ValueError(f"Y data shape mismatch: expected {expected_shape}, got {self.Y_trials.shape[:2]}")
            
        print(f"Loaded data for {self.n_trials} trials")
        print(f"Data directory: {data_dir}")
        print(f"Data dimensions: X{self.X_trials.shape}, Y{self.Y_trials.shape}")
        
    def run_monte_carlo(self) -> Dict:
        """
        Execute Monte Carlo experiment with all three algorithms.
        
        Returns:
        --------
        results : dict
            Dictionary containing all trial results and analysis
        """
        print(f"\n{'='*60}")
        print(f"STARTING MONTE CARLO EXPERIMENT")
        print(f"{'='*60}")
        print(f"Total trials: {self.n_trials}")
        print(f"Model dimensions: p={self.p}, q={self.q}, r={self.r}")
        print(f"Sample size: {self.n_samples}")
        print(f"Starting points per algorithm: {self.n_starts}")
        print(f"Algorithms: SLM-Manifold, BCD-SLM, SLM-Interior, SLM-Oracle, EM, ECM")




        print(f"Ground truth: Fixed for all trials")
        print(f"{'='*60}\n")
        
        # Track timing
        start_time = time.time()
        
        # Generate starting points once for all trials
        init_generator = InitialPointGenerator(
            p=self.p, q=self.q, r=self.r,
            n_starts=self.n_starts,
            random_seed=self.algorithm_seed
        )
        starting_points = init_generator.generate_starting_points()
        print(f"Generated {len(starting_points)} identical starting points for all trials")
        
        # Decide whether to parallelize trials (parameter-estimation stage)
        exp_cfg = self.config.get("experiment", {})
        parallel_trials = bool(exp_cfg.get("parallel_trials", False))
        n_jobs = int(exp_cfg.get("n_jobs", 1))
        if n_jobs == -1:
            n_jobs = os.cpu_count() or 1
        max_workers = max(1, min(n_jobs, self.n_trials))
        save_intermediate = bool(self.config.get("output", {}).get("save_intermediate", True))

        trial_results = []

        if parallel_trials and max_workers > 1:
            print(f"Running trials in parallel (workers={max_workers})...")

            mp_ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=max_workers,
                mp_context=mp_ctx,
                initializer=_init_trial_worker,
                initargs=(
                    self.config,
                    self.true_params,
                    starting_points,
                    self.p,
                    self.q,
                    self.r,
                    self.results_dir,
                    save_intermediate,
                ),
            ) as executor:
                futures = []
                for trial_id in range(self.n_trials):
                    X = self.X_trials[trial_id]
                    Y = self.Y_trials[trial_id]
                    # Deterministic per-trial seed (independent of process scheduling)
                    trial_seed = int(self.experiment_seed) + 2000 + int(trial_id)
                    futures.append(executor.submit(_run_trial_worker, trial_id, X, Y, trial_seed))

                completed = 0
                progress_every = max(1, self.n_trials // 20)
                for fut in as_completed(futures):
                    completed += 1
                    try:
                        result = fut.result()
                    except Exception as e:
                        warnings.warn(f"Trial future failed: {e}")
                        result = None

                    if result is not None:
                        trial_results.append(result)

                    if completed % progress_every == 0 or completed == self.n_trials:
                        print(f"  progress: {completed}/{self.n_trials}")

        else:
            # Run trials sequentially
            print("Running trials sequentially...")

            for trial_id in range(self.n_trials):
                print(f"\nRunning Trial {trial_id+1}/{self.n_trials}...")

                # Get data for this trial
                X = self.X_trials[trial_id]
                Y = self.Y_trials[trial_id]

                result = self._run_single_trial(trial_id, X, Y, self.true_params, starting_points)

                if result is not None:
                    trial_results.append(result)
                    print(f"[OK] Trial {trial_id+1} completed successfully")
                else:
                    print(f"[FAIL] Trial {trial_id+1} failed")

        
        # Filter out failed trials
        valid_results = [r for r in trial_results if r is not None]
        
        # Print summary
        elapsed_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"EXPERIMENT COMPLETED")
        print(f"{'='*60}")
        print(f"Successful trials: {len(valid_results)}/{self.n_trials}")
        print(f"Success rate: {len(valid_results)/self.n_trials*100:.1f}%")
        print(f"Total time: {elapsed_time:.1f} seconds")
        print(f"Average time per trial: {elapsed_time/self.n_trials:.1f} seconds")
        
        # Add runtime statistics summary if available
        if valid_results and 'statistics' in valid_results[0]:
            total_slm_manifold_runtime = sum(r['statistics']['slm_manifold']['runtime'] for r in valid_results)
            total_slm_interior_runtime = sum(r['statistics']['slm_interior']['runtime'] for r in valid_results)
            total_em_runtime = sum(r['statistics']['em']['runtime'] for r in valid_results)
            total_ecm_runtime = sum(r['statistics']['ecm']['runtime'] for r in valid_results)

            avg_slm_manifold_convergence = np.mean([r['statistics']['slm_manifold']['convergence_rate'] for r in valid_results])
            avg_slm_interior_convergence = np.mean([r['statistics']['slm_interior']['convergence_rate'] for r in valid_results])
            avg_em_convergence = np.mean([r['statistics']['em']['convergence_rate'] for r in valid_results])
            avg_ecm_convergence = np.mean([r['statistics']['ecm']['convergence_rate'] for r in valid_results])

            print(f"\nAlgorithm Performance Summary:")
            print(f"  SLM-Manifold: {total_slm_manifold_runtime:.1f}s total, {avg_slm_manifold_convergence*100:.1f}% convergence")
            print(f"  SLM-Interior: {total_slm_interior_runtime:.1f}s total, {avg_slm_interior_convergence*100:.1f}% convergence")
            print(f"  EM:           {total_em_runtime:.1f}s total, {avg_em_convergence*100:.1f}% convergence")
            print(f"  ECM:          {total_ecm_runtime:.1f}s total, {avg_ecm_convergence*100:.1f}% convergence")


        
        print(f"{'='*60}\n")
        
        if len(valid_results) == 0:
            raise RuntimeError("All trials failed. Please check your configuration and data.")
        
        # Analyze results
        print("Analyzing results...")
        analysis = self.analyze_results(valid_results)
        
        # Save experiment summary
        experiment_results = {
            'config': self.config,
            'n_trials_completed': len(valid_results),
            'trial_results': valid_results,
            'analysis': analysis,
            'timing': {
                'total_time': elapsed_time,
                'avg_time_per_trial': elapsed_time / self.n_trials
            },
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'ground_truth_params': self.true_params
        }
        
        self.save_experiment_summary(experiment_results)
        print("[OK] Experiment summary saved\n")
        
        return experiment_results
        
    def _run_single_trial(self, trial_id: int, X: np.ndarray, Y: np.ndarray,
                         true_params: Dict, starting_points: List[np.ndarray]) -> Optional[Dict]:
        """
        Execute one complete trial with all algorithms (SLM, SLM-Oracle, EM, ECM).

        
        Parameters:
        -----------
        trial_id : int
            Trial identifier
        X, Y : np.ndarray
            Data matrices for this trial
        true_params : dict
            Ground truth parameters
        starting_points : List[np.ndarray]
            Starting points for multi-start optimization
            
        Returns:
        --------
        trial_result : dict or None
            Results from this trial, or None if failed
        """
        try:
            print(f"  Data shape: X{X.shape}, Y{Y.shape}")
            
            # Run all three algorithms
            comparison_results = self._compare_algorithms(
                X, Y, starting_points, true_params, trial_id
            )
            
            # Combine results
            trial_result = {
                'trial_id': trial_id,
                'true_params': true_params,
                'slm_manifold_results': comparison_results['slm_manifold'],
                'bcd_slm_results': comparison_results['bcd_slm'],
                'slm_interior_results': comparison_results['slm_interior'],
                'slm_oracle_results': comparison_results['slm_oracle'],
                'em_results': comparison_results['em'],
                'ecm_results': comparison_results['ecm'],
                'data_shape': {'X': X.shape, 'Y': Y.shape},
                'statistics': comparison_results.get('statistics', {})
            }



            
            return trial_result
            
        except Exception as e:
            print(f"  ERROR in Trial {trial_id+1}: {str(e)}")
            warnings.warn(f"Trial {trial_id} failed: {str(e)}")
            
            # Save error information
            error_info = {
                'trial_id': trial_id,
                'error_type': type(e).__name__,
                'error_message': str(e),
                'traceback': traceback.format_exc(),
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            
            error_file = os.path.join(self.results_dir, f"trial_{trial_id:03d}_error.json")
            try:
                with open(error_file, 'w') as f:
                    json.dump(error_info, f, indent=4)
            except:
                pass
                
            return None
            
    def _compare_algorithms(self, X: np.ndarray, Y: np.ndarray,
                          starting_points: List[np.ndarray],
                          true_params: Dict, trial_id: int) -> Dict:
        """
        Compare SLM, EM, and ECM algorithms.
        
        Parameters:
        -----------
        X, Y : np.ndarray
            Data matrices
        starting_points : List[np.ndarray]
            Identical starting points for all algorithms
        true_params : dict
            Ground truth parameters
        trial_id : int
            Trial identifier
            
        Returns:
        --------
        results : dict
            Results from all three algorithms
        """
        # Initialize algorithms
        slm_cfg = self.config.get('algorithms', {}).get('slm_manifold', {})
        slm_manifold = ScalarLikelihoodMethod(
            p=self.p, q=self.q, r=self.r,
            optimizer=str(slm_cfg.get('optimizer', 'manifold')),
            max_iter=int(slm_cfg.get('max_iter', 2000)),
            use_noise_preestimation=bool(slm_cfg.get('use_noise_preestimation', True)),
        )


        slm_interior_cfg = self.config.get('algorithms', {}).get('slm_interior', {})
        slm_interior = ScalarLikelihoodMethod(
            p=self.p, q=self.q, r=self.r,
            optimizer='trust-constr',
            max_iter=int(slm_interior_cfg.get('max_iter', 2000)),
            use_noise_preestimation=bool(slm_interior_cfg.get('use_noise_preestimation', True)),
            gtol=float(slm_interior_cfg.get('gtol', 0.005)),
            xtol=float(slm_interior_cfg.get('xtol', 0.005)),
            barrier_tol=float(slm_interior_cfg.get('barrier_tol', 0.005)),
            constraint_slack=float(slm_interior_cfg.get('constraint_slack', 0.005)),
            initial_constr_penalty=float(slm_interior_cfg.get('initial_constr_penalty', 1.0)),
        )

        # Oracle-noise variant: uses true (sigma_e^2, sigma_f^2).
        slm_oracle = ScalarLikelihoodMethod(
            p=self.p, q=self.q, r=self.r,
            optimizer=str(slm_cfg.get('optimizer', 'manifold')),
            max_iter=int(slm_cfg.get('max_iter', 2000)),
            use_noise_preestimation=False,
            fixed_sigma_e2=float(true_params.get('sigma_e2')),
            fixed_sigma_f2=float(true_params.get('sigma_f2')),
        )


        # BCD-SLM (paper Algorithm 1)
        bcd_cfg = self.config.get('algorithms', {}).get('bcd_slm', {})
        if not isinstance(bcd_cfg, dict):
            bcd_cfg = {}
        bcd_slm = BCDScalarLikelihoodMethod(
            p=self.p,
            q=self.q,
            r=self.r,
            max_outer_iter=int(bcd_cfg.get('max_outer_iter', 200)),
            n_cg_steps_W=int(bcd_cfg.get('n_cg_steps_W', 5)),
            n_cg_steps_C=int(bcd_cfg.get('n_cg_steps_C', 5)),
            tolerance=float(bcd_cfg.get('tolerance', 1e-4)),
            use_noise_preestimation=bool(bcd_cfg.get('use_noise_preestimation', True)),
        )

        em = EMAlgorithm(
            p=self.p, q=self.q, r=self.r,
            max_iter=self.config['algorithms']['em']['max_iter'],
            tolerance=self.config['algorithms']['em']['tolerance']
        )

        ecm = ECMAlgorithm(
            p=self.p, q=self.q, r=self.r,
            max_iter=self.config['algorithms']['ecm']['max_iter'],
            tolerance=self.config['algorithms']['ecm']['tolerance']
        )

        print(f"\n{'='*60}")
        print(f"Trial {trial_id+1}: Algorithm Comparison")
        print(f"{'='*60}")

        print(f"Running SLM-Manifold with {len(starting_points)} starting points...")
        slm_manifold_start_time = time.time()
        slm_manifold_results, slm_manifold_stats = self._run_algorithm_with_stats(
            slm_manifold, X, Y, starting_points, trial_id, "SLM-Manifold"
        )
        slm_manifold_time = time.time() - slm_manifold_start_time
        print(
            f"SLM-Manifold completed: {slm_manifold_time:.2f}s, "
            f"convergence: {slm_manifold_stats['convergence_rate']*100:.1f}%"
        )

        print(f"Running BCD-SLM with {len(starting_points)} starting points...")
        bcd_start_time = time.time()
        bcd_slm_results, bcd_slm_stats = self._run_algorithm_with_stats(
            bcd_slm, X, Y, starting_points, trial_id, "BCD-SLM"
        )
        bcd_slm_time = time.time() - bcd_start_time
        print(
            f"BCD-SLM completed: {bcd_slm_time:.2f}s, "
            f"convergence: {bcd_slm_stats['convergence_rate']*100:.1f}%"
        )

        print(f"Running SLM-Interior with {len(starting_points)} starting points...")
        slm_interior_start_time = time.time()
        slm_interior_results, slm_interior_stats = self._run_algorithm_with_stats(
            slm_interior, X, Y, starting_points, trial_id, "SLM-Interior"
        )
        slm_interior_time = time.time() - slm_interior_start_time
        print(
            f"SLM-Interior completed: {slm_interior_time:.2f}s, "
            f"convergence: {slm_interior_stats['convergence_rate']*100:.1f}%"
        )

        print(f"Running SLM-Oracle with {len(starting_points)} starting points...")
        slm_oracle_start_time = time.time()
        slm_oracle_results, slm_oracle_stats = self._run_algorithm_with_stats(
            slm_oracle, X, Y, starting_points, trial_id, "SLM-Oracle"
        )
        slm_oracle_time = time.time() - slm_oracle_start_time
        print(
            f"SLM-Oracle completed: {slm_oracle_time:.2f}s, "
            f"convergence: {slm_oracle_stats['convergence_rate']*100:.1f}%"
        )

        print(f"Running EM with {len(starting_points)} starting points...")
        em_start_time = time.time()
        em_results, em_stats = self._run_algorithm_with_stats(
            em, X, Y, starting_points, trial_id, "EM"
        )
        em_time = time.time() - em_start_time
        print(f"EM completed: {em_time:.2f}s, convergence: {em_stats['convergence_rate']*100:.1f}%")

        print(f"Running ECM with {len(starting_points)} starting points...")
        ecm_start_time = time.time()
        ecm_results, ecm_stats = self._run_algorithm_with_stats(
            ecm, X, Y, starting_points, trial_id, "ECM"
        )
        ecm_time = time.time() - ecm_start_time
        print(f"ECM completed: {ecm_time:.2f}s, convergence: {ecm_stats['convergence_rate']*100:.1f}%")

        print(f"{'='*60}")

        stats_summary = {
            'trial_id': trial_id,
            'slm_manifold': {
                'runtime': slm_manifold_time,
                'avg_time_per_start': slm_manifold_time/len(starting_points),
                'converged': slm_manifold_stats['converged'],
                'failed': slm_manifold_stats['failed'],
                'convergence_rate': float(slm_manifold_stats['converged']),
                'best_objective': slm_manifold_stats['best_objective'],
                'avg_iterations': slm_manifold_stats['avg_iterations']
            },
            'bcd_slm': {
                'runtime': bcd_slm_time,
                'avg_time_per_start': bcd_slm_time/len(starting_points),
                'converged': bcd_slm_stats['converged'],
                'failed': bcd_slm_stats['failed'],
                'convergence_rate': float(bcd_slm_stats['converged']),
                'best_objective': bcd_slm_stats['best_objective'],
                'avg_iterations': bcd_slm_stats['avg_iterations']
            },
            'slm_interior': {
                'runtime': slm_interior_time,
                'avg_time_per_start': slm_interior_time/len(starting_points),
                'converged': slm_interior_stats['converged'],
                'failed': slm_interior_stats['failed'],
                'convergence_rate': float(slm_interior_stats['converged']),
                'best_objective': slm_interior_stats['best_objective'],
                'avg_iterations': slm_interior_stats['avg_iterations']
            },
            'slm_oracle': {
                'runtime': slm_oracle_time,
                'avg_time_per_start': slm_oracle_time/len(starting_points),
                'converged': slm_oracle_stats['converged'],
                'failed': slm_oracle_stats['failed'],
                'convergence_rate': float(slm_oracle_stats['converged']),
                'best_objective': slm_oracle_stats['best_objective'],
                'avg_iterations': slm_oracle_stats['avg_iterations']
            },
            'em': {
                'runtime': em_time,
                'avg_time_per_start': em_time/len(starting_points),
                'converged': em_stats['converged'],
                'failed': em_stats['failed'],
                'convergence_rate': float(em_stats['converged']),
                'best_likelihood': em_stats['best_likelihood'],
                'avg_iterations': em_stats['avg_iterations']
            },
            'ecm': {
                'runtime': ecm_time,
                'avg_time_per_start': ecm_time/len(starting_points),
                'converged': ecm_stats['converged'],
                'failed': ecm_stats['failed'],
                'convergence_rate': float(ecm_stats['converged']),
                'best_likelihood': ecm_stats['best_likelihood'],
                'avg_iterations': ecm_stats['avg_iterations']
            }
        }

        if self.config.get('output', {}).get('save_intermediate', True):
            stats_file = os.path.join(self.results_dir, f"trial_{trial_id:03d}_statistics.json")
            with open(stats_file, 'w') as f:
                json.dump(stats_summary, f, indent=4)

        return {
            'slm_manifold': slm_manifold_results,
            'bcd_slm': bcd_slm_results,
            'slm_interior': slm_interior_results,
            'slm_oracle': slm_oracle_results,
            'em': em_results,
            'ecm': ecm_results,
            'statistics': stats_summary
        }




        
    def _run_algorithm_with_stats(self, algorithm, X: np.ndarray, Y: np.ndarray,
                                 starting_points: List[np.ndarray],
                                 trial_id: int, algorithm_name: str) -> Tuple[Dict, Dict]:
        """
        Run algorithm and collect detailed statistics.
        
        Parameters:
        -----------
        algorithm : PPLSAlgorithm
            Algorithm instance (SLM, EM, or ECM)
        X, Y : np.ndarray
            Data matrices
        starting_points : List[np.ndarray]
            Starting points for multi-start
        trial_id : int
            Trial identifier
        algorithm_name : str
            Name of algorithm ("SLM", "EM", or "ECM")
            
        Returns:
        --------
        results : dict
            Best estimated parameters
        stats : dict
            Running statistics
        """
        algo_lower = str(algorithm_name).lower()

        def _fallback_estimate() -> Dict[str, Any]:
            return {
                'W': np.zeros((self.p, self.r)),
                'C': np.zeros((self.q, self.r)),
                'B': np.eye(self.r),
                'Sigma_t': np.eye(self.r),
                'sigma_e2': 0.0,
                'sigma_f2': 0.0,
                'sigma_h2': 0.0,
                'objective_value': float('inf'),
                'n_iterations': 0,
                'success': False,
            }

        # Use the algorithm's fit method with try/except guard.
        try:
            results = algorithm.fit(X, Y, starting_points)
        except Exception as e:
            warnings.warn(f"{algorithm_name} failed: {e}")
            results = _fallback_estimate()
            if algo_lower in ("em", "ecm"):
                results['log_likelihood'] = -np.inf

        # Compile basic statistics
        is_objective = ('slm' in algo_lower) or ('bcd' in algo_lower) or ('objective_value' in results)
        if is_objective:
            stats = {
                'converged': 1 if results.get('success', False) else 0,
                'failed': 0 if results.get('success', False) else 1,
                'convergence_rate': 1.0 if results.get('success', False) else 0.0,
                'avg_iterations': results.get('n_iterations', 0),
                'best_objective': results.get('objective_value', np.inf)
            }
        else:  # EM / ECM
            stats = {
                'converged': 1 if results.get('log_likelihood', -np.inf) > -np.inf else 0,
                'failed': 0 if results.get('log_likelihood', -np.inf) > -np.inf else 1,
                'convergence_rate': 1.0 if results.get('log_likelihood', -np.inf) > -np.inf else 0.0,
                'avg_iterations': results.get('n_iterations', 0),
                'best_likelihood': results.get('log_likelihood', -np.inf)
            }

        return results, stats

        
    def analyze_results(self, trial_results: List[Dict]) -> Dict:
        """
        Compute summary statistics and performance metrics across all trials.
        
        Parameters:
        -----------
        trial_results : List[Dict]
            Results from all successful trials
            
        Returns:
        --------
        analysis : dict
            Summary statistics and performance comparison
        """
        # Initialize metrics calculators
        slm_manifold_metrics_list = []
        bcd_slm_metrics_list = []
        slm_interior_metrics_list = []
        slm_oracle_metrics_list = []
        em_metrics_list = []
        ecm_metrics_list = []

        # Collect runtime statistics
        slm_manifold_runtime_stats = []
        bcd_slm_runtime_stats = []
        slm_interior_runtime_stats = []
        slm_oracle_runtime_stats = []
        em_runtime_stats = []
        ecm_runtime_stats = []

        # Collect metrics for each trial
        for trial in trial_results:
            metrics_calc = PerformanceMetrics(trial['true_params'])

            slm_manifold_metrics = metrics_calc.compute_mse(trial['slm_manifold_results'])
            slm_manifold_metrics_list.append(slm_manifold_metrics)

            bcd_res = trial.get('bcd_slm_results', None)
            if isinstance(bcd_res, dict) and bcd_res:
                bcd_metrics = metrics_calc.compute_mse(bcd_res)
                bcd_slm_metrics_list.append(bcd_metrics)

            slm_int_res = trial.get('slm_interior_results', None)
            if isinstance(slm_int_res, dict) and slm_int_res:
                slm_int_metrics = metrics_calc.compute_mse(slm_int_res)
                slm_interior_metrics_list.append(slm_int_metrics)

            if 'slm_oracle_results' in trial:
                slm_oracle_metrics = metrics_calc.compute_mse(trial['slm_oracle_results'])
                slm_oracle_metrics_list.append(slm_oracle_metrics)

            em_metrics = metrics_calc.compute_mse(trial['em_results'])
            em_metrics_list.append(em_metrics)

            ecm_metrics = metrics_calc.compute_mse(trial['ecm_results'])
            ecm_metrics_list.append(ecm_metrics)

            if 'statistics' in trial:
                slm_manifold_runtime_stats.append(trial['statistics']['slm_manifold'])
                if 'bcd_slm' in trial['statistics']:
                    bcd_slm_runtime_stats.append(trial['statistics']['bcd_slm'])
                if 'slm_interior' in trial['statistics']:
                    slm_interior_runtime_stats.append(trial['statistics']['slm_interior'])
                if 'slm_oracle' in trial['statistics']:
                    slm_oracle_runtime_stats.append(trial['statistics']['slm_oracle'])
                em_runtime_stats.append(trial['statistics']['em'])
                ecm_runtime_stats.append(trial['statistics']['ecm'])

        # Aggregate metrics
        analysis = {
            'slm_manifold': self._aggregate_metrics(slm_manifold_metrics_list),
            'bcd_slm': self._aggregate_metrics(bcd_slm_metrics_list),
            'slm_interior': self._aggregate_metrics(slm_interior_metrics_list),
            'slm_oracle': self._aggregate_metrics(slm_oracle_metrics_list),
            'em': self._aggregate_metrics(em_metrics_list),
            'ecm': self._aggregate_metrics(ecm_metrics_list),
            'comparison': self._compare_methods(slm_manifold_metrics_list, em_metrics_list, ecm_metrics_list)
        }

        # Add runtime statistics
        if slm_manifold_runtime_stats and em_runtime_stats and ecm_runtime_stats:
            runtime_statistics = {
                'slm_manifold': self._aggregate_runtime_stats(slm_manifold_runtime_stats),
                'em': self._aggregate_runtime_stats(em_runtime_stats),
                'ecm': self._aggregate_runtime_stats(ecm_runtime_stats),
                'overall': {
                    'total_runtime_slm_manifold': sum(s['runtime'] for s in slm_manifold_runtime_stats),
                    'total_runtime_em': sum(s['runtime'] for s in em_runtime_stats),
                    'total_runtime_ecm': sum(s['runtime'] for s in ecm_runtime_stats)
                }
            }

            if bcd_slm_runtime_stats:
                runtime_statistics['bcd_slm'] = self._aggregate_runtime_stats(bcd_slm_runtime_stats)
                runtime_statistics['overall']['total_runtime_bcd_slm'] = sum(
                    s['runtime'] for s in bcd_slm_runtime_stats if s.get('runtime') is not None
                )

            if slm_interior_runtime_stats:
                runtime_statistics['slm_interior'] = self._aggregate_runtime_stats(slm_interior_runtime_stats)
                runtime_statistics['overall']['total_runtime_slm_interior'] = sum(
                    s['runtime'] for s in slm_interior_runtime_stats if s.get('runtime') is not None
                )

            if slm_oracle_runtime_stats:
                runtime_statistics['slm_oracle'] = self._aggregate_runtime_stats(slm_oracle_runtime_stats)
                runtime_statistics['overall']['total_runtime_slm_oracle'] = sum(
                    s['runtime'] for s in slm_oracle_runtime_stats
                )

            analysis['runtime_statistics'] = runtime_statistics

        # Generate summary table
        if trial_results:
            metrics_calc = PerformanceMetrics(trial_results[0]['true_params'])
            summary_table = metrics_calc.generate_summary_table(
                slm_manifold_metrics_list, bcd_slm_metrics_list, em_metrics_list, ecm_metrics_list
            )
            analysis['summary_table'] = summary_table

        return analysis

        
    def _aggregate_metrics(self, metrics_list: List[Dict]) -> Dict:
        """Aggregate metrics across trials."""
        if not metrics_list:
            return {}
            
        # Extract metric names
        metric_names = list(metrics_list[0].keys())
        
        aggregated = {}
        for metric in metric_names:
            values = [m[metric] for m in metrics_list]
            aggregated[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
            
        return aggregated
        
    def _aggregate_runtime_stats(self, stats_list: List[Dict]) -> Dict:
        """Aggregate runtime statistics across trials."""
        if not stats_list:
            return {}
            
        aggregated = {
            'avg_runtime': np.mean([s['runtime'] for s in stats_list]),
            'std_runtime': np.std([s['runtime'] for s in stats_list]),
            'avg_time_per_start': np.mean([s['avg_time_per_start'] for s in stats_list]),
            'avg_convergence_rate': np.mean([s['convergence_rate'] for s in stats_list]),
            'total_converged': sum(s['converged'] for s in stats_list),
            'total_failed': sum(s['failed'] for s in stats_list),
            'avg_iterations': np.mean([s['avg_iterations'] for s in stats_list])
        }
        
        return aggregated
        
    def _compare_methods(self, slm_metrics: List[Dict], em_metrics: List[Dict], 
                        ecm_metrics: List[Dict]) -> Dict:
        """Compare performance between SLM, EM, and ECM."""
        comparison = {}
        
        if not slm_metrics or not em_metrics or not ecm_metrics:
            return comparison
            
        # Compare each metric
        metric_names = list(slm_metrics[0].keys())
        
        for metric in metric_names:
            slm_values = np.array([m[metric] for m in slm_metrics])
            em_values = np.array([m[metric] for m in em_metrics])
            ecm_values = np.array([m[metric] for m in ecm_metrics])
            
            comparison[metric] = {
                'slm_best_count': np.sum((slm_values <= em_values) & (slm_values <= ecm_values)),
                'em_best_count': np.sum((em_values <= slm_values) & (em_values <= ecm_values)),
                'ecm_best_count': np.sum((ecm_values <= slm_values) & (ecm_values <= em_values)),
                'slm_mean': np.mean(slm_values),
                'em_mean': np.mean(em_values),
                'ecm_mean': np.mean(ecm_values)
            }
            
        return comparison
        
    def save_experiment_summary(self, results: Dict):
        """Save overall experiment results."""
        # Save comprehensive experiment results
        with open(os.path.join(self.results_dir, "experiment_results.pkl"), 'wb') as f:
            pickle.dump(results, f)
            
        # Save human-readable experiment summary
        readable_summary = {
            'experiment_info': {
                'n_trials_completed': results['n_trials_completed'],
                'success_rate_percent': round(results['n_trials_completed'] / self.n_trials * 100, 1),
                'total_runtime_minutes': round(results['timing']['total_time'] / 60, 2),
                'timestamp': results['timestamp']
            },
            'algorithm_performance': {},
            'parameter_estimation_quality': {}
        }
        
        # Add algorithm performance comparison
        if 'runtime_statistics' in results['analysis']:
            runtime = results['analysis']['runtime_statistics']
            readable_summary['algorithm_performance'] = {
                'slm_manifold': {
                    'avg_time_seconds': round(runtime.get('slm_manifold', {}).get('avg_runtime', 0), 2),
                    'avg_convergence_rate_percent': round(runtime.get('slm_manifold', {}).get('avg_convergence_rate', 0) * 100, 1)
                },
                'bcd_slm': {
                    'avg_time_seconds': round(runtime.get('bcd_slm', {}).get('avg_runtime', 0), 2),
                    'avg_convergence_rate_percent': round(runtime.get('bcd_slm', {}).get('avg_convergence_rate', 0) * 100, 1)
                },
                'slm_interior': {
                    'avg_time_seconds': round(runtime.get('slm_interior', {}).get('avg_runtime', 0), 2),
                    'avg_convergence_rate_percent': round(runtime.get('slm_interior', {}).get('avg_convergence_rate', 0) * 100, 1)
                },
                'slm_oracle': {
                    'avg_time_seconds': round(runtime.get('slm_oracle', {}).get('avg_runtime', 0), 2),
                    'avg_convergence_rate_percent': round(runtime.get('slm_oracle', {}).get('avg_convergence_rate', 0) * 100, 1)
                },
                'em': {
                    'avg_time_seconds': round(runtime.get('em', {}).get('avg_runtime', 0), 2),
                    'avg_convergence_rate_percent': round(runtime.get('em', {}).get('avg_convergence_rate', 0) * 100, 1)
                },
                'ecm': {
                    'avg_time_seconds': round(runtime.get('ecm', {}).get('avg_runtime', 0), 2),
                    'avg_convergence_rate_percent': round(runtime.get('ecm', {}).get('avg_convergence_rate', 0) * 100, 1)
                }
            }

        # Add MSE comparison summary
        for method in ['slm_manifold', 'bcd_slm', 'slm_interior', 'slm_oracle', 'em', 'ecm']:
            if method in results['analysis']:
                method_mse = {}
                for param in ['W', 'C', 'B', 'Sigma_t', 'sigma_h2']:
                    key = f'mse_{param}'
                    if key in results['analysis'][method]:
                        method_mse[param] = {
                            'mean': round(results['analysis'][method][key]['mean'], 6),
                            'std': round(results['analysis'][method][key]['std'], 6)
                        }
                readable_summary['parameter_estimation_quality'][method] = method_mse

        
        with open(os.path.join(self.results_dir, "experiment_summary.json"), 'w') as f:
            json.dump(readable_summary, f, indent=2)
            

class PerformanceMetrics:
    """
    Calculate estimation quality metrics for PPLS parameter recovery.
    """
    
    def __init__(self, true_params: Dict):
        """
        Initialize with ground truth parameters.
        
        Parameters:
        -----------
        true_params : dict
            True parameter values
        """
        self.true_params = true_params
        self.recovery = ParameterRecovery()
        
    def compute_mse(self, estimated_params: Dict) -> Dict:
        """
        Compute mean squared error for all parameters with proper sign alignment.
        
        Parameters:
        -----------
        estimated_params : dict
            Estimated parameter values
            
        Returns:
        --------
        mse_dict : dict
            MSE for each parameter
        """
        # Align parameters to handle sign indeterminacy
        aligned_params = self.recovery.align_estimated_params(
            estimated_params, self.true_params
        )
        
        mse_dict = {}
        
        # MSE for loading matrices
        mse_dict['mse_W'] = np.mean((aligned_params['W'] - self.true_params['W'])**2)
        mse_dict['mse_C'] = np.mean((aligned_params['C'] - self.true_params['C'])**2)
        
        # MSE for diagonal matrices
        mse_dict['mse_B'] = np.mean(
            (np.diag(aligned_params['B']) - np.diag(self.true_params['B']))**2
        )
        mse_dict['mse_Sigma_t'] = np.mean(
            (np.diag(aligned_params['Sigma_t']) - np.diag(self.true_params['Sigma_t']))**2
        )
        
        # MSE for scalar parameters
        mse_dict['mse_sigma_h2'] = (
            aligned_params['sigma_h2'] - self.true_params['sigma_h2']
        )**2

        # Also track noise variances when present (all our algorithms return them).
        if 'sigma_e2' in aligned_params and 'sigma_e2' in self.true_params:
            mse_dict['mse_sigma_e2'] = (
                float(aligned_params['sigma_e2']) - float(self.true_params['sigma_e2'])
            )**2
        if 'sigma_f2' in aligned_params and 'sigma_f2' in self.true_params:
            mse_dict['mse_sigma_f2'] = (
                float(aligned_params['sigma_f2']) - float(self.true_params['sigma_f2'])
            )**2
        
        return mse_dict

        
    def generate_summary_table(
        self,
        slm_metrics_list: List[Dict],
        bcd_slm_metrics_list: List[Dict],
        em_metrics_list: List[Dict],
        ecm_metrics_list: List[Dict],
    ) -> pd.DataFrame:
        r"""Generate summary table (paper-style parameter MSE, \(\times 10^2\))."""
        params = ['W', 'C', 'B', 'Sigma_t', 'sigma_h2']

        table_data = []
        for param in params:
            mse_key = f'mse_{param}'

            # SLM-Fixed
            slm_values = [m[mse_key] for m in slm_metrics_list if mse_key in m]
            slm_mean = np.mean(slm_values) if slm_values else np.nan
            slm_std = np.std(slm_values) if slm_values else np.nan

            # BCD-SLM
            bcd_values = [m[mse_key] for m in bcd_slm_metrics_list if mse_key in m]
            bcd_mean = np.mean(bcd_values) if bcd_values else np.nan
            bcd_std = np.std(bcd_values) if bcd_values else np.nan

            # EM
            em_values = [m[mse_key] for m in em_metrics_list if mse_key in m]
            em_mean = np.mean(em_values) if em_values else np.nan
            em_std = np.std(em_values) if em_values else np.nan

            # ECM
            ecm_values = [m[mse_key] for m in ecm_metrics_list if mse_key in m]
            ecm_mean = np.mean(ecm_values) if ecm_values else np.nan
            ecm_std = np.std(ecm_values) if ecm_values else np.nan

            table_data.append({
                'Parameter': param,
                'SLM-Fixed': f"{slm_mean*100:.2f}±{slm_std*100:.2f}",
                'BCD-SLM': f"{bcd_mean*100:.2f}±{bcd_std*100:.2f}",
                'EM': f"{em_mean*100:.2f}±{em_std*100:.2f}",
                'ECM': f"{ecm_mean*100:.2f}±{ecm_std*100:.2f}",
            })

        return pd.DataFrame(table_data)



class ParameterRecovery:
    """
    Assess parameter recovery quality and handle identifiability issues.
    """
    
    def align_estimated_params(self, params_est: Dict, 
                              params_true: Dict) -> Dict:
        """
        Align estimated parameters with ground truth to handle sign indeterminacy.
        
        Parameters:
        -----------
        params_est : dict
            Estimated parameters
        params_true : dict
            True parameters
            
        Returns:
        --------
        aligned_params : dict
            Sign-aligned parameters
        """
        aligned_params = params_est.copy()
        
        W_est = params_est['W']
        C_est = params_est['C']
        B_est = params_est['B']
        
        W_true = params_true['W']
        C_true = params_true['C']
        
        r = W_true.shape[1]
        
        # Align each component
        for i in range(r):
            # Check correlation with true loadings
            w_corr = np.corrcoef(W_est[:, i], W_true[:, i])[0, 1]
            c_corr = np.corrcoef(C_est[:, i], C_true[:, i])[0, 1]
            
            # Flip signs if negative correlation
            if w_corr < 0:
                W_est[:, i] *= -1
            if c_corr < 0:
                C_est[:, i] *= -1
                
        # Ensure B diagonal elements are positive
        b_diag = np.diag(B_est)
        B_aligned = np.diag(np.abs(b_diag))
        
        aligned_params['W'] = W_est
        aligned_params['C'] = C_est
        aligned_params['B'] = B_aligned
        
        return aligned_params