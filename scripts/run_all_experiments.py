"""One-click runner for all paper experiments.

Runs (in order):
  1) Monte Carlo simulation pipeline (ppls_slm.cli.montecarlo)
  2) Speed experiment (ppls_slm.benchmarks.speed_experiment)
  3) Association application (ppls_slm.apps.association_analysis)
  4) Prediction application (synthetic; ppls_slm.apps.prediction)
  5) BRCA prediction benchmark (ppls_slm.apps.brca_prediction) [optional]
  6) BRCA calibration benchmark (ppls_slm.apps.brca_calibration) [optional]
  7) CITE-seq prediction benchmark (ppls_slm.apps.citeseq_prediction) [optional]
  8) PCCA simulation (ppls_slm.apps.pcca_simulation) [optional]
  9) PPCA verification (ppls_slm.apps.ppca_verification) [optional]
  10) Sync small paper artifacts into paper/artifacts (scripts/sync_artifacts.py)


Usage (from repo root):
  python scripts/run_all_experiments.py

Notes
-----
- This script temporarily sets config.json output.force_* = true to ensure a real re-run.
  It restores the previous values at the end (even if a step fails).
- Logs are written under: output/logs/one_click/
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional




def _get_total_memory_gb() -> Optional[float]:
    """Best-effort total physical memory (GB). Works on Windows without extra deps."""
    try:
        import ctypes
        from ctypes import wintypes

        M = type(
            "MEMORYSTATUSEX",
            (ctypes.Structure,),
            {
                "_fields_": [
                    ("dwLength", wintypes.DWORD),
                    ("dwMemoryLoad", wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            },
        )

        m = M()
        m.dwLength = ctypes.sizeof(M)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return float(m.ullTotalPhys) / (1024**3)

    except Exception:
        return None


def _recommend_n_jobs() -> int:
    """Heuristic for trial-parallel Monte Carlo on *this* machine.

    Rationale:
    - We run CPU-heavy SciPy/Numpy code per trial.
    - We limit BLAS/OMP threads to 1 (see _thread_limited_env), so process count maps well to cores.
    - Leave some headroom for OS / background tasks.
    """
    cpu = os.cpu_count() or 1
    mem_gb = _get_total_memory_gb()

    # Base: ~85% of CPU, leaving headroom.
    base = max(1, int(round(cpu * 0.85)))

    # Memory-constrained machines: be more conservative.
    if mem_gb is not None and mem_gb < 16.0:
        cap = max(1, cpu // 2)
    else:
        cap = max(1, cpu - 2)

    return max(1, min(base, cap, cpu))


def _thread_limited_env() -> dict:
    """Return env dict limiting BLAS/OMP threads to 1 to avoid oversubscription."""
    env = dict(os.environ)

    # Ensure real-time logs even when stdout is piped (tee_run uses stdout=PIPE).
    env["PYTHONUNBUFFERED"] = "1"

    for k in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        env[k] = "1"
    return env








def tee_run(cmd: List[str], cwd: Path, log_path: Path, env: Optional[dict] = None) -> int:
    """Run a command, streaming stdout/stderr to both console and a log file."""

    log_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"RUN  : {' '.join(cmd)}")
    print(f"CWD  : {cwd}")
    print(f"LOG  : {log_path}")
    print("=" * 80)

    with log_path.open("w", encoding="utf-8", errors="replace") as f:
        f.write(f"[{datetime.now().isoformat(timespec='seconds')}] CMD: {' '.join(cmd)}\n")
        f.write(f"CWD: {cwd}\n\n")
        f.flush()

        # Merge stderr into stdout for ordered logs
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )


        assert proc.stdout is not None
        for line in proc.stdout:
            # Windows consoles often use a GBK/CP936 code page.
            # The subprocess output may contain characters that cannot be encoded.
            # Write via the binary buffer with errors='replace' to avoid crashing.
            try:
                if hasattr(sys.stdout, "buffer") and sys.stdout.encoding:
                    sys.stdout.buffer.write(line.encode(sys.stdout.encoding, errors="replace"))
                else:
                    sys.stdout.write(line)
            except Exception:
                # Last-resort fallback: drop unencodable characters
                sys.stdout.write(line.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore"))

            # Ensure real-time visibility in the console and log file.
            try:
                sys.stdout.flush()
            except Exception:
                pass

            f.write(line)
            f.flush()


        proc.wait()

        f.write(f"\n[{datetime.now().isoformat(timespec='seconds')}] EXIT: {proc.returncode}\n")
        f.flush()

        return int(proc.returncode)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def set_force_flags(config: dict, value: bool) -> dict:

    cfg = dict(config)
    out = dict(cfg.get("output", {}))
    out["force_data_generation"] = bool(value)
    out["force_parameter_estimation"] = bool(value)
    out["force_visualization"] = bool(value)
    cfg["output"] = out
    return cfg


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="One-click runner for all experiments + artifact sync")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config JSON (default: config.json)")

    # These flags override config-driven selection
    parser.add_argument("--skip-montecarlo", action="store_true", help="Skip Monte Carlo pipeline")
    parser.add_argument("--skip-speed", action="store_true", help="Skip speed experiment")
    parser.add_argument("--skip-association", action="store_true", help="Skip association application")
    parser.add_argument("--skip-prediction", action="store_true", help="Skip prediction application")
    parser.add_argument("--skip-brca", action="store_true", help="Skip BRCA prediction + calibration")
    parser.add_argument("--skip-citeseq", action="store_true", help="Skip CITE-seq prediction benchmark")


    parser.add_argument("--no-sync", action="store_true", help="Do not run scripts/sync_artifacts.py")


    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete previous generated outputs (recommended after code changes)",
    )
    parser.add_argument(
        "--clean-artifacts",
        action="store_true",
        help="Also delete paper/artifacts before re-running (implies --clean)",
    )

    # Association app optional inputs (override config)
    parser.add_argument("--gene-expr", type=str, default=None, help="Path to gene expression file for association app")
    parser.add_argument("--protein-expr", type=str, default=None, help="Path to protein expression file for association app")
    parser.add_argument("--brca-data", type=str, default=None, help="Path to bundled BRCA combined dataset (.csv or .zip)")

    # Output dirs for apps (override config)
    parser.add_argument("--assoc-out", type=str, default=None, help="Association output dir")
    parser.add_argument("--pred-out", type=str, default=None, help="Prediction output dir")
    parser.add_argument("--noise-out", type=str, default=None, help="Noise ablation output dir")

    args = parser.parse_args(list(argv) if argv is not None else None)

    root = Path(__file__).resolve().parents[1]
    config_path = (root / args.config).resolve()




    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_dir = root / "output" / "logs" / "one_click" / ts

    logs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("One-click experiment runner")
    print(f"Repo root : {root}")

    print(f"Logs dir  : {logs_dir}")

    recommended_jobs = _recommend_n_jobs()
    run_env = _thread_limited_env()
    print(f"CPU cores : {os.cpu_count()}")
    total_mem = _get_total_memory_gb()
    if total_mem is not None:
        print(f"Memory    : {total_mem:.1f} GB")
    print(f"n_jobs(recommended) : {recommended_jobs}")
    print("BLAS/OMP threads    : 1 (via env)")

    print("=" * 80)


    # Load config and determine which experiments to run
    original_config = None
    try:
        if config_path.exists():
            original_config = load_json(config_path)
        else:
            raise FileNotFoundError(f"Missing config file at {config_path}")

        exp_cfg = original_config.get("experiments", {})
        run_cfg = exp_cfg.get("run", {})

        run_montecarlo = bool(run_cfg.get("montecarlo", True)) and (not args.skip_montecarlo)
        run_speed = bool(run_cfg.get("speed", True)) and (not args.skip_speed)
        run_association = bool(run_cfg.get("association", True)) and (not args.skip_association)
        run_prediction = bool(run_cfg.get("prediction", True)) and (not args.skip_prediction)
        run_brca_prediction = bool(run_cfg.get("brca_prediction", False)) and (not args.skip_brca)
        run_brca_calibration = bool(run_cfg.get("brca_calibration", False)) and (not args.skip_brca)
        run_citeseq_prediction = bool(run_cfg.get("citeseq_prediction", False)) and (not args.skip_citeseq)
        run_pcca_simulation = bool(run_cfg.get("pcca_simulation", False))

        run_ppca_verification = bool(run_cfg.get("ppca_verification", False))
        run_sync = bool(run_cfg.get("sync_artifacts", True)) and (not args.no_sync)



        # App-specific config
        assoc_cfg = exp_cfg.get("association", {})
        pred_cfg = exp_cfg.get("prediction", {})

        assoc_out = args.assoc_out or assoc_cfg.get("output_dir", "results_association")
        pred_out = args.pred_out or pred_cfg.get("output_dir", "results_prediction")


        # Prefer CLI override; otherwise use config; otherwise fall back to bundled zip if present.
        default_brca_zip = root / "application" / "brca_data_w_subtypes.csv.zip"

        brca_data = args.brca_data or assoc_cfg.get("brca_data") or (str(default_brca_zip) if default_brca_zip.exists() else None)

        def _rm_tree(p: Path) -> None:
            if p.exists():
                shutil.rmtree(p, ignore_errors=True)

        if args.clean_artifacts:
            args.clean = True

        if args.clean:
            # Clean common outputs to avoid mixing new results with stale files.
            # NOTE: We deliberately do NOT touch `.codebuddy/`.
            out_base = Path(original_config.get("output", {}).get("base_dir") or (root / "output")).resolve()
            _rm_tree(out_base / "data")
            _rm_tree(out_base / "data_high")
            _rm_tree(out_base / "results")
            _rm_tree(out_base / "results_high")
            _rm_tree(out_base / "figures")
            _rm_tree(out_base / "figures_high")

            _rm_tree(out_base / "pcca_simulation")
            _rm_tree(out_base / "ppca_verification")

            _rm_tree((root / str(assoc_out)).resolve())
            _rm_tree((root / str(pred_out)).resolve())
            _rm_tree((root / "results_prediction_brca").resolve())
            _rm_tree((root / "results_citeseq").resolve())

            if args.clean_artifacts:
                _rm_tree(root / "paper" / "artifacts")


            print("\n[OK] Cleaned previous outputs.")

        if run_montecarlo:
            forced = set_force_flags(original_config, True)
            save_json(config_path, forced)
            print("\n[OK] config force flags set to true for re-run.")


        start = time.time()

        # 1) Monte Carlo pipeline
        if run_montecarlo:
            code = tee_run([sys.executable, "-m", "ppls_slm.cli.montecarlo"], cwd=root, log_path=logs_dir / "01_main.log", env=run_env)
            if code != 0:
                return code

        # 2) Speed experiment
        if run_speed:
            code = tee_run(
                [sys.executable, "-u", "-m", "ppls_slm.benchmarks.speed_experiment"],
                cwd=root,
                log_path=logs_dir / "02_speed_experiment.log",
                env=run_env,
            )
            if code != 0:
                return code

        # 3) Association analysis
        if run_association:
            cmd = [sys.executable, "-m", "ppls_slm.apps.association_analysis", "--output_dir", str(assoc_out), "--plot"]
            if args.gene_expr and args.protein_expr:
                cmd += ["--gene_expr", args.gene_expr, "--protein_expr", args.protein_expr]
            elif brca_data:
                cmd += ["--brca_data", str(brca_data)]

            code = tee_run(cmd, cwd=root, log_path=logs_dir / "03_association.log", env=run_env)
            if code != 0:
                return code

        # 4) Prediction (synthetic)
        if run_prediction:
            cmd = [sys.executable, "-u", "-m", "ppls_slm.apps.prediction", "--config", str(config_path)]
            code = tee_run(cmd, cwd=root, log_path=logs_dir / "04_prediction.log", env=run_env)
            if code != 0:
                return code

        # 5) BRCA prediction benchmark (optional)
        if run_brca_prediction:
            cmd = [sys.executable, "-u", "-m", "ppls_slm.apps.brca_prediction", "--config", str(config_path)]
            code = tee_run(cmd, cwd=root, log_path=logs_dir / "05_brca_prediction.log", env=run_env)
            if code != 0:
                return code

        # 6) BRCA calibration benchmark (optional)
        if run_brca_calibration:
            cmd = [sys.executable, "-u", "-m", "ppls_slm.apps.brca_calibration", "--config", str(config_path)]
            code = tee_run(cmd, cwd=root, log_path=logs_dir / "06_brca_calibration.log", env=run_env)
            if code != 0:
                return code

        # 7) CITE-seq prediction benchmark (optional)
        if run_citeseq_prediction:
            cmd = [sys.executable, "-u", "-m", "ppls_slm.apps.citeseq_prediction", "--config", str(config_path)]
            code = tee_run(cmd, cwd=root, log_path=logs_dir / "07_citeseq_prediction.log", env=run_env)
            if code != 0:
                return code

        # 8) PCCA simulation (Table 1 extension)
        if run_pcca_simulation:
            code = tee_run(
                [sys.executable, "-u", "-m", "ppls_slm.apps.pcca_simulation", "--config", str(config_path)],
                cwd=root,
                log_path=logs_dir / "08_pcca_simulation.log",
                env=run_env,
            )
            if code != 0:
                return code

        # 9) PPCA verification (Appendix)
        if run_ppca_verification:
            code = tee_run(
                [sys.executable, "-u", "-m", "ppls_slm.apps.ppca_verification", "--config", str(config_path)],
                cwd=root,
                log_path=logs_dir / "09_ppca_verification.log",
                env=run_env,
            )
            if code != 0:
                return code

        # 10) Sync artifacts
        if run_sync:
            code = tee_run([sys.executable, "-u", "scripts/sync_artifacts.py"], cwd=root, log_path=logs_dir / "10_sync_artifacts.log", env=run_env)
            if code != 0:
                return code





        elapsed = time.time() - start
        print("\n" + "=" * 80)
        print(f"ALL DONE in {elapsed/60:.1f} minutes")
        print(f"Logs: {logs_dir}")
        print(f"Artifacts synced into: {root / 'paper' / 'artifacts'}")

        print("Next: tell me this timestamp folder so I can inspect outputs/logs.")
        print("=" * 80)

        return 0

    finally:
        # Restore original force flags (keep other config changes intact)
        if original_config is not None and config_path.exists():
            try:
                current = load_json(config_path)
                restored = dict(current)
                # Restore only the 3 force flags to their previous values
                out = dict(restored.get("output", {}))
                prev_out = dict(original_config.get("output", {}))
                for k in ("force_data_generation", "force_parameter_estimation", "force_visualization"):
                    if k in prev_out:
                        out[k] = prev_out[k]
                restored["output"] = out
                save_json(config_path, restored)
                print("\n[OK] config force flags restored.")
            except Exception as e:
                print(f"\n[WARN] Failed to restore config.json force flags: {e}")


if __name__ == "__main__":
    raise SystemExit(main())
