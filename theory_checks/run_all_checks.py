import importlib
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


CHECKS = [
    "check_lemma_rankn",
    "check_pcca_schur_pd",
    "check_scalar_expansion",
    "check_bcd_propositions",
    "check_noise_theorem",
]


def run_all() -> dict:
    summary = {
        "all_passed": True,
        "results": [],
    }

    for module_name in CHECKS:
        try:
            module = importlib.import_module(module_name)
            result = module.run()
            summary["results"].append({"module": module_name, **result})
        except Exception as exc:
            summary["all_passed"] = False
            summary["results"].append(
                {
                    "module": module_name,
                    "status": "FAIL",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )

    out_path = Path(__file__).resolve().parent / "results.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    result = run_all()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["all_passed"] else 1)
