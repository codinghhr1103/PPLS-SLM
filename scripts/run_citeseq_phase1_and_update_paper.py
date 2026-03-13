"""(Deprecated) Previously: one-click run for CITE-seq + paper update.

Use explicit steps instead:
  1) python scripts/run_citeseq_phase1.py
  2) python scripts/update_paper_artifacts.py
  3) python scripts/generate_paper_tex.py
  4) python scripts/compile_paper_pdflatex.py  (optional)
"""

from __future__ import annotations


def main() -> int:
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
