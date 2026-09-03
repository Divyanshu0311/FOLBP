"""Benchmark harness shared by every planner arm (FOLBP variants + PEFA).

Import from a planner `main.py` by putting the parent of this package on the path:

    import sys; from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from bench.llm_meter import METER
    from bench.record import write_task_record
"""
