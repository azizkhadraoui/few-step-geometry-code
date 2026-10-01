"""Optional Weights & Biases logging.

Measurement scripts do `import wandb_opt as wandb`. Logging to W&B is a convenience and
never part of a measurement: every number a script reports is printed to stdout and
written to its JSON file regardless. If USE_WANDB is not "1", or the wandb package is
absent, or init fails, the calls below become no-ops and the run proceeds unchanged.
"""
import os

_wb = None
if os.environ.get("USE_WANDB", "0") == "1":
    try:
        import wandb as _wb
    except Exception as e:                      # not installed
        print(f"[wandb] unavailable ({e}); continuing without it")
        _wb = None
else:
    print("[wandb] logging off (set USE_WANDB=1 to enable)")


def init(*a, **k):
    global _wb
    if _wb is None:
        return None
    try:
        return _wb.init(*a, **k)
    except Exception as e:                      # not logged in, offline, quota, ...
        print(f"[wandb] init failed ({e}); continuing without it")
        _wb = None
        return None


def log(*a, **k):
    if _wb is not None:
        try:
            _wb.log(*a, **k)
        except Exception:
            pass


def finish(*a, **k):
    if _wb is not None:
        try:
            _wb.finish(*a, **k)
        except Exception:
            pass


def Image(*a, **k):
    return _wb.Image(*a, **k) if _wb is not None else None


def Table(*a, **k):
    return _wb.Table(*a, **k) if _wb is not None else None
