"""Config loading with YAML + CLI overrides.

Usage:
    cfg = get_config()                          # from sys.argv
    cfg = get_config(["--config", "x.yaml"])    # explicit
    cfg = get_config(["--set", "stage1.batch_size=128"])  # override
"""

import argparse
import copy
import os
import yaml
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge overrides into base (in-place)."""
    for k, v in overrides.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def _set_nested(d: dict, dotpath: str, value: str):
    """Set a value in a nested dict using dot-path notation.

    Attempts to cast value to int, then float, then bool, else keeps as str.
    """
    keys = dotpath.split(".")
    for key in keys[:-1]:
        d = d.setdefault(key, {})

    # Auto-cast
    for caster in (int, float):
        try:
            value = caster(value)
            break
        except (ValueError, TypeError):
            continue
    else:
        if value.lower() in ("true", "false"):
            value = value.lower() == "true"

    d[keys[-1]] = value


def _resolve_paths(cfg: dict, root: Path):
    """Resolve relative paths in data and outputs sections against project root."""
    for section in ("data", "outputs"):
        if section not in cfg:
            continue
        for k, v in cfg[section].items():
            if isinstance(v, str) and ("/" in v or v.endswith((".csv", ".h5"))):
                p = Path(v)
                if not p.is_absolute():
                    cfg[section][k] = str(root / p)


def get_config(argv=None) -> dict:
    """Load config from YAML, apply CLI overrides, resolve paths."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args, _ = parser.parse_known_args(argv)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Apply --set dot.path=value overrides
    for override in args.overrides:
        if "=" not in override:
            raise ValueError(f"Override must be key=value, got: {override}")
        dotpath, value = override.split("=", 1)
        _set_nested(cfg, dotpath, value)

    # Also check OVERRIDE env var (comma-separated, for Slurm convenience)
    env_overrides = os.environ.get("OVERRIDE", "")
    if env_overrides:
        for item in env_overrides.split(","):
            item = item.strip()
            if not item:
                continue
            if "=" not in item:
                raise ValueError(f"OVERRIDE env var entry must be key=value, got: {item}")
            dotpath, value = item.split("=", 1)
            _set_nested(cfg, dotpath, value)

    _resolve_paths(cfg, PROJECT_ROOT)
    return cfg


def get_nested(cfg: dict, dotpath: str, default=None):
    """Safely get a nested value by dot-path."""
    keys = dotpath.split(".")
    d = cfg
    for key in keys:
        if not isinstance(d, dict) or key not in d:
            return default
        d = d[key]
    return d
