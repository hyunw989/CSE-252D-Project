"""
Minimal project env-file loader.

Loads KEY=VALUE pairs from .env without adding a runtime dependency.
Project .env values win over existing environment variables by default.
"""

from __future__ import annotations

import os
from pathlib import Path


DEFAULT_ENV_FILES = (".env",)


def load_project_env(
    root: str | Path | None = None,
    filenames: tuple[str, ...] = DEFAULT_ENV_FILES,
    override: bool = True,
) -> list[Path]:
    """Load env vars from project-local env files and return files read."""
    roots = []
    if root is not None:
        roots.append(Path(root))
    roots.extend([Path(__file__).resolve().parent, Path.cwd()])

    loaded: list[Path] = []
    seen: set[Path] = set()
    for base in roots:
        for name in filenames:
            path = (base / name).resolve()
            if path in seen:
                continue
            seen.add(path)
            if path.exists():
                _load_env_file(path, override=override)
                loaded.append(path)
    return loaded


def get_openai_api_key(explicit_key: str = "") -> str:
    """Return explicit key first, then .env/shell OPENAI_API_KEY."""
    load_project_env()
    return explicit_key or os.environ.get("OPENAI_API_KEY", "")


def _load_env_file(path: Path, override: bool) -> None:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = _strip_inline_comment(value.strip())
        value = _strip_quotes(value)
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value


def _strip_inline_comment(value: str) -> str:
    quote: str | None = None
    for index, char in enumerate(value):
        if char in {"'", '"'}:
            if quote == char:
                quote = None
            elif quote is None:
                quote = char
        elif char == "#" and quote is None and (index == 0 or value[index - 1].isspace()):
            return value[:index].rstrip()
    return value


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
