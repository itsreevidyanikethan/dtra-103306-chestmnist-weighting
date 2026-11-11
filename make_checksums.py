"""Create or verify portable SHA-256 manifests for source files or run artifacts."""
import argparse
import hashlib
from pathlib import Path
import re

SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", ".pytest_cache"}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_path(name, base):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe manifest path: {name}")
    resolved = (base / path).resolve()
    if not resolved.is_relative_to(base):
        raise ValueError(f"path escapes working directory: {name}")
    return base / path


def generate(root="outputs", output="CHECKSUMS.txt"):
    base = Path.cwd().resolve()
    root = safe_path(root, base)
    destination = safe_path(output, base)
    if not root.is_dir():
        raise ValueError(f"artifact directory does not exist: {root.name}")
    lines = []
    for path in sorted(root.rglob("*")):
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        if path.resolve() == destination.resolve() or not path.is_file():
            continue
        name = path.relative_to(base).as_posix()
        safe_path(name, base)
        if "\n" in name or "\r" in name:
            raise ValueError("newline in artifact filename")
        lines.append(f"{digest(path)}  {name}")
    if not lines:
        raise ValueError("no artifacts found; refusing to create an empty manifest")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def verify(manifest):
    base = Path.cwd().resolve()
    manifest = safe_path(manifest, base)
    lines = manifest.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError("empty checksum manifest")
    seen = set()
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match:
            raise ValueError("malformed checksum entry")
        expected, name = match.groups()
        path = safe_path(name, base)
        resolved = path.resolve()
        if resolved in seen:
            raise ValueError(f"duplicate checksum entry: {name}")
        seen.add(resolved)
        if not path.is_file():
            raise ValueError(f"missing artifact: {name}")
        if digest(path) != expected:
            raise ValueError(f"SHA-256 mismatch: {name}")
    return len(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default="outputs")
    parser.add_argument("--output", default="CHECKSUMS.txt")
    parser.add_argument("--verify", metavar="MANIFEST")
    args = parser.parse_args()
    try:
        if args.verify:
            print(f"Verified {verify(args.verify)} files in {args.verify}")
        else:
            print(f"Wrote {generate(args.root, args.output)} checksums to {args.output}")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Integrity check failed: {exc}\n")


if __name__ == "__main__":
    main()
