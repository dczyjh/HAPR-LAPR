#!/usr/bin/env python3
"""Read-only integrity/security checks; does not import torch or access network."""
from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "metadata" / "release_manifest.json"
EXEMPT = {"metadata/release_manifest.json", "metadata/validation_report.json"}
GENERATED = {".git", "__pycache__", ".pytest_cache", ".venv", ".mypy_cache"}
SECRET_PATTERNS = {
    "private-key": re.compile(r"-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----"),
    "github-token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{35,})\b"),
    "aws-access-key": re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    "ssh-target": re.compile(r"root@connect\.[A-Za-z0-9.-]+|connect\.cqa1\.seetacloud\.com"),
    "inline-password": re.compile(r"(?i)(?:password|passwd|ssh_pass|api_key|access_token)\s*[:=]\s*['\"][^'\"\s]{6,}['\"]"),
    "private-mac-path": re.compile(r"/" + r"Users/[^/\s]+/"),
}


def files() -> list[Path]:
    return sorted(p for p in ROOT.rglob("*") if p.is_file()
                  and not set(p.relative_to(ROOT).parts).intersection(GENERATED)
                  and p.suffix not in {".pyc", ".pyo"})


def main() -> int:
    errors: list[str] = []
    if not MANIFEST.is_file():
        print(json.dumps({"status": "failed", "errors": ["Missing release manifest"]}))
        return 1
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected = manifest["files"]
    present: set[str] = set()
    counts = {"files": 0, "python_ast": 0, "json": 0}
    for path in files():
        rel = path.relative_to(ROOT).as_posix()
        present.add(rel)
        data = path.read_bytes()
        counts["files"] += 1
        if path.is_symlink():
            errors.append(f"Symlink not allowed: {rel}")
        if len(data) > 10 * 1024 * 1024:
            errors.append(f"File exceeds 10 MiB publication budget: {rel}")
        if path.suffix in {".pt", ".pth", ".whl", ".safetensors", ".zip"} or rel.endswith(".pth.tar"):
            errors.append(f"External/binary asset not allowed: {rel}")
        if path.name == ".env" or path.name.startswith(".env.") or path.suffix in {".pem", ".key"} or ".ssh" in path.parts:
            errors.append(f"Credential/local-settings filename not allowed: {rel}")
        if rel not in EXEMPT:
            entry = expected.get(rel)
            if entry is None:
                errors.append(f"Unmanifested file: {rel}")
            elif entry != {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}:
                errors.append(f"Digest/size mismatch: {rel}")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        for name, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                errors.append(f"Sensitive-pattern {name}: {rel}")
        if path.suffix == ".py":
            try:
                ast.parse(text, filename=rel)
                counts["python_ast"] += 1
            except SyntaxError as exc:
                errors.append(f"Syntax error {rel}:{exc.lineno}: {exc.msg}")
        if path.suffix == ".json":
            try:
                json.loads(text)
                counts["json"] += 1
            except (ValueError, TypeError) as exc:
                errors.append(f"Invalid JSON {rel}: {exc}")
    for rel in set(expected).difference(present):
        errors.append(f"Missing file: {rel}")
    result = {"status": "passed" if not errors else "failed", "checks": counts,
              "errors": errors, "scope": "offline files only; no GPU or accuracy validation"}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
