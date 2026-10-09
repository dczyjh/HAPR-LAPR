"""Offline documentation QA; commands are parsed, never executed as training."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import re
import shlex
import subprocess
import sys
import unittest
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
DOCS = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md")),
        *sorted((ROOT / "routes").glob("*/README.md"))]


class DocumentationTests(unittest.TestCase):
    def test_local_links_and_source_line_anchors_exist(self):
        for document in DOCS:
            text = document.read_text()
            for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
                parts = urlsplit(target)
                if parts.scheme or not parts.path:
                    continue
                path = (document.parent / unquote(parts.path)).resolve()
                self.assertTrue(path.is_relative_to(ROOT), (document.name, target))
                self.assertTrue(path.exists(), (document.name, target))
                if re.fullmatch(r"L\d+", parts.fragment):
                    self.assertTrue(path.is_file())
                    self.assertLessEqual(int(parts.fragment[1:]), len(path.read_text().splitlines()), target)

    def test_python_example_syntax(self):
        count = 0
        for document in DOCS:
            for code in re.findall(r"```python\s*\n(.*?)```", document.read_text(), re.S):
                ast.parse(code, filename=document.name)
                count += 1
        self.assertGreaterEqual(count, 2)

    def test_shell_examples_are_syntax_only(self):
        count = 0
        for document in DOCS:
            for code in re.findall(r"```bash\s*\n(.*?)```", document.read_text(), re.S):
                result = subprocess.run(["bash", "-n"], input=code, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, (document.name, result.stderr))
                count += 1
        self.assertGreaterEqual(count, 10)

    def test_reproduction_examples_use_real_arguments(self):
        sys.path.insert(0, str(ROOT))
        try:
            spec = importlib.util.spec_from_file_location("documentation_reproduce", ROOT / "reproduce.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            count = 0
            for document in DOCS:
                for code in re.findall(r"```bash\s*\n(.*?)```", document.read_text(), re.S):
                    for line in code.replace("\\\n", " ").splitlines():
                        if not line.strip() or line.lstrip().startswith("#"):
                            continue
                        words = shlex.split(line, comments=True)
                        if "reproduce.py" not in words or words[-1:] == ["--help"]:
                            continue
                        args = module.parser().parse_args(words[words.index("reproduce.py") + 1:])
                        if args.command in ("plan", "run"):
                            plan = module.execution_plan(args.route, args.dataset, args.seed, args.variant, args.all_variants)
                            self.assertEqual(plan["epochs_each"], 20)
                        count += 1
            self.assertGreaterEqual(count, 8)
        finally:
            sys.path.remove(str(ROOT))


if __name__ == "__main__":
    unittest.main()
