"""No model, network, server, CUDA, or data access is needed for these tests."""
import ast
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("larp_release_cli", ROOT / "cli.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


class ArchiveTests(unittest.TestCase):
    def test_all_final_scientific_hashes(self):
        result = cli.check_sources()
        self.assertEqual(result["status"], "passed", result)
        self.assertEqual(result["frozen_scientific_sources"], 341)

    def test_full_matrix_preserves_fixed_scientific_protocol(self):
        protocol = cli.read_json("protocol/protocol.json")
        variants = cli.read_json("protocol/variants.json")
        jobs = set()
        for dataset in protocol["datasets"]:
            for seed in protocol["seeds"]:
                for variant, variant_spec in variants.items():
                    cfg = cli.resolve_config(dataset, variant, seed, "DATA", "CLIP", "TEACHER", "OUTPUT")
                    p = cfg["TRAINER"]["PROMPTKD"]
                    a = p["ADAPTATION"]
                    self.assertEqual(cfg["SEED"], seed)
                    self.assertEqual(cfg["OPTIM"]["MAX_EPOCH"], 20)
                    self.assertEqual(cfg["OPTIM"]["NAME"], "sgd")
                    self.assertEqual(cfg["DATALOADER"]["TRAIN_X"]["BATCH_SIZE"], 8)
                    self.assertEqual(p["PREC"], "fp16")
                    self.assertEqual(p["CE_WEIGHT"], 0.0)
                    self.assertEqual(p["KD_WEIGHT"], 200 if dataset in ("dtd", "fgvc_aircraft", "oxford_flowers") else 1000)
                    self.assertEqual(a["REP_WEIGHT"], 0.0)
                    if variant == "r1":
                        self.assertEqual(a["TYPE"], "none")
                        self.assertEqual(a["LR_MULT"], 1.0)
                    else:
                        self.assertEqual(a["TYPE"], "larp")
                        self.assertEqual(a["LORA_TARGETS"], ["o"])
                        self.assertEqual(a["LAYERS"], variant_spec["layers"])
                        self.assertEqual(a["LORA_R"], variant_spec["rank"])
                        self.assertEqual(a["LR_MULT"], .1)
                        self.assertEqual(a["LORA_ALPHA"], 1.0)
                        self.assertEqual(a["LORA_DROPOUT"], 0.0)
                    self.assertNotIn("/root/", json.dumps(cfg))
                    jobs.add((dataset, seed, variant))
        self.assertEqual(len(jobs), 150)
        self.assertTrue(protocol["backend"]["efficient"])
        self.assertTrue(protocol["portable_training_enabled"])

    def test_custom_paths_are_preserved_without_shell_interpretation(self):
        cfg = cli.resolve_config("dtd", "o4-r2", 3, "data with spaces", "clip $dollar", "teacher [x]", "runs odd")
        self.assertEqual(cfg["DATASET"]["ROOT"], "data with spaces")
        self.assertEqual(cfg["TRAINER"]["PROMPTKD"]["WEIGHTS_ROOT"], "clip $dollar")
        self.assertEqual(cfg["TRAINER"]["PROMPTKD"]["TEACHER_ROOT"], "teacher [x]")
        self.assertEqual(cfg["OUTPUT_DIR"], "runs odd/dtd/seed_3/o4-r2")
        self.assertFalse(Path(cfg["OUTPUT_DIR"]).exists())

    def test_full_templates_change_only_external_paths(self):
        contract = json.loads((ROOT / "archive/source_contract.json.txt").read_text())
        templates = cli.read_json("protocol/resolved_templates.json")
        path_fields = cli.read_json("protocol/protocol.json")["path_substitutions"]
        for dataset, roles in templates.items():
            for role, actual in roles.items():
                expected = copy.deepcopy(contract["configs"][dataset][role])
                for dotted, replacement in path_fields.items():
                    keys = dotted.split(".")
                    parent = expected
                    for key in keys[:-1]:
                        parent = parent[key]
                    parent[keys[-1]] = replacement
                self.assertEqual(actual, expected, (dataset, role))

    def test_plan_counts_and_no_files(self):
        args = cli.make_parser().parse_args(["plan", "--all-variants", "--data-root", "d", "--clip-root", "c",
                                            "--teacher-root", "t", "--output-root", "NO_CREATED_OUTPUT"])
        result = cli.plan(args)
        self.assertEqual(result["job_count"], 150)
        self.assertEqual(result["epoch_count"], 3000)
        self.assertFalse(Path("NO_CREATED_OUTPUT").exists())

    def test_train_requires_explicit_roots(self):
        code = cli.main(["train"])
        self.assertEqual(code, 2)

    def test_original_install_function_ast_is_unchanged(self):
        inventory = cli.read_json("provenance/source_inventory.json")
        source = (ROOT / "protocol/original_sgd.py").read_text()
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "install")
        digest = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
        self.assertEqual(digest, inventory["extracted_function"]["ast_sha256"])

    def test_locks_contain_no_machine_paths_or_local_wheel_urls(self):
        lock = (ROOT / "environment/requirements.lock.txt").read_text()
        for marker in ("/root/", "/Users/", "file://", "-e "):
            self.assertNotIn(marker, lock)
        self.assertIn("torch==2.0.1+cu118", lock)

    def test_unknown_protocol_members_rejected(self):
        for dataset, variant, seed in (("imagenet", "o4-r2", 1), ("dtd", "v4", 1), ("dtd", "o4-r2", 4)):
            with self.assertRaises(ValueError):
                cli.resolve_config(dataset, variant, seed, "d", "c", "t", "o")


if __name__ == "__main__":
    unittest.main()
