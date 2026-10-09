"""Fresh-process, paired execution of the frozen PromptKD route trainers.

This module records inputs and checkpoint evidence. The model, loss, optimizer,
schedule, epoch loop and selection rule remain the original route implementation.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import traceback
import warnings

ROOT = Path(__file__).resolve().parents[1]
TOLERANCES = {"forward": {"atol": .002, "rtol": .002},
              "gradient": {"atol": .0001, "rtol": .01},
              "update": {"atol": .00001, "rtol": .001}}
SPECS = {"caltech101": (4128,1549,916), "stanford_cars": (6509,4002,4039),
         "sun397": (15880,9950,9900), "oxford_pets": (2944,1881,1788),
         "eurosat": (13500,4200,3900), "ucf101": (7639,1934,1849),
         "food101": (50500,15300,15000), "fgvc_aircraft": (3334,1666,1667),
         "oxford_flowers": (4093,1053,1410), "dtd": (2820,864,828)}
VARIANTS = {"harp": ["r1", "harp", "mid4_d32", "last4_d16", "last4_d64"],
            "larp": ["r1", "o4-r2", "o4-r1", "o4-r4", "o3-r2"], "r0": ["r0"]}
ANCHOR = {"harp": "harp", "larp": "o4-r2", "r0": "r0"}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_adapter(route):
    path = ROOT / "routes" / route / "portable.py"
    spec = importlib.util.spec_from_file_location("portable_route_" + route, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def pair_dir(reg, dataset, seed):
    return Path(reg["output_root"]) / reg["route"] / dataset / f"seed_{seed}"


def config_for(adapter, reg, dataset, seed, variant, output):
    paths = {k: reg[k] for k in ("data_root", "clip_root", "teacher_root")}
    paths.update(output=str(output), output_dir=str(output), output_root=str(output))
    return adapter.resolve_config(dataset, seed, variant, paths)


def initial_path(reg, dataset, seed, variant):
    return pair_dir(reg, dataset, seed) / "initial" / variant / "state.pt"


def verify_registration(reg):
    assert sha(ROOT / "metadata/release_manifest.json") == reg["release_manifest_sha256"], "Release version changed during run"
    manifest = read(ROOT / "metadata/release_manifest.json")
    for relative, entry in manifest["files"].items():
        path = ROOT / relative
        assert path.is_file() and sha(path) == entry["sha256"], ("Release source changed", relative)


def hash_model(adapter, model, frozen=False):
    return {adapter.canonical(name): adapter.tensor_hash(p) for name, p in model.named_parameters()
            if not adapter.added(name) and (not frozen or not p.requires_grad)}


def identity(adapter, batch, data_root):
    paths = batch.get("impath")
    if paths is not None:
        root = Path(os.path.abspath(data_root))
        # Image-directory symlinks are part of the published material layout.
        # Use logical membership paths, not resolved paths outside DATASET.ROOT.
        paths = [Path(os.path.abspath(p)).relative_to(root).as_posix() for p in paths]
    return {"image": adapter.tensor_hash(batch["img"]), "label": adapter.tensor_hash(batch["label"]), "impath": paths}


def rng_hash(adapter, torch):
    return {"cpu": adapter.tensor_hash(torch.get_rng_state()),
            "cuda": adapter.tensor_hash(torch.cuda.get_rng_state()) if torch.cuda.is_available() else None}


def assert_initial(adapter, tr, reference):
    import torch
    current = adapter.capture_initial(tr)
    for key in ("common", "rng"):
        assert current[key] == reference[key], ("Initialization mismatch", key)
    assert current["added"].keys() == reference["added"].keys(), "Added parameter names changed"
    for name, value in current["added"].items():
        assert torch.equal(value, reference["added"][name]), ("Added initialization mismatch", name)


def nominal_rates(tr):
    """Inspect and restore the original scheduler, without consuming RNG/updates."""
    # Warmup state_dict includes the successor object. Deep-copying that and
    # loading it back disconnects its optimizer from the live trainer.
    chain, scheduler = [], tr.sched
    while scheduler is not None:
        values = {k: copy.deepcopy(v) for k, v in scheduler.state_dict().items()
                  if k not in ("optimizer", "successor")}
        chain.append((scheduler, values))
        scheduler = getattr(scheduler, "successor", None)
    learning_rates = [g["lr"] for g in tr.optim.param_groups]
    result = []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for _ in range(20):
                result.append([g["lr"] for g in tr.optim.param_groups])
                tr.sched.step()
    finally:
        for scheduler, values in chain:
            scheduler.__dict__.update(values)
        for group, rate in zip(tr.optim.param_groups, learning_rates):
            group["lr"] = rate
    return result


def state_exact(tr, checkpoint, torch):
    actual = tr.model.state_dict()
    expected = checkpoint["state_dict"]
    assert actual.keys() == expected.keys(), "Checkpoint state names differ"
    assert all(torch.equal(v.detach().cpu(), expected[n]) for n, v in actual.items()), "Checkpoint parameters/buffers differ"


def first_step_compare(a, b, label, torch):
    assert a.shape == b.shape and a.dtype == b.dtype, label
    assert torch.isfinite(a).all() and torch.isfinite(b).all(), label
    delta = (a.float() - b.float()).abs()
    if label == "student_logits":
        assert a.ndim == 2
        scale = torch.maximum(a.float().abs().amax(-1, keepdim=True), b.float().abs().amax(-1, keepdim=True))
        bound = .002 + .002 * scale
    else:
        bound = .002 + .002 * b.float().abs()
    bad = delta > bound
    result = {"label": label, "max_abs": float(delta.max()), "failed_elements": int(bad.sum()), "passed": not bool(bad.any())}
    assert result["passed"], ("First step differs from paired R1", result)
    return result


def gate(adapter, tr, reg, dataset, seed, variant, out):
    import torch
    initial = {n: p.detach().cpu().clone() for n, p in tr.model.named_parameters() if p.requires_grad}
    frozen = hash_model(adapter, tr.model, True)
    teacher = {n: adapter.tensor_hash(v) for n, v in tr.model_teacher.state_dict().items()}
    count = len(tr.train_loader_x)
    assert count > 1
    rates = nominal_rates(tr)
    tr.num_batches = count
    batch = next(iter(tr.train_loader_x))
    input_identity = identity(adapter, batch, reg["data_root"])
    base_path = pair_dir(reg, dataset, seed) / "gate" / "r1"
    if reg["route"] != "r0" and variant != "r1":
        assert input_identity == read(base_path / "audit.json")["input"]
    tr.set_model_mode("train")
    first, records = {}, []
    capturing = [True]
    def teacher_hook(module, inputs, output):
        if capturing[0]:
            first.update(teacher_text=output[1].detach().cpu().clone(), teacher_logits=output[2].detach().cpu().clone())
    def student_hook(module, inputs, output):
        if capturing[0]:
            with torch.no_grad():
                first["student_logits"] = (output[1] * output[0] @ first["teacher_text"].to(output[0].device).t()).detach().cpu().clone()
    hooks = [tr.model_teacher.register_forward_hook(teacher_hook), tr.model.register_forward_hook(student_hook)]
    try:
        for epoch, batch_index in ((0,0), (1,0), (1,count-1), (2,0)):
            tr.epoch, tr.batch_idx = epoch, batch_index
            for group, rate in zip(tr.optim.param_groups, rates[epoch]):
                group["lr"] = rate
            tr.failure_batch = batch
            result = tr.forward_backward(batch)
            assert all(math.isfinite(float(v)) for v in result.values()), "Nonfinite gate loss"
            record = dict(getattr(tr, "actual_step", {}), loss=result)
            records.append(record)
            if capturing[0]:
                first["loss"] = torch.tensor(result["loss"])
                capturing[0] = False
    finally:
        for hook in hooks:
            hook.remove()
    torch.save(first, out / "first_step.pt")
    comparisons = []
    if reg["route"] != "r0" and variant != "r1":
        reference = torch.load(base_path / "first_step.pt", map_location="cpu")
        comparisons = [first_step_compare(first[k], reference[k], k, torch) for k in first]
    assert hash_model(adapter, tr.model, True) == frozen, "Frozen student parameters changed"
    assert teacher == {n: adapter.tensor_hash(v) for n, v in tr.model_teacher.state_dict().items()}, "Teacher changed"
    changes = {n: int(torch.count_nonzero(p.detach().cpu() - initial[n])) for n, p in tr.model.named_parameters() if p.requires_grad}
    for prefix in ("image_encoder.", "VPT_image_trans."):
        assert any(v for n, v in changes.items() if n.startswith(prefix) and not adapter.added(n)), prefix
    for name, p in tr.model.named_parameters():
        if not adapter.added(name) or not p.requires_grad:
            continue
        momentum = tr.optim.state[p].get("momentum_buffer")
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert momentum is not None and torch.isfinite(momentum).all(), name
        assert torch.count_nonzero(p.grad) and torch.count_nonzero(momentum), name
        if "lora_B_o" in name:
            assert changes[name] > 0, name
    if reg["route"] == "larp" and variant != "r1":
        for block in tr.model.image_encoder.transformer.resblocks:
            op = block.attn.out_proj
            if hasattr(op, "parametrizations"):
                module = op.parametrizations.weight[0]
                norm = float((module.lora_B_o.float() @ module.lora_A_o.float()).norm())
                assert norm > 0 and math.isfinite(norm)
    before = {n: adapter.tensor_hash(v) for n, v in tr.model.state_dict().items()}
    tr.save_model(2, str(out), val_result=0., model_name="model-best.pth.tar")
    tr.load_model(str(out))
    assert before == {n: adapter.tensor_hash(v) for n, v in tr.model.state_dict().items()}, "Gate save/load differs"
    tr.close_writer()
    dump(out / "audit.json", {"status": "passed", "time": now(), "dataset": dataset, "seed": seed,
          "variant": variant, "steps": records, "input": input_identity, "changes": changes,
          "first_step_comparison": comparisons, "first_step_sha256": sha(out / "first_step.pt"),
          "roundtrip_exact": True, "nominal_lr_20": rates})


def train(adapter, tr, reg, dataset, seed, variant, out):
    import torch
    pair = pair_dir(reg, dataset, seed)
    gate_record = read(pair / "gate" / variant / "audit.json")
    assert gate_record["status"] == "passed"
    frozen = hash_model(adapter, tr.model, True)
    teacher = {n: adapter.tensor_hash(v) for n, v in tr.model_teacher.state_dict().items()}
    torch.save({n: p.detach().cpu().clone() for n,p in tr.model.named_parameters() if p.requires_grad}, out / "initial_trainable.pt")
    dump(out / "initial_shared.json", hash_model(adapter, tr.model))
    dump(out / "frozen_before.json", frozen)
    count = len(tr.train_loader_x)
    sums, counts, last_base = [0.] * 20, [0] * 20, [None]
    paired_stream = None
    if reg["route"] != "r0" and variant != "r1":
        control = pair / "formal" / "r1"
        assert read(control / "audited_metrics.json")["status"] == "passed", "R1 must finish before its candidates"
        assert hash_model(adapter, tr.model) == read(control / "initial_shared.json")
        paired_stream = (control / "batches.jsonl").open(encoding="utf-8")
    original_fb = tr.forward_backward
    def audited_fb(batch):
        tr.failure_batch = batch
        current = identity(adapter, batch, reg["data_root"])
        e, b = tr.epoch, tr.batch_idx
        if e == 0 and b == 0:
            assert current == gate_record["input"], "Training starts on a different batch from its gate"
        reference = None
        if paired_stream is not None:
            line = paired_stream.readline()
            assert line, "R1 trace ended too early"
            reference = json.loads(line)
            assert current == reference["identity"], ("Paired training input mismatch", e + 1, b + 1)
        result = original_fb(batch)
        assert all(math.isfinite(float(v)) for v in result.values()), "Nonfinite training loss"
        actual = getattr(tr, "actual_step", None)
        assert actual is not None, "Route optimizer did not record actual learning rates"
        row = dict(actual, identity=current, loss=float(result["loss"]), loss_kd=float(result.get("loss_kd", result["loss"])))
        if reference:
            for key in ("prompt_lr", "projector_lr"):
                assert row[key] == reference[key], ("Paired public LR mismatch", key, e + 1, b + 1)
        with (out / "batches.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        counts[e] += 1
        sums[e] += row["loss_kd"]
        return result
    tr.forward_backward = audited_fb
    evaluation = adapter.evidence_module()
    evidence = evaluation.attach(tr, torch, adapter.tensor_hash)
    old_save = tr.save_model
    def saved(*args, **kwargs):
        result = old_save(*args, **kwargs)
        if kwargs.get("model_name") == "model-best.pth.tar":
            assert evidence["last"] is not None
            torch.save(dict(evidence["last"], checkpoint_sha256=sha(out / "VLPromptLearner/model-best.pth.tar")), out / "best_val_evidence.pt")
        return result
    tr.save_model = saved
    old_test = tr.test
    def test(split=None):
        value = old_test(split)
        if split == "val":
            last_base[0] = float(value)
        return value
    tr.test = test
    old_epoch = tr.after_epoch
    def after_epoch():
        old_epoch()
        e = tr.epoch
        assert counts[e] == count and last_base[0] is not None
        with (out / "epochs.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"epoch": e + 1, "mean_kd": sums[e] / count, "base": last_base[0]}, allow_nan=False) + "\n")
        assert shutil.disk_usage(out).free >= reg.get("disk_reserve_bytes", 20 * 1024**3), "Disk reserve exhausted"
    tr.after_epoch = after_epoch
    old_after = tr.after_train
    def after_train():
        old = tr.test
        def native(split=None):
            checkpoint = torch.load(out / "VLPromptLearner/model-best.pth.tar", map_location="cpu")
            state_exact(tr, checkpoint, torch)
            value = old(split)
            torch.save(dict(evidence["last"], checkpoint_sha256=sha(out / "VLPromptLearner/model-best.pth.tar")), out / f"native_{split}_evidence.pt")
            return value
        tr.test = native
        if reg["route"] == "r0":
            tr.load_model(str(out))
            tr.test("val")
            tr.test("test")
            tr.close_writer()
        else:
            old_after()
    tr.after_train = after_train
    try:
        tr.train()
        if paired_stream:
            assert not paired_stream.readline(), "Unused R1 training records remain"
        assert counts == [count] * 20, ("Incomplete 20 epoch training", counts)
        assert hash_model(adapter, tr.model, True) == frozen, "Frozen student changed during training"
        assert teacher == {n: adapter.tensor_hash(v) for n,v in tr.model_teacher.state_dict().items()}, "Teacher changed during training"
        dump(out / "train_complete.json", {"status": "passed", "time": now(), "epochs": 20,
             "batches": sum(counts), "teacher_unchanged": True, "frozen_student_unchanged": True,
             "best_sha256": sha(out / "VLPromptLearner/model-best.pth.tar"),
             "last_sha256": sha(out / "VLPromptLearner/model.pth.tar-20")})
    finally:
        if paired_stream:
            paired_stream.close()


def audit_checkpoint(adapter, tr, checkpoint, initial, frozen, route, variant, tag, torch):
    epoch = checkpoint["epoch"]
    assert 1 <= epoch <= 20 and (tag != "last" or epoch == 20)
    assert all(not torch.is_floating_point(x) or torch.isfinite(x).all() for x in checkpoint["state_dict"].values())
    for name,p in tr.model.named_parameters():
        if not p.requires_grad and not adapter.added(name):
            assert adapter.tensor_hash(checkpoint["state_dict"][name]) == frozen[adapter.canonical(name)], name
    names = {id(p): n for n,p in tr.model.named_parameters()}
    assert len(checkpoint["optimizer"]["param_groups"]) == len(tr.optim.param_groups)
    rows = []
    for saved, live in zip(checkpoint["optimizer"]["param_groups"], tr.optim.param_groups):
        assert len(saved["params"]) == len(live["params"])
        for key, p in zip(saved["params"], live["params"]):
            if not p.requires_grad:
                continue
            name = names[id(p)]
            changed = int(torch.count_nonzero(checkpoint["state_dict"][name] - initial[name]))
            momentum = checkpoint["optimizer"]["state"].get(key, {}).get("momentum_buffer")
            held = route == "harp" and variant != "r1" and epoch == 1 and adapter.added(name)
            if held:
                assert momentum is None and changed == 0, name
            else:
                assert momentum is not None and torch.isfinite(momentum).all(), name
            rows.append({"name": name, "changed_elements": changed,
                         "momentum_nonzero": int(torch.count_nonzero(momentum)) if momentum is not None else 0})
    assert {r["name"] for r in rows} == set(initial)
    groups = ["visual_prompt", "projector"]
    if variant not in ("r1", "r0") and not (route == "harp" and epoch == 1):
        groups.append("adaptation")
    for group in groups:
        predicate = ((lambda n: adapter.added(n)) if group == "adaptation" else
                     (lambda n: n.startswith("VPT_image_trans.")) if group == "projector" else
                     (lambda n: n.startswith("image_encoder.") and not adapter.added(n)))
        assert any(r["changed_elements"] and r["momentum_nonzero"] for r in rows if predicate(r["name"])), group
    if route == "larp" and variant != "r1":
        assert any(r["changed_elements"] and r["momentum_nonzero"] for r in rows if "lora_B_o" in r["name"])
    return rows


def evaluate(adapter, tr, reg, dataset, seed, variant, out):
    import torch
    formal = pair_dir(reg, dataset, seed) / "formal" / variant
    complete = read(formal / "train_complete.json")
    assert complete["status"] == "passed" and complete["epochs"] == 20
    checkpoint_path = formal / "VLPromptLearner/model-best.pth.tar"
    assert sha(checkpoint_path) == complete["best_sha256"]
    assert sha(formal / "VLPromptLearner/model.pth.tar-20") == complete["last_sha256"]
    initial = torch.load(formal / "initial_trainable.pt", map_location="cpu")
    assert hash_model(adapter, tr.model) == read(formal / "initial_shared.json")
    frozen, audits, best = hash_model(adapter, tr.model, True), {}, None
    for tag, name in (("best", "model-best.pth.tar"), ("last", "model.pth.tar-20")):
        checkpoint = torch.load(formal / "VLPromptLearner" / name, map_location="cpu")
        audits[tag] = audit_checkpoint(adapter, tr, checkpoint, initial, frozen, reg["route"], variant, tag, torch)
        if tag == "best":
            best = checkpoint
    epochs = [json.loads(line) for line in (formal / "epochs.jsonl").read_text().splitlines()]
    assert len(epochs) == 20 and [x["epoch"] for x in epochs] == list(range(1,21))
    assert max(epochs, key=lambda x: x["base"])["epoch"] == best["epoch"], "Best does not match strict first maximum Base"
    tr.load_model(str(formal))
    state_exact(tr, best, torch)
    tr.epoch = best["epoch"] - 1
    module = adapter.evidence_module()
    evidence = module.attach(tr, torch, adapter.tensor_hash)
    base = float(tr.test("val"))
    bc, base_evidence = [tr.evaluator._total, tr.evaluator._correct], evidence["last"]
    novel = float(tr.test("test"))
    nc, novel_evidence = [tr.evaluator._total, tr.evaluator._correct], evidence["last"]
    assert (bc[0], nc[0]) == tuple(reg.get("expected_counts", {}).get(dataset, SPECS[dataset])[1:3])
    assert math.isfinite(base + novel) and 0 <= base <= 100 and 0 <= novel <= 100
    checkpoint_hash = sha(checkpoint_path)
    # Persist the first independent evaluation before a comparison can fail.
    for label, record in (("val", base_evidence), ("test", novel_evidence)):
        torch.save(dict(record, checkpoint_sha256=checkpoint_hash), out / f"canonical_{label}_evidence.pt")
    result = {"status": "recorded_before_hard_checks", "base": base, "novel": novel,
              "hm": 2 * base * novel / (base + novel) if base + novel else 0., "selected_epoch": best["epoch"],
              "counts_base_novel": [bc,nc], "checkpoint_sha256": checkpoint_hash}
    dump(out / "first_independent_metrics.json", result)
    checks = {}
    for label, record in (("val",base_evidence), ("test",novel_evidence)):
        prior = torch.load(formal / f"native_{label}_evidence.pt", map_location="cpu")
        assert prior["checkpoint_sha256"] == checkpoint_hash
        torch.save({"first": record, "reference": prior, "tolerances": TOLERANCES}, out / f"reload_pair_native_{label}.pt")
        checks["native_" + label] = module.compare(record, prior, torch, TOLERANCES)
    selected = torch.load(formal / "best_val_evidence.pt", map_location="cpu")
    assert selected["checkpoint_sha256"] == checkpoint_hash
    assert abs(selected["accuracy"] - float(best["val_result"])) < 1e-8
    torch.save({"first": base_evidence, "reference": selected, "tolerances": TOLERANCES}, out / "reload_pair_selection_val.pt")
    checks["selection_val"] = module.compare(base_evidence, selected, torch, TOLERANCES)
    result.update(status="passed", route=reg["route"], dataset=dataset, seed=seed, variant=variant, epochs=20,
                  kd=200 if dataset in ("dtd", "fgvc_aircraft", "oxford_flowers") else 1000,
                  time=now(), audits=audits, reload_audit=checks, best_last_sha256=[complete["best_sha256"],complete["last_sha256"]],
                  canonical_evaluation="first_independent_reload", release_manifest_sha256=reg["release_manifest_sha256"])
    dump(out / "audited_metrics.json", result)
    dump(formal / "audited_metrics.json", result)


def preserve_failure(adapter, tr, out):
    import numpy as np
    import torch
    path = out / "first_failure_state.pt"
    if path.exists():
        return
    torch.save({"model": tr.model.state_dict(), "optimizer": tr.optim.state_dict(), "scheduler": tr.sched.state_dict(),
        "epoch": getattr(tr, "epoch", None), "batch_idx": getattr(tr, "batch_idx", None), "batch": getattr(tr, "failure_batch", None),
        "gradients": {n: None if p.grad is None else p.grad.detach().cpu() for n,p in tr.model.named_parameters() if p.requires_grad},
        "python_rng": random.getstate(), "numpy_rng": np.random.get_state(), "cpu_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}, path)


def worker(reg, action, dataset, seed, variant, adapter=None):
    import torch
    testing = adapter is not None
    if not testing:
        verify_registration(reg)
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1, "Use one visible CUDA GPU"
        adapter = load_adapter(reg["route"])
        validate_selected_metadata(reg, dataset)
    pair = pair_dir(reg, dataset, seed)
    section = "initial" if action == "init" else "formal" if action == "train" else "evaluation" if action == "evaluate" else "gate"
    out = pair / section / variant
    out.mkdir(parents=True, exist_ok=False)
    cfg = config_for(adapter, reg, dataset, seed, variant, out)
    dump(out / "resolved.json", cfg)
    tr = None
    try:
        reference, reference_variant = None, None
        if action == "init":
            anchor = ANCHOR[reg["route"]]
            if variant not in ("r1", "r0", anchor):
                reference_variant = anchor
            elif reg["route"] != "r0" and variant == anchor:
                reference_variant = "r1"
        else:
            reference_variant = variant
        if reference_variant:
            path = initial_path(reg, dataset, seed, reference_variant)
            receipt = read(path.with_name("audit.json"))
            assert receipt["status"] == "passed" and receipt["state_sha256"] == sha(path), "Initialization artifact changed"
            reference = torch.load(path, map_location="cpu")
        tr = adapter.build(cfg, device="cpu" if testing else "cuda", init_reference=reference)
        if reference and action != "init":
            assert_initial(adapter, tr, reference)
        if action == "init":
            state = adapter.capture_initial(tr)
            if reference:
                assert state["common"] == reference["common"] and state["rng"] == reference["rng"], "Public initialization/RNG differ from R1"
            torch.save(state, out / "state.pt")
            dump(out / "audit.json", {"status": "passed", "dataset": dataset, "seed": seed, "variant": variant,
                 "state_sha256": sha(out / "state.pt"), "common": state["common"], "rng": state["rng"]})
        elif action in ("gate", "train"):
            adapter.install_schedule(tr, variant, len(tr.train_loader_x))
            (gate if action == "gate" else train)(adapter, tr, reg, dataset, seed, variant, out)
        elif action == "evaluate":
            evaluate(adapter, tr, reg, dataset, seed, variant, out)
        else:
            raise ValueError(action)
        dump(out / "action_complete.json", {"status": "passed", "action": action, "time": now(), "variant": variant})
        return out
    except BaseException:
        failure = traceback.format_exc()
        if tr is not None:
            try:
                preserve_failure(adapter, tr, out)
            except BaseException:
                failure += "\nFailure-state capture also failed:\n" + traceback.format_exc()
        dump(out / "failure.json", {"status": "failed", "action": action, "time": now(), "error": failure,
             "automatic_retry": False, "original_hard_limits_preserved": True})
        raise


def validate_selected_metadata(reg, dataset):
    """Reject changed/missing fixed splits before upstream readers can generate any."""
    directories = {"caltech101": "caltech-101", "oxford_flowers": "oxford_flowers", "food101": "food-101"}
    directory = directories.get(dataset, dataset)
    items = [entry for entry in read(ROOT / "metadata/external_assets.json")["assets"]
             if entry["kind"] == "data_metadata" and entry["relative_path"].split("/")[0] == directory]
    assert len(items) == (4 if dataset == "fgvc_aircraft" else 1), "Metadata manifest incomplete"
    for entry in items:
        path = Path(reg["data_root"]) / entry["relative_path"]
        assert path.is_file() and path.stat().st_size == entry["bytes"] and sha(path) == entry["sha256"], ("Fixed metadata mismatch", entry["relative_path"])


def execution_plan(route, datasets, seeds, variants, all_variants=False):
    if route not in VARIANTS:
        raise ValueError(route)
    datasets = list(dict.fromkeys(datasets or ["dtd","oxford_pets","fgvc_aircraft","oxford_flowers","caltech101","stanford_cars","ucf101","eurosat","sun397","food101"]))
    seeds = list(dict.fromkeys(seeds or [1,2,3]))
    requested = list(VARIANTS[route]) if all_variants else list(dict.fromkeys(variants or (["r0"] if route == "r0" else ["r1", ANCHOR[route]])))
    if any(x not in SPECS for x in datasets) or any(x not in (1,2,3) for x in seeds) or any(x not in VARIANTS[route] for x in requested):
        raise ValueError("Unknown dataset, seed or variant")
    if route != "r0" and "r1" not in requested:
        requested.insert(0, "r1")
    requested = [v for v in VARIANTS[route] if v in requested]
    initials = list(requested)
    if any(v not in ("r1", "r0", ANCHOR[route]) for v in requested) and ANCHOR[route] not in initials:
        initials.insert(1, ANCHOR[route])
    actions = []
    for dataset in datasets:
        for seed in seeds:
            actions.extend({"action": "init", "dataset": dataset, "seed": seed, "variant": v} for v in initials)
            actions.extend({"action": "gate", "dataset": dataset, "seed": seed, "variant": v} for v in requested)
            actions.extend({"action": action, "dataset": dataset, "seed": seed, "variant": v}
                           for v in requested for action in ("train", "evaluate"))
    return {"route": route, "datasets": datasets, "seeds": seeds, "variants": requested,
            "formal_runs": len(datasets) * len(seeds) * len(requested),
            "formal_epochs": 20 * len(datasets) * len(seeds) * len(requested),
            "engineering_steps": 4 * len(datasets) * len(seeds) * len(requested),
            "actions": actions, "epochs_each": 20, "automatic_retry": False}


def worker_environment(reg):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=reg["gpu"], OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
               OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    env.pop("PYTHONPATH", None)
    env.pop("CUBLAS_WORKSPACE_CONFIG", None)
    env.pop("PYTHONOPTIMIZE", None)
    return env


def gpu_lock(reg):
    import fcntl
    lock_path = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "promptkd-harp-larp" / "gpu.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        lock.close()
        raise
    return lock


def execute_item(reg, item, logfile):
    logfile = Path(logfile)
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open("x") as stream:
        subprocess.run([sys.executable, "-B", str(ROOT / "reproduce.py"), "_worker", "--registration",
            str(Path(reg["output_root"]) / "registration.json"), "--action", item["action"],
            "--dataset", item["dataset"], "--seed", str(item["seed"]), "--variant", item["variant"]],
            env=worker_environment(reg), cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)


def first_evaluation(reg, item):
    """Standalone entry uses the identical process environment and GPU lock."""
    verify_registration(reg)
    lock = gpu_lock(reg)
    try:
        logfile = Path(reg["output_root"]) / "console" / f"{item['dataset']}_seed{item['seed']}_{item['variant']}_evaluate.log"
        execute_item(reg, item, logfile)
    finally:
        lock.close()


def run_queue(reg):
    output = Path(reg["output_root"])
    output.mkdir(parents=True, exist_ok=False)
    dump(output / "registration.json", reg)
    lock = None
    state = {"status": "running", "started": now(), "completed": [], "automatic_retry": False, "pid": os.getpid()}
    dump(output / "status.json", state)
    try:
        verify_registration(reg)
        if shutil.disk_usage(output).free < 40 * 1024**3:
            raise RuntimeError("At least 40 GiB free disk is required before starting")
        # Shared by all routes on this host/user, including manual evaluation.
        lock = gpu_lock(reg)
        for item in reg["plan"]["actions"]:
            state.update(current=item, updated=now())
            dump(output / "status.json", state)
            verify_registration(reg)
            logfile = output / "console" / f"{item['dataset']}_seed{item['seed']}_{item['variant']}_{item['action']}.log"
            execute_item(reg, item, logfile)
            state["completed"].append(item)
            dump(output / "status.json", state)
        state.update(status="completed", finished=now(), current=None)
    except BaseException:
        state.update(status="failed", finished=now(), error=traceback.format_exc())
        raise
    finally:
        dump(output / "status.json", state)
        if lock is not None:
            lock.close()
    return state
