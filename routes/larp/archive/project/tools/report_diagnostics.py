"""Summarize training diagnostics without pretending they measure accuracy."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    files = [args.root] if args.root.is_file() else sorted(args.root.rglob("diagnostics.jsonl"))
    if not files:
        print("No diagnostics found. No conclusion can be drawn.")
        return 2
    for path in files:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not rows:
            continue
        last = rows[-1]
        print(f"\n{path.parent}")
        print(f"sampled_steps={len(rows)}, last_step={last['step']}, groups={last['optimizer_groups']}")
        for name, group in last["groups"].items():
            zeros = sum(row["groups"][name]["update_norm"] == 0 for row in rows)
            print(f"{name}: grad_norm={group['gradient_norm']}, update_norm={group['update_norm']}, "
                  f"relative_update={group['relative_update_norm']}, zero_update_samples={zeros}/{len(rows)}")
        print("same-checkpoint module on/off:", last["contribution"])
        if last["lora_weights"]:
            print("LoRA weight-cast diagnostics:", last["lora_weights"])
    print("\nThese are training-probe signals, NOT accuracy or a comparison against a separately trained baseline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
