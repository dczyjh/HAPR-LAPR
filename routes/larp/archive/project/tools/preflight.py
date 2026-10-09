from __future__ import annotations

import argparse
import importlib
from pathlib import Path


DATASET_DIRS = {
    "caltech101": "caltech-101",
    "oxford_pets": "oxford_pets",
    "stanford_cars": "stanford_cars",
    "oxford_flowers": "oxford_flowers",
    "food101": "food-101",
    "fgvc_aircraft": "fgvc_aircraft",
    "sun397": "sun397",
    "dtd": "dtd",
    "eurosat": "eurosat",
    "ucf101": "ucf101",
}

TEACHER_DIRS = {
    "caltech101": "Caltech101",
    "oxford_pets": "OxfordPets",
    "stanford_cars": "StanfordCars",
    "oxford_flowers": "OxfordFlowers",
    "food101": "Food101",
    "fgvc_aircraft": "FGVCAircraft",
    "sun397": "SUN397",
    "dtd": "DescribableTextures",
    "eurosat": "EuroSAT",
    "ucf101": "UCF101",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--datasets", nargs="+", choices=list(DATASET_DIRS), default=list(DATASET_DIRS))
    parser.add_argument("--weights-root", type=Path)
    parser.add_argument("--teacher-root", type=Path)
    parser.add_argument("--verify-hashes", action="store_true")
    args = parser.parse_args()

    failures = []
    for module_name in ("torch", "torchvision", "yacs", "dassl"):
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "unknown")
            print(f"[OK] import {module_name} ({version})")
        except Exception as error:
            failures.append(f"cannot import {module_name}: {error}")

    try:
        import torch

        if not torch.cuda.is_available():
            failures.append("torch.cuda.is_available() is False")
        else:
            print(f"[OK] GPU: {torch.cuda.get_device_name(0)}")
            print(f"[OK] CUDA runtime: {torch.version.cuda}")
            capability = torch.cuda.get_device_capability(0)
            print(f"[OK] compute capability: {capability[0]}.{capability[1]}")
    except Exception:
        pass

    weights_root = args.weights_root or args.project_root / "clip"
    teacher_root = args.teacher_root or args.project_root / "teacher_model"
    for relative_path in ("ViT-B-16.pt", "ViT-L-14.pt"):
        path = weights_root / relative_path
        if path.is_file():
            print(f"[OK] {relative_path}")
            if args.verify_hashes:
                import hashlib
                expected = {"ViT-B-16.pt": "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f",
                            "ViT-L-14.pt": "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"}
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                        digest.update(chunk)
                if digest.hexdigest() != expected[relative_path]:
                    failures.append(f"CLIP checksum mismatch: {path}")
        else:
            failures.append(f"missing {path}")

    for dataset, directory in DATASET_DIRS.items():
        if dataset not in args.datasets:
            continue
        path = args.data_root / directory
        if path.is_dir():
            print(f"[OK] dataset {dataset}: {path}")
            required = (
                ["variants.txt", "images_variant_train.txt", "images_variant_val.txt", "images_variant_test.txt"]
                if dataset == "fgvc_aircraft"
                else [f"split_zhou_{TEACHER_DIRS[dataset]}.json"]
            )
            for filename in required:
                if not (path / filename).is_file():
                    failures.append(f"missing fixed split metadata: {path / filename}")
        else:
            failures.append(f"missing dataset {dataset}: {path}")

    for dataset, directory in TEACHER_DIRS.items():
        if dataset not in args.datasets:
            continue
        path = teacher_root / directory / "VLPromptLearner" / "model-best.pth.tar"
        if path.is_file():
            print(f"[OK] teacher {dataset}: {path}")
        else:
            failures.append(f"missing teacher {dataset}: {path}")

    if failures:
        print("\nPreflight failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("\nPreflight passed. The project is ready for a one-dataset smoke run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
