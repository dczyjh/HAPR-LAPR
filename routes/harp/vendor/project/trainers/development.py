"""Use the original unused validation images, never the benchmark test split."""
import json
import math
from pathlib import Path
from dassl.data.datasets import Datum, DatasetBase


def base_development_dataset(dataset):
    """Keep student training unchanged; expose ONLY Base development evaluation.

    The released PromptKD loaders replace ``val`` with Base test. Recover the
    original val metadata explicitly; never randomly carve up test images.
    This proves student split separation, not an unknown teacher's provenance.
    """
    if hasattr(dataset, "split_path"):
        with Path(dataset.split_path).open() as stream:
            metadata = json.load(stream)
        if "val" not in metadata:
            raise ValueError("Original validation split is missing; do not fall back to test.")
        image_root = (dataset.dataset_dir if type(dataset).__name__ == "StanfordCars"
                      else dataset.image_dir)
        items = [Datum(impath=str(Path(image_root) / path), label=int(label), classname=name)
                 for path, label, name in metadata["val"]]
    elif type(dataset).__name__ == "FGVCAircraft":
        items = dataset.read_data({name: label for label, name in enumerate(dataset.classnames)},
                                  "images_variant_val.txt")
    else:
        raise ValueError("No audited original validation reader for this dataset.")

    n_base = math.ceil(dataset.num_classes / 2)
    development = [item for item in items if 0 <= item.label < n_base]
    if {item.label for item in development} != set(range(n_base)):
        raise ValueError("Base development split must contain every Base class.")
    paths = [str(Path(item.impath).resolve()) for item in development]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate development image paths.")
    forbidden = {str(Path(item.impath).resolve())
                 for item in dataset.train_x + (dataset.val or []) + (dataset.test or [])}
    if set(paths) & forbidden:
        raise ValueError("Development images overlap training or benchmark test images.")
    for item in development:
        if item.classname != dataset.classnames[item.label]:
            raise ValueError("Development class ordering differs from training/teacher classifier.")
    # DataManager requires a test loader. It receives development data too;
    # PromptKD.test additionally forbids split='test' in development mode.
    result = DatasetBase(train_x=dataset.train_x, val=development, test=development)
    result.development_source = "original_val_base_only"
    return result
