import json
from argparse import Namespace
from pathlib import Path

import numpy as np
from PIL import Image

from train import BundleDataset


def create_dummy_dataset(tmp_path: Path):
    dataset_dir = tmp_path / "ds"
    dataset_dir.mkdir()

    frames = []
    for i in range(2):
        img = Image.new("RGB", (4, 4), color=(i * 40, 0, 255 - i * 40))
        img_path = dataset_dir / f"{i:03d}.png"
        img.save(img_path)
        frames.append({
            "file_path": f"./{i:03d}.png",
            "fl_x": 2.0,
            "fl_y": 2.0,
            "cx": 2.0,
            "cy": 2.0,
            "w": 4,
            "h": 4,
            "transform_matrix": [
                [1, 0, 0, i * 0.1],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
        })
    with open(dataset_dir / "transforms.json", "w") as f:
        json.dump({"frames": frames}, f)
    return dataset_dir


def test_nerfstudio_loader(tmp_path):
    ds_dir = create_dummy_dataset(tmp_path)
    args = Namespace(
        data_path=str(ds_dir),
        frames=None,
        point_batch_size=8,
        num_batches=1,
        rolling_shutter=False,
        cache=False,
        max_percentile=100,
    )

    dataset = BundleDataset(args)
    dataset.load_volume()

    assert dataset.num_frames == 2
    assert dataset.rgb_volume.shape == (2, 3, 4, 4)
    sample = dataset[0]
    assert len(sample) == 6
