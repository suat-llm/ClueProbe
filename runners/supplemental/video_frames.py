#!/usr/bin/env python3
"""Official-style frame sampling shared by InternVL and InternVideo2.5."""

from __future__ import annotations

import numpy as np
import torch
import torchvision.transforms as transforms
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int):
    return transforms.Compose(
        [
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.Resize(
                (input_size, input_size), interpolation=InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_index(fps: float, max_frame: int, num_segments: int) -> np.ndarray:
    start_idx = 0
    end_idx = max_frame
    segment_size = float(end_idx - start_idx) / num_segments
    return np.array(
        [
            int(start_idx + segment_size / 2 + np.round(segment_size * index))
            for index in range(num_segments)
        ]
    )


def load_video(video_path: str, input_size: int, num_segments: int):
    reader = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    if len(reader) == 0:
        raise ValueError(f"video has no decodable frames: {video_path}")
    max_frame = len(reader) - 1
    fps = float(reader.get_avg_fps())
    transform = build_transform(input_size)
    frame_indices = get_index(fps, max_frame, num_segments)

    pixel_values_list = []
    num_patches_list = []
    for frame_index in frame_indices:
        image = Image.fromarray(reader[int(frame_index)].asnumpy()).convert("RGB")
        # Official video examples use max_num=1, so each frame is one 448px tile.
        pixel_values = torch.stack([transform(image)])
        pixel_values_list.append(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
    return torch.cat(pixel_values_list), num_patches_list

