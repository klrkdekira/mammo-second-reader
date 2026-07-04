"""Albumentations augmentation pipelines.

No vertical flip. Mammograms are not vertically symmetric.

`train_augment` takes a `level` so augmentation strength is a config knob
(DataConfig.augment): "light" for flips only, "default" for the original
pipeline, "heavy" for stronger geometric/intensity jitter plus cutout.
"""


def train_augment(image_size: int = 224, level: str = "default"):
    import albumentations as A

    tail = [
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485], std=[0.229], max_pixel_value=1.0),
    ]
    level = level.lower()
    if level == "light":
        ops = [A.HorizontalFlip(p=0.5)]
    elif level == "default":
        ops = [
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.5, border_mode=0),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        ]
    elif level == "heavy":
        ops = [
            A.HorizontalFlip(p=0.5),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(-0.05, 0.05),
                rotate=(-15, 15),
                border_mode=0,
                p=0.7,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(0.05, 0.15),
                hole_width_range=(0.05, 0.15),
                fill=0,
                p=0.3,
            ),
        ]
    else:
        raise ValueError(f"Unknown augment level {level!r}")
    return A.Compose(ops + tail)


def val_augment(image_size: int = 224):
    import albumentations as A

    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485], std=[0.229], max_pixel_value=1.0),
        ]
    )
