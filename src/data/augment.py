"""Albumentations augmentation pipelines.

No vertical flip. Mammograms are not vertically symmetric.

`train_augment` takes a `level` so augmentation strength is a config knob
(DataConfig.augment): "light" for flips only, "default" for the original
pipeline, "heavy" for stronger geometric/intensity jitter plus cutout.

Seeding: Albumentations 2.x gives every Compose its own random generator, so
`random.seed`/`np.random.seed` (and therefore `set_global_seed`) do NOT reach
these transforms. Pass `seed` to make the augmentation stream reproducible;
without it Albumentations picks its own nondeterministic seed and two runs of
the same config see different augmentations. The locked 22-run evidence
predates this argument and was trained with an unseeded stream.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import albumentations as A


def train_augment(
    image_size: int = 224, level: str = "default", seed: int | None = None
) -> "A.Compose":
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
    return A.Compose(ops + tail, seed=seed)


def val_augment(image_size: int = 224, seed: int | None = None) -> "A.Compose":
    """Deterministic by construction; `seed` is accepted for symmetry only."""
    import albumentations as A

    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485], std=[0.229], max_pixel_value=1.0),
        ],
        seed=seed,
    )
