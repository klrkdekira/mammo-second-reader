"""Albumentations augmentation pipelines.

No vertical flip. Mammograms are not vertically symmetric.
"""


def train_augment(image_size: int = 224):
    import albumentations as A

    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.5, border_mode=0),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485], std=[0.229], max_pixel_value=1.0),
        ]
    )


def val_augment(image_size: int = 224):
    import albumentations as A

    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485], std=[0.229], max_pixel_value=1.0),
        ]
    )
