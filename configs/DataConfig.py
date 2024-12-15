from albumentations import *
import ever as er

SOURCE_DATA_CONFIG = dict(
    image_dir = ['./LoveDA/Train/Rural/images_png/'],
    # label
    mask_dir = ['./LoveDA/Train/Rural/masks_png/'],
    transforms=Compose([
        RandomCrop(512, 512),
        OneOf([
            HorizontalFlip(True),
            VerticalFlip(True),
            RandomRotate90(True)
        ], p=0.75),
        Normalize(
            mean=(69.72734352, 77.96519433, 76.02640537),
            std=(40.23553791, 35.05373492, 33.22030439),
            max_pixel_value=1,
            always_apply=True
        ),
        er.preprocess.albu.ToTensor()
    ]),
    CV=dict(k=10, i=-1),
    training=True,
    batch_size=10,
    num_workers=8,
)

TARGET_DATA_CONFIG = dict(
    image_dir = ['./LoveDA/Val/Urban/images_png/'],
    mask_dir = ['./LoveDA/Val/Urban/masks_png/'],
    transforms=Compose([
        RandomCrop(512, 512),
        OneOf([
            HorizontalFlip(True),
            VerticalFlip(True),
            RandomRotate90(True)
        ], p=0.75),
        Normalize(
            mean=(73.5995142, 78.71683497, 76.75850136),
            std=(39.68001699, 36.70139487, 35.74589312),
            max_pixel_value=1,
            always_apply=True
        ),
        er.preprocess.albu.ToTensor()
    ]),
    CV=dict(k=10, i=-1),
    training=True,
    batch_size=10,
    num_workers=8,
)

EVAL_DATA_CONFIG = dict(
    image_dir = ['./LoveDA/Val/Urban/images_png/'],
    mask_dir = ['./LoveDA/Val/Urban/masks_png/'],
    transforms=Compose([
        Normalize(
            mean=(73.5995142, 78.71683497, 76.75850136),
            std=(39.68001699, 36.70139487, 35.74589312),
            max_pixel_value=1,
            always_apply=True
        ),
        er.preprocess.albu.ToTensor()
    ]),
    CV=dict(k=10, i=-1),
    training=False,
    batch_size=1,
    num_workers=8,
)

TEST_DATA_CONFIG = dict(
    image_dir = ['./LoveDA/Test/Urban/images_png/'],
    transforms=Compose([
        Normalize(
            mean=(73.5995142, 78.71683497, 76.75850136),
            std=(39.68001699, 36.70139487, 35.74589312),
            max_pixel_value=1,
            always_apply=True
        ),
        er.preprocess.albu.ToTensor()
    ]),
    CV=dict(k=10, i=-1),
    training=False,
    batch_size=1,
    num_workers=8,
)