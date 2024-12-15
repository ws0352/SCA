import glob
import os
import numpy as np
import logging

from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler
from skimage.io import imread
from albumentations.pytorch import ToTensorV2
from albumentations import HorizontalFlip, VerticalFlip, RandomRotate90, Normalize
from albumentations import OneOf, Compose
from ever.interface import ConfigurableMixin
from ever.api.data import CrossValSamplerGenerator

class TrainDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transforms=None):
        self.rgb_filepath_list = []
        self.cls_filepath_list = []

        if isinstance(image_dir, list):
            for img_dir_path, mask_dir_path in zip(image_dir, mask_dir):
                self.batch_generate(img_dir_path, mask_dir_path)
        else:
            self.batch_generate(image_dir, mask_dir)

        self.transforms = transforms

    def batch_generate(self, image_dir, mask_dir):
        rgb_filepath_list = []
        cls_filepath_list = []

        rgb_filepath_list += glob.glob(os.path.join(image_dir, '*.tif'))
        rgb_filepath_list += glob.glob(os.path.join(image_dir, '*.png'))
        rgb_filename_list = [os.path.split(fp)[-1] for fp in rgb_filepath_list]

        if mask_dir is not None:
            for fn in rgb_filename_list:
                cls_filepath_list.append(os.path.join(mask_dir, fn))

        self.rgb_filepath_list += rgb_filepath_list
        self.cls_filepath_list += cls_filepath_list

    def __getitem__(self, idx):
        image = imread(self.rgb_filepath_list[idx])

        # 0~7 -> -1~6 (-1: ignore label, 0~6: valid label)
        mask = imread(self.cls_filepath_list[idx]).astype(np.int64) - 1

        if self.transforms is not None:
            blob = self.transforms(image=image, mask=mask)
            image = blob['image']
            mask = blob['mask']

        return image, dict(cls=mask, fname=os.path.basename(self.rgb_filepath_list[idx]))

    def __len__(self):
        return len(self.rgb_filepath_list)


class TrainLoader(DataLoader, ConfigurableMixin):
    def __init__(self, config, logger=None):
        ConfigurableMixin.__init__(self, config)

        dataset = TrainDataset(self.config.image_dir, self.config.mask_dir, self.config.transforms)

        if logger is None:
            logger = logging.getLogger(__name__)

        logger.info('Dataset Image Number: %d' % len(dataset))

        if self.config.CV.i != -1:
            CV = CrossValSamplerGenerator(dataset, distributed=True, seed=3111)
            sampler_pairs = CV.k_fold(self.config.CV.k)
            train_sampler, val_sampler = sampler_pairs[self.config.CV.i]
            if self.config.training:
                sampler = train_sampler
            else:
                sampler = val_sampler
        else:
            sampler = RandomSampler(dataset) if self.config.training else SequentialSampler(dataset)

        super(TrainLoader, self).__init__(
            dataset,
            self.config.batch_size,
            sampler=sampler,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True
        )

    def set_default_config(self):
        self.config.update(dict(
            image_dir=None,
            mask_dir=None,
            batch_size=4,
            num_workers=4,
            scale_size=None,
            transforms=Compose([
                OneOf([
                    HorizontalFlip(True),
                    VerticalFlip(True),
                    RandomRotate90(True),
                ], p=0.75),
                Normalize(
                    mean=(),
                    std=(),
                    max_pixel_value=1,
                    always_apply=True
                ),
                ToTensorV2()
            ]),
        ))


class TestDataset(Dataset):
    def __init__(self, image_dir, transforms=None):
        self.rgb_filepath_list = []

        if isinstance(image_dir, list):
            for img_dir_path in zip(image_dir):
                self.batch_generate(img_dir_path)
        else:
            self.batch_generate(image_dir)

        self.transforms = transforms

    def batch_generate(self, image_dir):
        rgb_filepath_list = []

        image_dir = image_dir[0]
        rgb_filepath_list += glob.glob(os.path.join(image_dir, '*.tif'))
        rgb_filepath_list += glob.glob(os.path.join(image_dir, '*.png'))

        self.rgb_filepath_list += rgb_filepath_list

    def __getitem__(self, idx):
        image = imread(self.rgb_filepath_list[idx])

        mask = np.zeros([512, 512])

        if self.transforms is not None:
            blob = self.transforms(image=image, mask=mask)
            image = blob['image']

        return dict(rgb=image, fname=os.path.basename(self.rgb_filepath_list[idx]))

    def __len__(self):
        return len(self.rgb_filepath_list)


class TestLoader(DataLoader, ConfigurableMixin):
    def __init__(self, config, logger=None):
        ConfigurableMixin.__init__(self, config)

        dataset = TestDataset(self.config.image_dir, self.config.transforms)

        if logger is None:
            logger = logging.getLogger(__name__)

        logger.info('Test Dataset Image Number: %d' % len(dataset))

        if self.config.CV.i != -1:
            CV = CrossValSamplerGenerator(dataset, distributed=True, seed=3111)
            sampler_pairs = CV.k_fold(self.config.CV.k)
            train_sampler, val_sampler = sampler_pairs[self.config.CV.i]
            if self.config.training:
                sampler = train_sampler
            else:
                sampler = val_sampler
        else:
            sampler = RandomSampler(dataset) if self.config.training else SequentialSampler(dataset)

        super(TestLoader, self).__init__(
            dataset,
            self.config.batch_size,
            sampler=sampler,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True
        )

    def set_default_config(self):
        self.config.update(dict(
            image_dir=None,
            mask_dir=None,
            batch_size=4,
            num_workers=4,
            scale_size=None,
            transforms=Compose([
                OneOf([
                    HorizontalFlip(True),
                    VerticalFlip(True),
                    RandomRotate90(True),
                ], p=0.75),
                Normalize(
                    mean=(),
                    std=(),
                    max_pixel_value=1,
                    always_apply=True
                ),
                ToTensorV2()
            ]),
        ))
