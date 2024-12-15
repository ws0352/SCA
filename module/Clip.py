import torch.nn.functional as F
import ttach as tta
import torch
from math import *


"""
    Pad an image up to the target size.
"""
def pad_image(img, target_size):
    rows_missing = target_size[0] - img.shape[2]
    cols_missing = target_size[1] - img.shape[3]
    padded_img = F.pad(img, (0, 0, rows_missing, cols_missing), 'constant', 0)
    return padded_img


"""
    Test-Time Augmentation
"""
def tta_predict(model, img):
    tta_transforms = tta.Compose([
        tta.HorizontalFlip(),
        tta.Rotate90(angles=[0, 90, 180, 270]),
    ])

    xs = []

    for t in tta_transforms:
        aug_img = t.augment_image(img)
        aug_x = model(aug_img)
        # aug_x = F.softmax(aug_x, dim=1)

        x = t.deaugment_mask(aug_x)
        xs.append(x)

    xs = torch.cat(xs, 0)
    x = torch.mean(xs, dim=0, keepdim=True)

    return x


"""
    Sliding window prediction
"""
def pre_slide(model, image, num_classes=7, tile_size=(512, 512), tta=False):
    image_size = image.shape
    overlap = 1 / 2

    stride = ceil(tile_size[0] * (1 - overlap))
    tile_rows = int(ceil((image_size[2] - tile_size[0]) / stride) + 1)
    tile_cols = int(ceil((image_size[3] - tile_size[1]) / stride) + 1)

    full_probs = torch.zeros((1, num_classes, image_size[2], image_size[3])).cuda()

    count_predictions = torch.zeros((1, 1, image_size[2], image_size[3])).cuda()
    tile_counter = 0

    for row in range(tile_rows):
        for col in range(tile_cols):
            x1 = int(col * stride)
            y1 = int(row * stride)
            x2 = min(x1 + tile_size[1], image_size[3])
            y2 = min(y1 + tile_size[0], image_size[2])
            x1 = max(int(x2 - tile_size[1]), 0)
            y1 = max(int(y2 - tile_size[0]), 0)

            img = image[:, :, y1:y2, x1:x2]
            padded_img = pad_image(img, tile_size)

            tile_counter += 1

            if tta is True:
                padded = tta_predict(model, padded_img)
            else:
                padded = model(padded_img)
                # padded = F.softmax(padded, dim=1)

            pre = padded[:, :, 0:img.shape[2], 0:img.shape[3]]

            count_predictions[:, :, y1:y2, x1:x2] += 1
            full_probs[:, :, y1:y2, x1:x2] += pre

    # average the predictions in the overlapping regions
    full_probs /= count_predictions

    return full_probs