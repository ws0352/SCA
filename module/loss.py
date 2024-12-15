import torch.nn.functional as F
import torch

"""
    cross entropy loss for semantic segmentation
"""
def loss_calc(pred, label, reduction='mean', multi=False):
    if multi is True:
        loss = 0
        num = 0
        for p in pred:
            if p.size()[-2:] != label.size()[-2:]:
                p = F.interpolate(p, size=label.size()[-2:], mode='bilinear', align_corners=True)
            l = F.cross_entropy(p, label.long(), ignore_index=-1, reduction=reduction)
            loss += l
            num += 1
        loss = loss / num
    else:
        if pred.size()[-2:] != label.size()[-2:]:
            pred = F.interpolate(pred, size=label.size()[-2:], mode='bilinear', align_corners=True)
        loss = F.cross_entropy(pred, label.long(), ignore_index=-1, reduction=reduction)

    return loss


def gradient_magnitude(x):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(x.device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(x.device)

    grad_x = F.conv2d(x, sobel_x, padding=1, groups=x.shape[1])
    grad_y = F.conv2d(x, sobel_y, padding=1, groups=x.shape[1])

    grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
    return grad_magnitude


def feature_contrast(x):
    mean = x.mean(dim=[2, 3], keepdim=True)
    var = ((x - mean) ** 2).mean(dim=[2, 3], keepdim=True)
    return var

def cloud_ratio(x, threshold=0.5):
    binary_cloud = (x > threshold).float()
    ratio = binary_cloud.mean(dim=[2, 3])
    return ratio

def loss_adapt(feat1, feat2, alpha=1.0, beta=1.0, gamma=1.0, threshold=0.5):
    grad1 = gradient_magnitude(feat1)
    grad2 = gradient_magnitude(feat2)
    loss_complexity = torch.mean(torch.abs(grad1 - grad2))

    contrast1 = feature_contrast(feat1)
    contrast2 = feature_contrast(feat2)
    loss_contrast = torch.mean(torch.abs(contrast1 - contrast2))

    cloud1 = cloud_ratio(feat1, threshold)
    cloud2 = cloud_ratio(feat2, threshold)
    loss_cloud = torch.mean(torch.abs(cloud1 - cloud2))

    loss = alpha * loss_complexity + beta * loss_contrast + gamma * loss_cloud
    return loss