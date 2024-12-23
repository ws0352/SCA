import torch.nn as nn
import torch.nn.functional as F
import torch
from audtorch.metrics.functional import pearsonr


class CategoryAlign_Module(nn.Module):
    def __init__(self, num_classes=7, ignore_bg=False):
        super(CategoryAlign_Module, self).__init__()

        self.num_classes = num_classes
        self.ignore_bg = ignore_bg

    def get_context(self, preds, feats):
        b, c, h, w = feats.size()
        _, num_cls, _, _ = preds.size()

        # softmax preds
        assert preds.max() <= 1 and preds.min() >= 0, print(preds.max(), preds.min())
        preds = preds.view(b, num_cls, 1, h * w)
        feats = feats.view(b, 1, c, h * w)

        vectors = (feats * preds).sum(-1) / preds.sum(-1)

        if self.ignore_bg:
            vectors = vectors[:, 1:, :]  # ignore the background

        vectors = F.normalize(vectors, dim=1)

        return vectors

    def get_intra_corcoef_mat(self, preds, feats):
        context = self.get_context(preds, feats).mean(0)

        n, c = context.size()
        mat = torch.zeros([n, n]).to(context.device)
        for i in range(n):
            for j in range(n):
                # cor = pearsonr(context[i, :], context[j, :])
                # mat[i, j] += cor[0]

                mu_f1 = torch.mean(context[i, :])
                mu_f2 = torch.mean(context[j, :])
                f1_centered = context[i, :] - mu_f1
                f2_centered = context[j, :] - mu_f2
                corr = torch.dot(f1_centered, f2_centered) / (torch.norm(f1_centered) * torch.norm(f2_centered) + 1e-6)  # 避免分母为0
                mat[i, j] += corr
        return mat

    def get_cross_corcoef_mat(self, preds1, feats1, preds2, feats2):
        context1 = self.get_context(preds1, feats1).mean(0)
        context2 = self.get_context(preds2, feats2).mean(0)

        n, c = context1.size()
        mat = torch.zeros([n, n]).to(context1.device)
        for i in range(n):
            for j in range(n):
                # cor = pearsonr(context1[i, :], context2[j, :])
                # mat[i, j] += cor[0]

                mu_f1 = torch.mean(context1[i, :])
                mu_f2 = torch.mean(context2[j, :])
                f1_centered = context1[i, :] - mu_f1
                f2_centered = context2[j, :] - mu_f2
                corr = torch.dot(f1_centered, f2_centered) / (torch.norm(f1_centered) * torch.norm(f2_centered) + 1e-6)  # 避免分母为0
                mat[i, j] += corr

        return mat

    def regularize(self, cor_mat):
        n = self.num_classes - 1 if self.ignore_bg else self.num_classes
        assert cor_mat.size()[0] == n

        pos = - torch.log(torch.diag(cor_mat)).mean()
        undiag = cor_mat.flatten()[:-1].view(n-1, n+1)[:, 1:].flatten()
        # undiag_clone = undiag.clone()
        low = torch.Tensor([1e-6]).to(undiag.device)
        neg = - torch.log(1 - undiag.max(low)).mean()

        loss = pos + neg
        return loss


"""
    Intra-domain Covariance Regularization
"""
def ICR(inputs, multi_layer=False, ignore_bg=True):

    m = CategoryAlign_Module(ignore_bg=ignore_bg)

    if multi_layer:
        preds1, preds2, feats = inputs

        B = preds1.size()[0]
        preds = ((preds1.softmax(dim=1) + preds2.softmax(dim=1))/2).detach()

        preds1, feats1 = preds[:B // 2, :, :, :], feats[:B // 2, :, :, :]
        preds2, feats2 = preds[B // 2:, :, :, :], feats[B // 2:, :, :, :]
        cor_mat = m.get_cross_corcoef_mat(preds1, feats1, preds2, feats2)
        loss = m.regularize(cor_mat)
    else:
        preds, feats = inputs
        # cor_mat = m.get_intra_corcoef_mat(preds, feats)

        B = preds.size()[0]
        preds = preds.softmax(dim=1).detach()
        preds1, feats1 = preds[:B // 2, :, :, :], feats[:B // 2, :, :, :]
        preds2, feats2 = preds[B // 2:, :, :, :], feats[B // 2:, :, :, :]
        cor_mat = m.get_cross_corcoef_mat(preds1, feats1, preds2, feats2)
        loss = m.regularize(cor_mat)
    return loss


"""
    Cross-domain Covariance Regularization
"""
def CCR(source, target, multi_layer=False, ignore_bg=True):

    m = CategoryAlign_Module(ignore_bg=ignore_bg)

    if multi_layer:
        S_preds1, S_preds2, S_feats = source
        T_preds1, T_preds2, T_feats = target

        S_preds = ((S_preds1.softmax(dim=1) + S_preds2.softmax(dim=1))/2)
        T_preds = ((T_preds1.softmax(dim=1) + T_preds2.softmax(dim=1))/2)

        cor_mat = m.get_cross_corcoef_mat(S_preds.detach(), S_feats.detach(), T_preds.detach(), T_feats)
        loss = m.regularize(cor_mat)

    else:
        S_preds, S_feats = source
        T_preds, T_feats = target
        S_preds, T_preds = S_preds.softmax(dim=1), T_preds.softmax(dim=1)

        cor_mat = m.get_cross_corcoef_mat(S_preds.detach(), S_feats.detach(), T_preds.detach(), T_feats)
        loss = m.regularize(cor_mat)
    return loss


# def MSE_intra(inputs, multi_layer=False, ignore_bg=True):
#     """
#         MSE-Alignment for intra domain
#     """
#     m = CategoryAlign_Module(ignore_bg=ignore_bg)
#
#     if multi_layer:
#         preds1, preds2, feats = inputs
#
#         B = preds1.size()[0]
#         preds = ((preds1.softmax(dim=1) + preds2.softmax(dim=1))/2).detach()
#         preds1, feats1 = preds[:B // 2, :, :, :], feats[:B // 2, :, :, :]
#         preds2, feats2 = preds[B // 2:, :, :, :], feats[B // 2:, :, :, :]
#         context1, context2 = m.get_context(preds1, feats1), m.get_context(preds2, feats2)
#         loss = F.mse_loss(context1, context2, reduction='mean')
#     else:
#         preds, feats = inputs
#         B = preds.size()[0]
#         preds = preds.softmax(dim=1).detach()
#         preds1, feats1 = preds[:B // 2, :, :, :], feats[:B // 2, :, :, :]
#         preds2, feats2 = preds[B // 2:, :, :, :], feats[B // 2:, :, :, :]
#         context1, context2 = m.get_context(preds1, feats1), m.get_context(preds2, feats2)
#         loss = F.mse_loss(context1, context2, reduction='mean')
#     return loss


# def MSE_cross(source, target, multi_layer=False, ignore_bg=True):
#     """
#         MSE-Alignment for cross domain
#     """
#     m = CategoryAlign_Module(ignore_bg=ignore_bg)
#
#     if multi_layer:
#         S_preds1, S_preds2, S_feats = source
#         T_preds1, T_preds2, T_feats = target
#
#         S_preds = ((S_preds1.softmax(dim=1) + S_preds2.softmax(dim=1))/2)
#         T_preds = ((T_preds1.softmax(dim=1) + T_preds2.softmax(dim=1))/2)
#
#         context1, context2 = m.get_context(S_preds.detach(), S_feats.detach()), \
#                              m.get_context(T_preds.detach(), T_feats)
#         loss = F.mse_loss(context1, context2, reduction='mean')
#
#     else:
#         S_preds, S_feats = source
#         T_preds, T_feats = target
#
#         S_preds, T_preds = S_preds.softmax(dim=1), T_preds.softmax(dim=1)
#
#         context1, context2 = m.get_context(S_preds.detach(), S_feats.detach()), \
#                              m.get_context(T_preds.detach(), T_feats)
#         loss = F.mse_loss(context1, context2, reduction='mean')
#     return loss


