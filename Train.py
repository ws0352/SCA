import cv2
import os.path as osp
import torch.optim as optim

from ever.core.iterator import Iterator
from tqdm import tqdm
from torch.nn.utils import clip_grad

from utils.DataLoader import TrainLoader
from utils.Tools import *
from module.MyModel import Deeplabv2
from module.CR import *
from module.viz import VisualizeSegmm
from module.loss import loss_calc
from module.Clip import pre_slide
from Eval import evaluate


def train(cfg):
    logger = get_console_file_logger(name='311', logdir=cfg.SNAPSHOT_DIR)

    # Directory
    save_pseudo_label_path = osp.join(cfg.SNAPSHOT_DIR, 'pseudo_label')
    if not os.path.exists(save_pseudo_label_path):
        os.makedirs(save_pseudo_label_path)
    save_pseudo_color_path = save_pseudo_label_path + '_color'
    if not os.path.exists(save_pseudo_color_path):
        os.makedirs(save_pseudo_color_path)

    # Loader
    trainloader = TrainLoader(cfg.SOURCE_DATA_CONFIG, logger)
    trainloader_iter = Iterator(trainloader)
    evalloader = TrainLoader(cfg.EVAL_DATA_CONFIG, logger)
    targetloader = None

    logger.info('epochs ~= %.3f' % (cfg.NUM_STEPS_STOP / len(trainloader)))

    model = Deeplabv2(dict(
        backbone=dict(
            resnet_type='resnet50',
            output_stride=16,
            pretrained=True,
        ),
        multi_layer=True,
        cascade=False,
        use_ppm=True,
        ppm=dict(
            num_classes=7,
            fc_dim=2048,
            use_aux=False,
            pool_scales=(1, 2, 3, 6)
        ),
        # ppm1=dict(
        #     num_classes=7,
        #     fc_dim=2048,
        #     use_aux=False,
        #     pool_scales=(1, 2, 3, 6)
        # ),
        # ppm2=dict(
        #     num_classes=7,
        #     fc_dim=2048,
        #     use_aux=False,
        #     pool_scales=(1, 2, 4, 8)
        # ),
        inchannels=2048,
        num_classes=7
    )).cuda()

    # model_state_dict = torch.load('/root/autodl-tmp/SCA/log/2urban/Iter4000.pth')
    # model.load_state_dict(model_state_dict, strict=True)

    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        momentum=cfg.MOMENTUM,
        weight_decay=cfg.WEIGHT_DECAY
    )
    optimizer.zero_grad()

    for i_iter in tqdm(range(cfg.NUM_STEPS_STOP)):
        if i_iter <= cfg.FIRST_STAGE_STEP:
            # First Stage(Train with Source)
            if i_iter == 0:
                logger.info('###### First Stage!!! ######')

            optimizer.zero_grad()
            lr = adjust_learning_rate(optimizer, i_iter, cfg)

            batch = trainloader_iter.next()
            images_s, labels_s = batch[0]
            preds1, preds2, feats = model(images_s.cuda())

            loss_seg = loss_calc([preds1, preds2], labels_s['cls'].cuda(), multi=True)
            source_intra = ICR([preds1, preds2, feats], multi_layer=True)
            loss = loss_seg + cfg.LAMBDA2 * source_intra

            loss.backward()
            clip_grad.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                max_norm=35,
                norm_type=2
            )
            optimizer.step()

            if i_iter % 50 == 0:
                text = 'iter = %d, total = %.3f, seg = %.3f, sour_intra = %.3f, lr = %.3f' % (i_iter, loss, loss_seg, source_intra, lr)
                logger.info(text)

            if i_iter % cfg.EVAL_EVERY == 0 and i_iter != 0:
                ckpt_path = osp.join(cfg.SNAPSHOT_DIR, 'Iter' + str(i_iter) + '.pth')
                torch.save(model.state_dict(), ckpt_path)
                evaluate(model, cfg, True, ckpt_path, logger)
                model.train()
        else:
            # Second Stage(Generate pseudo label, Train with Target)
            if i_iter == (cfg.FIRST_STAGE_STEP + 1):
                logger.info('###### Second Stage in round {}!!! ######'.format(i_iter))

            if i_iter % cfg.GENERATE_PSEDO_EVERY == 0 or targetloader is None:
                logger.info('###### Start generate pseudo and model retraining in round {}! ######'.format(i_iter))

                gener_target_pseudo(cfg, model, evalloader, save_pseudo_label_path, save_pseudo_color_path)

                target_config = cfg.TARGET_DATA_CONFIG
                target_config['mask_dir'] = [save_pseudo_label_path]

                targetloader = TrainLoader(target_config, logger)
                targetloader_iter = Iterator(targetloader)

            torch.cuda.synchronize()

            if i_iter < cfg.NUM_STEPS_STOP and targetloader is not None:
                model.train()

                optimizer.zero_grad()
                lr = adjust_learning_rate(optimizer, i_iter, cfg)

                # source output
                batch_s = trainloader_iter.next()
                images_s, label_s = batch_s[0]
                pred_s1, pred_s2, feat_s = model(images_s.cuda())

                # target output
                batch_t = targetloader_iter.next()
                images_t, label_t = batch_t[0]
                pred_t1, pred_t2, feat_t = model(images_t.cuda())

                # loss
                loss_seg = loss_calc([pred_s1, pred_s2], label_s['cls'].cuda(), multi=True)
                loss_pseudo = loss_calc([pred_t1, pred_t2], label_t['cls'].cuda(), multi=True)
                source_intra = ICR([pred_s1, pred_s2, feat_s], multi_layer=True)
                # target_intra = ICR([pred_t1, pred_t2, feat_t], multi_layer=True)
                domain_cross = CCR([pred_s1, pred_s2, feat_s],[pred_t1, pred_t2, feat_t], multi_layer=True)
                # loss_adapt = loss_adapt(feat_s, feat_t)
                # loss = loss_seg + cfg.LAMBDA1 * loss_pseudo + cfg.LAMBDA2 * source_intra + cfg.LAMBDA3 * domain_cross + cfg.LAMBDA4 * loss_adapt
                loss = loss_seg + cfg.LAMBDA1 * loss_pseudo + cfg.LAMBDA2 * source_intra + cfg.LAMBDA3 * domain_cross

                loss.backward()
                clip_grad.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    max_norm=35,
                    norm_type=2
                )
                optimizer.step()

                if i_iter % 50 == 0:
                    text = 'iter = %d, total = %.3f, seg = %.3f, pseudo = %.3f, sour_intra = %.3f, cross = %.3f, lr = %.3f' % (i_iter, loss, loss_seg, loss_pseudo, source_intra, domain_cross, lr)
                    logger.info(text)

                if i_iter % cfg.EVAL_EVERY == 0 and i_iter != 0:
                    ckpt_path = osp.join(cfg.SNAPSHOT_DIR, 'Iter' + str(i_iter) + '.pth')
                    torch.save(model.state_dict(), ckpt_path)
                    evaluate(model, cfg, True, ckpt_path, logger)
                    model.train()


def gener_target_pseudo(cfg, model, evalloader, save_pseudo_label_path, save_pseudo_color_path):
    model.eval()

    palette = np.asarray(list(COLOR_MAP.values())).reshape((-1,)).tolist()
    viz_op = VisualizeSegmm(save_pseudo_color_path, palette)

    with torch.no_grad():
        for ret, ret_gt in tqdm(evalloader):
            ret = ret.to(torch.device('cuda'))

            # cls = model(ret)
            cls = pre_slide(model, ret, tta=True)

            if cfg.PSEUDO_SELECT:
                cls = pseudo_selection(cls)
            else:
                cls = cls.argmax(dim=1).cpu().numpy()

            cv2.imwrite(save_pseudo_label_path + '/' + ret_gt['fname'][0], (cls + 1).reshape(1024, 1024).astype(np.uint8))

            if cfg.SNAPSHOT_DIR is not None:
                for fname, pred in zip(ret_gt['fname'], cls):
                    viz_op(pred, fname.replace('tif', 'png'))


def pseudo_selection(mask, cutoff_top=0.8, cutoff_low=0.6):
    """Convert continuous mask into binary mask"""
    assert mask.max() <= 1 and mask.min() >= 0, print(mask.max(), mask.min())

    bs, c, h, w = mask.size()
    mask = mask.view(bs, c, -1)

    # for each class extract the max confidence
    mask_max, _ = mask.max(-1, keepdim=True)
    mask_max *= cutoff_top

    # if the top score is too low, ignore it
    lowest = torch.Tensor([cutoff_low]).type_as(mask_max)
    mask_max = mask_max.max(lowest)

    pseudo_gt = (mask > mask_max).type_as(mask)
    # remove ambiguous pixels, ambiguous = 1 means ignore
    ambiguous = (pseudo_gt.sum(1, keepdim=True) != 1).type_as(mask)

    pseudo_gt = pseudo_gt.argmax(dim=1, keepdim=True)
    pseudo_gt[ambiguous == 1] = -1

    return pseudo_gt.view(bs, h, w).cpu().numpy()


if __name__ == '__main__':
    seed_torch(3111)
    cfg = import_config('configs.ModelConfig')

    train(cfg)