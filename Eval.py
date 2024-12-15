import ever as er
import logging
from tqdm import tqdm
from ever.util.param_util import count_model_parameters
from utils.DataLoader import TrainLoader
from utils.Tools import *
from module.viz import VisualizeSegmm
from module.MyModel import Deeplabv2
from module.Clip import pre_slide


def evaluate(model, cfg, is_training=False, ckpt_path=None, logger=None):
    if cfg.SNAPSHOT_DIR is not None:
        vis_dir = os.path.join(cfg.SNAPSHOT_DIR, 'vis-{}'.format(os.path.basename(ckpt_path)))
        palette = np.asarray(list(COLOR_MAP.values())).reshape((-1,)).tolist()
        viz_op = VisualizeSegmm(vis_dir+'_EVAL', palette)

    if not is_training:
        model_state_dict = torch.load(ckpt_path)
        model.load_state_dict(model_state_dict, strict=True)

        logger.info('[Load params] from {}'.format(ckpt_path))
        count_model_parameters(model, logger)

    model.eval()

    eval_dataloader = TrainLoader(cfg.EVAL_DATA_CONFIG)
    logger.info(cfg.EVAL_DATA_CONFIG)

    metric_op = er.metric.PixelMetric(len(COLOR_MAP.keys()), logdir=cfg.SNAPSHOT_DIR, logger=logger)

    with torch.no_grad():
        for ret, ret_gt in tqdm(eval_dataloader):
            ret = ret.to(torch.device('cuda'))

            # cls = model(ret)
            cls = pre_slide(model, ret, tta=False)

            cls = cls.argmax(dim=1).cpu().numpy()

            cls_gt = ret_gt['cls'].cpu().numpy().astype(np.int32)
            mask = cls_gt >= 0

            y_true = cls_gt[mask].ravel()
            y_pred = cls[mask].ravel()
            metric_op.forward(y_true, y_pred)

            if cfg.SNAPSHOT_DIR is not None:
                for fname, pred in zip(ret_gt['fname'], cls):
                    viz_op(pred, fname.replace('tif', 'png'))

    metric_op.summary_all()

    torch.cuda.empty_cache()


if __name__ == '__main__':
    seed_torch(3111)
    ckpt_path = '/root/autodl-tmp/SCA/log/train/LoveDA2Urban/LoveDA2Urban_N210000.pth'

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
            use_aux=False,
        ),
        inchannels=2048,
        num_classes=7
    )).cuda()

    cfg = import_config('configs.ModelConfig')
    logger = get_console_file_logger(name='311', logdir=cfg.SNAPSHOT_DIR)

    evaluate(model, cfg, False, ckpt_path, logger)
