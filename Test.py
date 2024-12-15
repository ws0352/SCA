import cv2
from tqdm import tqdm
from utils.DataLoader import TestLoader
from utils.Tools import *
from module.viz import VisualizeSegmm
from module.MyModel import Deeplabv2
from module.Clip import pre_slide

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def test(model, cfg, ckpt_path=None):

    if cfg.SNAPSHOT_DIR is not None:
        vis_dir = os.path.join(cfg.SNAPSHOT_DIR, 'vis-{}'.format(os.path.basename(ckpt_path)))
        palette = np.asarray(list(COLOR_MAP.values())).reshape((-1,)).tolist()
        viz_op = VisualizeSegmm(vis_dir+'_TEST', palette)

    if not os.path.exists(cfg.TEST_SAVE_PATH):
        os.makedirs(cfg.TEST_SAVE_PATH)

    model_state_dict = torch.load(ckpt_path)
    model.load_state_dict(model_state_dict, strict=True)

    model.eval()

    test_dataloader = TestLoader(cfg.TEST_DATA_CONFIG)
    print(cfg.TEST_DATA_CONFIG)

    with torch.no_grad():
        for ret in tqdm(test_dataloader):
            rgb = ret['rgb'].to(torch.device('cuda'))

            rgb = rgb[:, :3, :, :]

            cls = pre_slide(model, rgb, num_classes=7, tile_size=(512, 512), tta=True)
            cls = cls.argmax(dim=1).cpu().numpy()

            cv2.imwrite(cfg.TEST_SAVE_PATH + '/' + ret['fname'][0], cls.reshape(1024, 1024).astype(np.uint8))

            for fname, pred in zip(ret['fname'], cls):
                viz_op(pred, fname.replace('tif', 'png'))

    torch.cuda.empty_cache()


if __name__ == '__main__':
    seed_torch(3111)
    ckpt_path = '/root/autodl-tmp/SCA/log/train/LoveDA2Urban/LoveDA2Urban_S4010000.pth'

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
            fc_dim=2048,
        ),
        inchannels=2048,
        num_classes=7
    )).cuda()

    cfg = import_config('configs.ModelConfig')

    test(model, cfg, ckpt_path)

