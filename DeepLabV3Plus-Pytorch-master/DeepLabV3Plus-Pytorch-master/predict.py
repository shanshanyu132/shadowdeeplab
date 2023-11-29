from d2l import torch as d2l
from datasets import load_data_voc
import os
import numpy as np
import torchvision
import torch
from PIL import Image

# import torch.nn as nn
# from PIL import Image
# import matplotlib
# import matplotlib.pyplot as plt
# from glob import glob
# from torch.utils.data import dataset
# from tqdm import tqdm
# import network
# import utils
# import random
# import argparse
# # from torch.nn import Functional
# from torch.utils import data
# # from datasets import VOCSegmentation, Cityscapes, cityscapes
# # from torchvision import transforms as T
# from metrics import StreamSegMetrics

# def get_argparser():
#     parser = argparse.ArgumentParser()
#
#     # Datset Options
#     parser.add_argument("--input", type=str, required=True,
#                         help="path to a single image or image directory")
#     parser.add_argument("--dataset", type=str, default='voc',
#                         choices=['voc', 'cityscapes'], help='Name of training set')
#
#     # Deeplab Options
#     available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
#                               not (name.startswith("__") or name.startswith('_')) and callable(
#                               network.modeling.__dict__[name])
#                               )
#
#     parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
#                         choices=available_models, help='model name')
#     parser.add_argument("--separable_conv", action='store_true', default=False,
#                         help="apply separable conv to decoder and aspp")
#     parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])
#
#     # Train Options
#     parser.add_argument("--save_val_results_to", default=None,
#                         help="save segmentation results to the specified dir")
#
#     parser.add_argument("--crop_val", action='store_true', default=False,
#                         help='crop validation (default: False)')
#     parser.add_argument("--val_batch_size", type=int, default=4,
#                         help='batch size for validation (default: 4)')
#     parser.add_argument("--crop_size", type=int, default=513)
#
#
#     parser.add_argument("--ckpt", default=None, type=str,
#                         help="resume from checkpoint")
#     parser.add_argument("--gpu_id", type=str, default='0',
#                         help="GPU ID")
#     return parser

def read_postdamval_images(voc_dir):
    """读取所有postdam图像并标注"""
    txt_fname = os.path.join(voc_dir, 'seg_val.txt')
    # ImageReadMode: 读取模式,将JPEG或PNG图像读入三维RGB张量uint8(0,255)
    mode = torchvision.io.image.ImageReadMode.RGB
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    sum = 0
    for i, fname in enumerate(images):
        # features.append(torchvision.io.read_image(os.path.join(
        #     voc_dir, 'data', 'postdam', 'train' if is_train else 'test', 'img', f'{fname}')))
        tran = torchvision.io.read_image(os.path.join(voc_dir, 'val', 'label_vis', f'{fname}'), mode)
        # 过滤标签，过滤为建筑物和背景
        a = np.where((tran[0]*1+tran[1]*2+tran[2]*10)==2550, 1, 0 )

        a0 = a * 0
        a1 = np.array([a0, a0, a])
        tran1 = tran * a1
        if (a.sum()>=256*256*0.1) & (a.sum()<=256*256*0.8):
            features.append(torchvision.io.read_image(os.path.join(voc_dir, 'val', 'img', f'{fname}')))
            labels.append(tran1)
    return features, labels





VOC_COLORMAP = [[0, 0, 255], [0, 0, 0]]
VOC_CLASSES = ['Building','background']





def main():
    # opts = get_argparser().parse_args()
    model = torch.load("checkpoints/latest_1.pth")

    devices = d2l.try_all_gpus()
    crop_size = (250, 250)
    train_iter, test_iter = load_data_voc(128, 64, crop_size)

    def label2image(pred):
        # pred值为类别RGB的index
        VOC_COLORMAP = [[0, 0, 255], [0, 0, 0]]
        colormap = torch.tensor(VOC_COLORMAP, device=devices[0])
        X = pred.long()
        return colormap[X, :]

    def predict(img):
        X = test_iter.dataset.normalize_image(img).unsqueeze(0)
        # 预测在通道维
        pred = model(X.to(devices[0])).argmax(dim=1)
        return pred.reshape(pred.shape[1], pred.shape[2])

    # 测试数据集中的图像大小和形状各异
    # 步幅为32的转置卷积核，需要考虑是否能被整除
    # 读取几张较大的测试图像，并从图像的左上角开始截取形状为320*480的区域用于预测
    voc_dir = r"E:\buildingdeeplab\dataset\postdam"
    test_images, test_labels = read_postdamval_images(voc_dir)
    n, imgs = 10, []
    for i in range(n):
        crop_rect = (0, 0, 250, 250)
        X = torchvision.transforms.functional.crop(test_images[i], *crop_rect)
        pred = label2image(predict(X))
        Ximg = np.uint8(np.array(X.detach().permute(1,2,0)))
        predimg = np.uint8(np.array(pred.cpu()))
        labimg = np.uint8(np.array((torchvision.transforms.functional.crop(
                     test_labels[i], *crop_rect).permute(1,2,0))))
        iXimg, ipredimg, ilabimg = Image.fromarray(Ximg), Image.fromarray(predimg), Image.fromarray(labimg)
        iXimg.save(r"samples/x_%d.jpg" % (i))
        ipredimg.save(r"samples/pred_%d.jpg" % (i))
        ilabimg.save(r"samples/lab_%d.jpg" % (i))











    # if opts.dataset.lower() == 'voc':
    #     opts.num_classes = 21
    #     decode_fn = VOCSegmentation.decode_target
    # elif opts.dataset.lower() == 'cityscapes':
    #     opts.num_classes = 19
    #     decode_fn = Cityscapes.decode_target

    # os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print("Device: %s" % device)

    # Setup dataloader
    # image_files = []
    # #判断某一路径是否为目录
    # if os.path.isdir(opts.input):
    #     for ext in ['png', 'jpeg', 'jpg', 'JPEG']:
    #         #glob文件操作相关模块，可以查找符合自己目的的文件，类似与windows下的文件搜索
    #         files = glob(os.path.join(opts.input, '**/*.%s'%(ext)), recursive=True)
    #         if len(files)>0:
    #             #extend用于在列表末尾一次性追加另一个序列中的多个值
    #             image_files.extend(files)
    # #判断某一对象是否为文件
    # elif os.path.isfile(opts.input):
    #     image_files.append(opts.input)
    
    # # Set up model (all models are 'constructed at network.modeling)
    # model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    # if opts.separable_conv and 'plus' in opts.model:
    #     network.convert_to_separable_conv(model.classifier)
    # utils.set_bn_momentum(model.backbone, momentum=0.01)
    
    # if opts.ckpt is not None and os.path.isfile(opts.ckpt):
    #     # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
    #     checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
    #     model.load_state_dict(checkpoint["model_state"])
    #     model = nn.DataParallel(model)
    #     model.to(device)
    #     print("Resume model from %s" % opts.ckpt)
    #     del checkpoint
    # else:
    #     print("[!] Retrain")
    #     model = nn.DataParallel(model)
    #     model.to(device)

    #denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    # if opts.crop_val:
    #     transform = T.Compose([
    #             T.Resize(opts.crop_size),
    #             T.CenterCrop(opts.crop_size),
    #             T.ToTensor(),
    #             T.Normalize(mean=[0.485, 0.456, 0.406],
    #                             std=[0.229, 0.224, 0.225]),
    #         ])
    # else:
    #     transform = T.Compose([
    #             T.ToTensor(),
    #             T.Normalize(mean=[0.485, 0.456, 0.406],
    #                             std=[0.229, 0.224, 0.225]),
    #         ])
    # if opts.save_val_results_to is not None:
    #     os.makedirs(opts.save_val_results_to, exist_ok=True)
    # with torch.no_grad():
    #     model = model.eval()
    #     for img_path in tqdm(image_files):
    #         #得到文件后缀
    #         ext = os.path.basename(img_path).split('.')[-1]
    #         img_name = os.path.basename(img_path)[:-len(ext)-1]
    #         img = Image.open(img_path).convert('RGB')
    #         img = transform(img).unsqueeze(0) # To tensor of NCHW
    #         img = img.to(device)
    #
    #         pred = model(img).max(1)[1].cpu().numpy()[0] # HW
    #         colorized_preds = decode_fn(pred).astype('uint8')
    #         colorized_preds = Image.fromarray(colorized_preds)
    #         if opts.save_val_results_to:
    #             colorized_preds.save(os.path.join(opts.save_val_results_to, img_name+'.png'))

if __name__ == '__main__':
    main()
