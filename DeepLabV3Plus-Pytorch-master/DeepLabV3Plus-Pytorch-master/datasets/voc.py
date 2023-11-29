import os
import torch
import torchvision
from d2l import torch as d2l
import numpy as np


def read_postdam_images(voc_dir, is_train=True):
    """读取所有postdam图像并标注"""
    txt_fname = os.path.join(voc_dir, 'seg_train.txt' if is_train else 'seg_test.txt')
    # ImageReadMode: 读取模式,将JPEG或PNG图像读入三维RGB张量uint8(0,255)
    mode = torchvision.io.image.ImageReadMode.RGB
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    sum = 0
    for i, fname in enumerate(images):
        # features.append(torchvision.io.read_image(os.path.join(
        #     voc_dir, 'data', 'postdam', 'train' if is_train else 'test', 'img', f'{fname}')))
        tran = torchvision.io.read_image(os.path.join(voc_dir, 'train' if is_train else 'test', 'label_vis', f'{fname}'), mode)
        # 过滤标签，过滤为建筑物和背景
        a = np.where((tran[0]*1+tran[1]*2+tran[2]*10)==2550, 1, 0 )

        a0 = a * 0
        a1 = np.array([a0, a0, a])
        tran1 = tran * a1
        if (a.sum()>=256*256*0.05) & (a.sum()<=256*256*0.95):
            features.append(torchvision.io.read_image(os.path.join(voc_dir, 'train' if is_train else 'test', 'img', f'{fname}')))
            labels.append(tran1)
    return features, labels

#我们列举RGB颜色值和类名
#方便地查找标签中每个像素的类索引
VOC_COLORMAP = [[0, 0, 255], [0, 0, 0]]
VOC_CLASSES = ['Building','background']

#voc_colormap2label从RGB值到类别的索引
#voc_label_indices将RGB值映射到数据集中的类别索引
def voc_colormap2label():
    """构建从RGB到VOC类别索引的映射"""
    colormap2label = torch.zeros(256 ** 3, dtype=torch.long)
    for i, colormap in enumerate(VOC_COLORMAP):
        #构造256进制
        #很妙的构造，256进制转换为10进制
        #(255*256 + 255)*256 +255 + 1== 256**3
        colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    return colormap2label

def voc_label_indices(colormap, colormap2label):
    """将VOC标签中的RGB值映射到它们的类别索引"""
    #把RGB转换成类别索引，输出（1*256*256），每个像素为【0，21】
    colormap = colormap.permute(1, 2, 0).numpy().astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    return colormap2label[idx]

def voc_rand_crop(feature, label, height, width, is_train):
    """随机裁剪特征和标签图像"""
    #允许输入高宽进行裁剪返回其框（(i, j, h, w)）
    if is_train:
        rect = torchvision.transforms.RandomCrop.get_params(feature, (height, width))
        feature = torchvision.transforms.functional.crop(feature, *rect)
        label = torchvision.transforms.functional.crop(label, *rect)
    else:
        return feature, label
    return feature, label


# 自定义语义分割数据集类，继承高级API提供的Dataset类
class VOCSegDataset(torch.utils.data.Dataset):
    """一个用于加载VOC数据集的自定义数据集"""

    # crop_size给定模型训练时小批量图片里的高宽
    def __init__(self, is_train, crop_size, voc_dir):
        # ImageNet的标准化
        self.is_train = is_train
        self.transform = torchvision.transforms.Normalize(
            mean=[0.3365, 0.3597, 0.3331], std=[0.1405, 0.1378, 0.1432])
        self.crop_size = crop_size
        features, labels = read_postdam_images(voc_dir, self.is_train)
        self.features = [self.normalize_image(feature)
                         for feature in self.filter(features)]
        self.labels = self.filter(labels)
        self.colormap2label = voc_colormap2label()
        print('read ' + str(len(self.features)) + ' examples')

    def normalize_image(self, img):
        return self.transform(img.float() / 255)

    def filter(self, imgs):
        # 当图片小于crop_size,直接删除
        return [img for img in imgs if (img.shape[1] >= self.crop_size[0] and
                                        img.shape[2] >= self.crop_size[1])]

    def __getitem__(self, idx):
        # 通过实现__getitem__函数，我们可以任意访问数据集中索引为idx的输入图像
        # 及其每个像素的类别索引
        # 使用了数据增强
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx],
                                       *self.crop_size, self.is_train)
        # feature, label = self.features[idx], self.labels[idx]
        # label返回一张是【0，21】的像素图，通道为1
        return (feature, voc_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features)

def load_data_voc(batch_size, val_batch_size, crop_size):
    """加载VOC语义分割数据集"""
    voc_dir = r"E:\buildingdeeplab\dataset\postdam"
    num_workers = d2l.get_dataloader_workers()
    train_iter = torch.utils.data.DataLoader(
        VOCSegDataset(True, crop_size, voc_dir), batch_size,
        shuffle=True, drop_last=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(
        VOCSegDataset(False, crop_size, voc_dir), val_batch_size,
        drop_last=True, num_workers=num_workers)
    return train_iter, test_iter

def label2image(pred, device):
    #pred值为类别RGB的index
    colormap = torch.tensor(VOC_COLORMAP, device="cuda:0")
    X = pred.long()
    return colormap[X, :]
