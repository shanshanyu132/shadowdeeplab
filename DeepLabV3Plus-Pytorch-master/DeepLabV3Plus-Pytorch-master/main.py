from tqdm import tqdm
import network
import utils
import os
from d2l import torch as d2l
import random
import argparse
import numpy as np
from torch.nn import functional as F
import torch
import torch.nn as nn
from datasets import load_data_voc, label2image

# from torch.utils import data
# from utils import ext_transforms as et
# from metrics import StreamSegMetrics
# from utils.visualizer import Visualizer
# from PIL import Image
# import matplotlib
# import matplotlib.pyplot as plt


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", type=int, default=2,
                        help="num classes (default: None)")

    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
        network.modeling.__dict__[name]))
    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    #model index setting
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])
    parser.add_argument("--batch_size", type=int, default=128, help="train batch size")
    parser.add_argument("--val_batch_size", type=int, default=64, help="train batch size")
    parser.add_argument("--gpu_id", type=str, default='0', help="GPU ID")
    parser.add_argument("--num_epochs", type=int, default=100, help="Epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate (default: 0.01)")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')




    # Datset Options
    # parser.add_argument("--data_root", type=str, default='./datasets/data',
    #                     help="path to Dataset")
    # parser.add_argument("--dataset", type=str, default='voc',
    #                      help='Name of dataset')
    # Train Options
    # parser.add_argument("--test_only", action='store_true', default=False)
    # parser.add_argument("--save_val_results", action='store_true', default=False,
    #                     help="save segmentation results to \"./results\"")
    # parser.add_argument("--total_itrs", type=int, default=10e3,
    #                     help="epoch number (default: 30k)")
    # parser.add_argument("--lr", type=float, default=0.01,
    #                     help="learning rate (default: 0.01)")
    # parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
    #                     help="learning rate scheduler policy")
    # parser.add_argument("--step_size", type=int, default=10000)
    # parser.add_argument("--crop_val", action='store_true', default=False,
    #                     help='crop validation (default: False)')
    # parser.add_argument("--batch_size", type=int, default=128,
    #                     help='batch size (default: 16)')
    # parser.add_argument("--val_batch_size", type=int, default=64,
    #                     help='batch size for validation (default: 4)')
    # parser.add_argument("--crop_size", type=int, default=(150, 200))
    #
    # parser.add_argument("--ckpt", default=None, type=str,
    #                     help="restore from checkpoint")
    # parser.add_argument("--continue_training", action='store_true', default=False)
    #
    # parser.add_argument("--loss_type", type=str, default='cross_entropy',
    #                     choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    # parser.add_argument("--gpu_id", type=str, default='0',
    #                     help="GPU ID")
    # parser.add_argument("--weight_decay", type=float, default=1e-4,
    #                     help='weight decay (default: 1e-4)')
    # parser.add_argument("--random_seed", type=int, default=1,
    #                     help="random seed (default: 1)")
    # parser.add_argument("--print_interval", type=int, default=10,
    #                     help="print interval of loss (default: 10)")
    # parser.add_argument("--val_interval", type=int, default=100,
    #                     help="epoch interval for eval (default: 100)")
    # parser.add_argument("--download", action='store_true', default=False,
    #                     help="download datasets")
    #
    # # PASCAL VOC Options
    # parser.add_argument("--year", type=str, default='2012',
    #                     choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC')
    #
    # # Visdom options
    # parser.add_argument("--enable_vis", action='store_true', default=False,
    #                     help="use visdom for visualization")
    # parser.add_argument("--vis_port", type=str, default='18097',
    #                     help='port for visdom')
    # parser.add_argument("--vis_env", type=str, default='main',
    #                     help='env for visdom')
    # parser.add_argument("--vis_num_samples", type=int, default=8,
    #                     help='number of samples for visualization (default: 8)')
    return parser




def try_gpu(i=0):
    """如果存在， 则返回GPU（i),否则返回CPU()."""
    if torch.cuda.device_count() >= i+1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():
    """返回所有可用的GPU，如果没有GPU，则返回[CPU(),]。"""
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

def loss(inputs, targets):
    #在损失计算中需指定通道维，然后mean(1)先宽再高
    #这里在确定一下
    return F.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)

#定义一个函数，使用多GPU对模型进行训练和评估
def train_batch_ch13(net, X, y, loss, trainer, devices, scheduler):
    """用多GPU进行小批量训练"""
    X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    scheduler.step()
    train_loss_sum = l.sum()
    train_acc_sum = d2l.accuracy(pred, y)
    return train_loss_sum, train_acc_sum

def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices=try_all_gpus()):
    """用多GPU进行模型训练"""
    net.train()
    timer, num_batches = d2l.Timer(), len(train_iter)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, step_size=1000, gamma=0.8)
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        #四个维度：储存训练损失，训练准确度，实例数，特点数
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(net, features, labels, loss, trainer, devices, scheduler)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
                print("Epoch:%d, iter:%d, loss:%.3f, train_acc:%.3f, test_acc:%.3f" % (epoch, epoch+(i + 1)/num_batches, (metric[0] / metric[2]), (metric[1] / metric[3]), test_acc))
                with open("metric.txt", 'a+') as f:
                    f.write("Epoch:%d, iter:%d, loss:%.3f, train_acc:%.3f, test_acc:%.3f" % (epoch, epoch+(i + 1)/num_batches, (metric[0] / metric[2]), (metric[1] / metric[3]), test_acc) + '\n')
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')

def save_ckpt(path, cur_itrs, model, optimizer, scheduler, best_score):
    """ save current model
    """
    torch.save({
        "cur_itrs": cur_itrs,
        "model_state": model.module.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_score": best_score,
    }, path)
    print("Model saved as %s" % path)



# def validate(opts, model, loader, device, metrics, ret_samples_ids=None, test_dst):
#     """验证集可视化及保存
#     Do validation and return specified samples"""
#     metrics.reset()
#     ret_samples = []
#     if opts.save_val_results:
#         if not os.path.exists('results'):
#             os.mkdir('results')
#         #反标准化
#         denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
#                                    std=[0.229, 0.224, 0.225])
#         img_id = 0
#
#     with torch.no_grad():
#         #测试时不需要存梯度
#         for i, (images, labels) in tqdm(enumerate(loader)):
#
#             images = images.to(device, dtype=torch.float32)
#             labels = labels.to(device, dtype=torch.long)
#
#             X = test_dst.dataset.normalize_image(img).unsqueeze(0)
#             outputs = label2image(model(images), device=try_gpu())
#             preds = outputs.detach().max(dim=1)[1].cpu().numpy()
#             targets = labels.cpu().numpy()
#
#             metrics.update(targets, preds)
#             #可视化例子
#             if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
#                 ret_samples.append(
#                     (images[0].detach().cpu().numpy(), targets[0], preds[0]))
#
#            #保存验证结果
#             if opts.save_val_results:
#                 for i in range(len(images)):
#                     image = images[i].detach().cpu().numpy()
#                     target = targets[i]
#                     pred = preds[i]
#
#                     image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
#                     target = loader.dataset.decode_target(target).astype(np.uint8)
#                     pred = loader.dataset.decode_target(pred).astype(np.uint8)
#
#                     Image.fromarray(image).save('results/%d_image.png' % img_id)
#                     Image.fromarray(target).save('results/%d_target.png' % img_id)
#                     Image.fromarray(pred).save('results/%d_pred.png' % img_id)
#
#                     fig = plt.figure()
#                     plt.imshow(image)
#                     plt.axis('off')
#                     plt.imshow(pred, alpha=0.7)
#                     ax = plt.gca()
#                     ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
#                     ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
#                     plt.savefig('results/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
#                     plt.close()
#                     img_id += 1
#
#         score = metrics.get_results()
#     return score, ret_samples


def main():
    opts = get_argparser().parse_args()
    device = try_gpu(i=0)
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id

    crop_size = (250, 250)
    train_iter, test_iter = load_data_voc(opts.batch_size, opts.val_batch_size, crop_size)
    print("Dataset: %s, Train set: %d, Val set: %d" % ("postdam", len(train_iter), len(test_iter)))

    # Setup random seed
    # torch.manual_seed(opts.random_seed)
    # np.random.seed(opts.random_seed)
    # random.seed(opts.random_seed)

    # Set up model (all models are 'constructed at network.modeling)
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    model.to(device)

    # Set up optimizer
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1 * opts.lr},
        {'params': model.classifier.parameters(), 'lr': opts.lr},
    ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)

    devices = try_all_gpus()
    train_ch13(model, train_iter, test_iter, loss, optimizer, opts.num_epochs, devices)
    torch.save(model, "checkpoints/latest_1.pth")


    # Set up metrics
    # metrics = StreamSegMetrics(opts.num_classes)
    # utils.mkdir('checkpoints')
    # Restore
    # best_score = 0.0
    # cur_itrs = 0
    # cur_epochs = 0
    # interval_loss = 0
    # while True:  # cur_itrs < opts.total_itrs:
    #     # 训练函数
    #     model.train()
    #     cur_epochs += 1
    #     for (images, labels) in train_dst:
    #         cur_itrs += 1
    #
    #         images = images.to(device, dtype=torch.float32)
    #         labels = labels.to(device, dtype=torch.long)
    #
    #         optimizer.zero_grad()
    #         outputs = model(images)
    #         ls = loss(outputs, labels)
    #         ls.sum().backward()
    #         optimizer.step()
    #
    #         np_loss = ls.detach().cpu().numpy()
    #         interval_loss += np_loss
    #         if vis is not None:
    #             vis.vis_scalar('Loss', cur_itrs, np.mean(np_loss))
    #
    #         if (cur_itrs) % 10 == 0:
    #             interval_loss = interval_loss / 10
    #             print("Epoch %d, Itrs %d/%d, Loss=%f" %
    #                   (cur_epochs, cur_itrs, opts.total_itrs, np.mean(interval_loss)))
    #             interval_loss = 0.0
    #
    #         if (cur_itrs) % opts.val_interval == 0:
    #             save_ckpt('checkpoints/latest_%s_%s_os%d.pth' %
    #                       (opts.model, opts.dataset, opts.output_stride))
    #             print("validation...")
    #             model.eval()
    #             val_score, ret_samples = validate(
    #                 opts=opts, model=model, loader=test_dst, device=device, metrics=metrics,
    #                 ret_samples_ids=vis_sample_id)
    #             print(metrics.to_str(val_score))
    #             if val_score['Mean IoU'] > best_score:  # save best model
    #                 best_score = val_score['Mean IoU']
    #                 save_ckpt('checkpoints/best_%s_%s_os%d.pth' %
    #                           (opts.model, opts.dataset, opts.output_stride))
    #
    #             if vis is not None:  # visualize validation score and samples
    #                 vis.vis_scalar("[Val] Overall Acc", cur_itrs, val_score['Overall Acc'])
    #                 vis.vis_scalar("[Val] Mean IoU", cur_itrs, val_score['Mean IoU'])
    #                 vis.vis_table("[Val] Class IoU", val_score['Class IoU'])
    #
    #                 for k, (img, target, lbl) in enumerate(ret_samples):
    #                     img = (denorm(img) * 255).astype(np.uint8)
    #                     target = train_dst.decode_target(target).transpose(2, 0, 1).astype(np.uint8)
    #                     lbl = train_dst.decode_target(lbl).transpose(2, 0, 1).astype(np.uint8)
    #                     concat_img = np.concatenate((img, target, lbl), axis=2)  # concat along width
    #                     vis.vis_image('Sample %d' % k, concat_img)
    #             model.train()
    #         scheduler.step()
    #
    #         if cur_itrs >= opts.total_itrs:
    #             return

    # if opts.dataset == 'voc' and not opts.crop_val:
    #     opts.val_batch_size = 128
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # devices = try_all_gpus()
    # print("Device: %s" % device)
    # Setup dataloader
    # print("Dataset: %s, Train set: %d, Val set: %d" % (opts.dataset, len(train_dst), len(test_dst)))
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)
    # if opts.lr_policy == 'poly':
    #     scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    # elif opts.lr_policy == 'step':
    #     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)
    # Set up criterion
    # criterion = utils.get_loss(opts.loss_type)
    # if opts.loss_type == 'focal_loss':
    #     criterion = utils.FocalLoss(size_average=True)
    # elif opts.loss_type == 'cross_entropy':
    #       criterion = nn.CrossEntropyLoss(reduction='mean')
    # utils.mkdir('checkpoints')
    # # Restore
    # best_score = 0.0
    # cur_itrs = 0
    # cur_epochs = 0
    # #读取预训练模型判断某一对象是否是文件
    # if opts.ckpt is not None and os.path.isfile(opts.ckpt):
    #     # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
    #     checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
    #     model.load_state_dict(checkpoint["model_state"])
    #     model = nn.DataParallel(model)
    #     model.to(device)
    #     if opts.continue_training:
    #         optimizer.load_state_dict(checkpoint["optimizer_state"])
    #         scheduler.load_state_dict(checkpoint["scheduler_state"])
    #         cur_itrs = checkpoint["cur_itrs"]
    #         best_score = checkpoint['best_score']
    #         print("Training state restored from %s" % opts.ckpt)
    #     print("Model restored from %s" % opts.ckpt)
    #     del checkpoint  # free memory
    # else:
    #     print("[!] Retrain")
    #     model = nn.DataParallel(model)
    #     model.to(device)
    # # ==========   Train Loop   ==========#
    # vis_sample_id = np.random.randint(0, len(test_dst), opts.vis_num_samples,
    #                                   np.int32) if opts.enable_vis else None  # sample idxs for visualization
    # denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images
    # if opts.test_only:
    #     model.eval()
    #     val_score, ret_samples = validate(
    #         opts=opts, model=model, loader=test_dst, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
    #     print(metrics.to_str(val_score))
    #     return

if __name__ == '__main__':
    main()
