# -*- coding:utf-8 -*-
# @Time   : 2022/4/30 15:22
# @Author : quzhuo
import os
import random
import time
import scipy
import numpy as np
from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as ut
import torchvision.transforms as tr


# ---------------------------------------------------------#
#   单步训练
# ---------------------------------------------------------#
from Utils.prune_utils import BNOptimizer, get_sr_flag

# def updateBN(model,args):
#     for m in model.modules():
#         #  isinstance() 函数来判断一个对象是否是一个已知的类型
#         if isinstance(m, nn.BatchNorm2d):
#             #  hasattr() 函数用于判断对象是否包含对应的属性
#             if hasattr(m.weight, 'data'):
#                 if m.weight.grad is None:
#                     continue
#                 else:
#                     m.weight.grad.data.add_(args.s*torch.sign(m.weight.data))  # L1正则


def fit_one_epoch(model, optimizer, scheduler, criterion, epoch, train_loader, val_loader, train_lines, val_lines, args, best_record):
    lr = args.lr

    train_main_loss = AverageMeter()

    epoch_step_train = len(train_lines) // args.batch_size
    # epoch_step_test = len(val_lines) // args.batch_size

    model.train()
    print('-----Start Train-----')
    start_time = time.time()

    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        N = inputs.size(0) * inputs.size(2) * inputs.size(3)    # batch_size * input_size

        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = outputs[0]

        main_loss = criterion(outputs, targets)
        loss = main_loss
        loss.backward()

        # updateBN(model, args)

        optimizer.step()

        train_main_loss.update(main_loss.item(), N)

        if (i + 1) % args.print_freq == 0:
            process_time = time.time() - start_time
            print('[epoch %d], [iter %d / %d], [loss %.5f], [lr %.10f], [time %.3f]' % (epoch + 1, i + 1, epoch_step_train, train_main_loss.avg, optimizer.param_groups[0]['lr'], process_time))

            start_time = time.time()

    validate(model, epoch, val_loader, optimizer, scheduler, criterion, args, best_record, start_time)


# ---------------------------------------------------------#
#   保存预测结果(图片)
# ---------------------------------------------------------#
def fit_one_test(model, optimizer, scheduler, criterion, test_loader, test_lines, args, best_record):
    start_time = time.time()
    validate(model, 0, test_loader, optimizer, scheduler, criterion, args, best_record, start_time, is_test=True)
    # if args.save_results:
    #     print(1)


# ---------------------------------------------------------#
#   保存预测结果(图片)
# ---------------------------------------------------------#
def validate(model, epoch, val_loader, optimizer, scheduler, criterion, args, best_record, start_time, is_test=False):
    model.eval()

    print('-----Start Validate-----')

    val_loss = AverageMeter()

    inputs_all, targets_all, predictions_all = [], [], []

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            N = inputs.size(0) * inputs.size(2) * inputs.size(3)

            outputs = model(inputs)
            outputs = outputs[0]

            predictions = outputs.data.max(1)[1].squeeze(1).squeeze(0).cpu().numpy()
            val_loss.update(criterion(outputs, targets).item(), N)

            if random.random() > args.save_rate:
                inputs_all.append(None)
            else:
                inputs_all.append(inputs.data.squeeze(0).cpu())
            targets_all.append(targets.data.squeeze(0).cpu().numpy())
            predictions_all.append(predictions)

    update_ckpt(model, epoch, optimizer, val_loss, inputs_all, targets_all, predictions_all, args, best_record, start_time, is_test)

    model.train()
    return val_loss, inputs_all, targets_all, predictions_all


# ---------------------------------------------------------#
#   模型评价指标保存与可视化(tensorboard)
# ---------------------------------------------------------#
def update_ckpt(model, epoch, optimizer, val_loss, inputs_all, targets_all, predictions_all, args, best_record, start_time, is_test=False):
    avg_loss = val_loss.avg

    acc, acc_cls, mean_iu, fwavacc, f1, con_matrix_n = evaluate(predictions_all, targets_all, args.num_classes)

    # loss_history.append_loss(val_loss)
    # acc_history.append_acc(acc)

    with open(os.path.join(args.save_path, 'logs.txt'), 'a') as f:
        f.write('[epoch %d]:   [val_loss %.5f]  [lr %.5f]  [acc %.5f]  [mean_iu %.5f]  [f1_score %.5f]  [time %.5f]\n' % (epoch + 1, avg_loss, optimizer.param_groups[0]['lr'], acc, mean_iu, f1, time.time()-start_time))
        f.close()

    updated = update_best_record(best_record, epoch, avg_loss, acc, acc_cls, mean_iu, fwavacc, f1, con_matrix_n, args)

    if not is_test:
        # last.pth
        check_mkdir(args.save_path)
        if os.path.exists(os.path.join(args.save_path, 'last.pth')):
            os.remove(os.path.join(args.save_path, 'last.pth'))
        torch.save(model.state_dict(), os.path.join(args.save_path, 'last.pth'))

        # ckpt_last.pth
        ckpt_path = os.path.join(args.save_path, 'checkpoint')
        check_mkdir(ckpt_path)
        if os.path.exists(os.path.join(args.save_path, 'ckpt_last.pth')):
            os.remove(os.path.join(args.save_path, 'ckpt_last.pth'))
        checkpoint = {'net': model.state_dict(),
                      'optimizer': optimizer.state_dict(),
                      'epoch': epoch}
        torch.save(checkpoint, os.path.join(ckpt_path, 'ckpt_last.pth'))

        # best.pth
        if updated or (best_record['val_loss'] > avg_loss):
            check_mkdir(args.save_path)
            if os.path.exists(os.path.join(args.save_path, 'best.pth')):
                os.remove(os.path.join(args.save_path, 'best.pth'))
            torch.save(model.state_dict(), os.path.join(args.save_path, 'best.pth'))

            # ckpt_best.pth
            if os.path.exists(os.path.join(args.save_path, 'ckpt_best.pth')):
                os.remove(os.path.join(args.save_path, 'ckpt_best.pth'))
            checkpoint = {'net': model.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'epoch': epoch}
            torch.save(checkpoint, os.path.join(ckpt_path, 'ckpt_best.pth'))

    val_visual = visual_ckpt(epoch, inputs_all, targets_all, predictions_all, args, is_test)

    # if len(val_visual) > 0:
    #     val_visual = torch.stack(val_visual, 0)
    #     val_visual = ut.make_grid(val_visual, nrow=3, padding=5)
    #     writer.add_image(snapshot_name, val_visual)


# ---------------------------------------------------------#
#   .......
# ---------------------------------------------------------#
def visual_ckpt(epoch, inputs_all, targets_all, predictions_all, args, is_test=False):
    palette = args.palette
    val_visual = []

    to_save_dir = os.path.join(args.save_path, str(epoch + 1))
    check_mkdir(to_save_dir)

    visualize, restore = get_visualize()

    for idx, data in enumerate(zip(inputs_all, targets_all, predictions_all)):
        if data[0] is None:
            continue

        if args.batch_size == 1:
            input_pil = restore(data[0][0:3, :, :])
            target_pil = colorize_mask(data[1], palette)
            prediction_pil = colorize_mask(data[2], palette)
        else:
            input_pil = restore(data[0][0][0:3, :, :])
            target_pil = colorize_mask(data[1][0], palette)
            prediction_pil = colorize_mask(data[2][0], palette)

        if not is_test:
            input_pil.save(os.path.join(to_save_dir, '%d_input.png' % idx))
            target_pil.save(os.path.join(to_save_dir, '%d_target.png' % idx))
            prediction_pil.save(os.path.join(to_save_dir, '%d_prediction.png' % idx))

            val_visual.extend([visualize(input_pil.convert('RGB')),
                               visualize(target_pil.convert('RGB')),
                               visualize(prediction_pil.convert('RGB'))])
        else:
            check_mkdir(os.path.join(args.save_path, 'Validate'))
            input_pil.save(os.path.join(os.path.join(args.save_path, 'Validate'), '%d_input.png' % idx))
            target_pil.save(os.path.join(os.path.join(args.save_path, 'Validate'), '%d_target.png' % idx))
            prediction_pil.save(os.path.join(os.path.join(args.save_path, 'Validate'), '%d_prediction.png' % idx))

            val_visual.extend([visualize(input_pil.convert('RGB')),
                               visualize(target_pil.convert('RGB')),
                               visualize(prediction_pil.convert('RGB'))])

    return val_visual


def get_visualize():
    visualize = tr.Compose([
        tr.Resize(300),
        tr.CenterCrop(300),
        tr.ToTensor()
    ])

    restore = tr.Compose([
        tr.ToPILImage()
    ])

    return visualize, restore


def setup_palette(palette):
    palette_rgb = []
    for _, color in palette.items():
        palette_rgb += color

    zero_pad = 256 * 3 - len(palette_rgb)

    for i in range(zero_pad):
        palette_rgb.append(0)

    return palette_rgb


def colorize_mask(mask, palette):
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(setup_palette(palette))

    return new_mask


# ---------------------------------------------------------#
#   模型评价指标计算
# ---------------------------------------------------------#
def evaluate(predictions, targets, num_classes):
    con_matrix = np.zeros((num_classes, num_classes))

    labels = np.arange(num_classes).tolist()

    for lp, lt in zip(predictions, targets):
        con_matrix += confusion_matrix(lt.flatten(), lp.flatten(), labels=labels)

    M, N = con_matrix.shape
    tp = np.zeros(M, dtype=np.uint)
    fp = np.zeros(M, dtype=np.uint)
    fn = np.zeros(M, dtype=np.uint)

    # -------------------------
    print('------Confusion Matrix------')
    # print(con_matrix)

    for i in range(M):
        tp[i] = con_matrix[i, i]
        fp[i] = np.sum(con_matrix[:, i]) - tp[i]
        fn[i] = np.sum(con_matrix[i, :]) - tp[i]

    # 混淆矩阵归一化
    con_matrix_n = np.zeros((num_classes, num_classes))
    count = np.zeros(num_classes)
    for i in range(M):
        for j in range(N):
            count[i] += con_matrix[i][j]
        for j in range(N):
            if count[i] == 0:
                con_matrix_n[i][j] = 0
            else:
                con_matrix_n[i][j] = con_matrix[i][j] / count[i]
    print(con_matrix_n)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * recall * precision / (recall + precision)

    ax_p = 0
    acc = np.diag(con_matrix).sum() / con_matrix.sum()
    acc_cls = np.diag(con_matrix) / con_matrix.sum(axis=ax_p)
    acc_cls = np.nanmean(acc_cls)
    iu = tp / (tp + fp + fn)
    mean_iu = np.nanmean(iu)
    freq = con_matrix.sum(axis=ax_p) / con_matrix.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    return acc, acc_cls, mean_iu, fwavacc, np.nanmean(f1_score), con_matrix_n


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# ---------------------------------------------------------#
#   最佳参数保存
# ---------------------------------------------------------#
def print_best_record(best_record):
    print('[best %d]: [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f], [f1 %.5f]' %
          (best_record['epoch'],
           best_record['val_loss'],
           best_record['acc'],
           best_record['acc_cls'],
           best_record['mean_iu'],
           best_record['fwavacc'],
           best_record['f1'])
          )


def update_best_record(best_record, epoch, val_loss, acc, acc_cls, mean_iu, fwavacc, f1, con_matrix_n, args):
    print('----------------------------------------------------------------------------------------')
    print('[epoch %d]: [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f], [f1 %.5f]' %
          (epoch + 1, val_loss, acc, acc_cls, mean_iu, fwavacc, f1))
    print_best_record(best_record)
    print('----------------------------------------------------------------------------------------')

    if args.update_mode == 0:
        if mean_iu > best_record['mean_iu']:
            best_record['epoch'] = epoch + 1
            best_record['val_loss'] = val_loss
            best_record['acc'] = acc
            best_record['acc_cls'] = acc_cls
            best_record['mean_iu'] = mean_iu
            best_record['fwavacc'] = fwavacc
            best_record['f1'] = f1

            plot_confusion_matrix(cm=con_matrix_n, classes=args.features, save_path=args.save_path)
            return True
        else:
            return False
    elif args.update_mode == 1:
        if f1 > best_record['f1']:
            best_record['epoch'] = epoch + 1
            best_record['val_loss'] = val_loss
            best_record['acc'] = acc
            best_record['acc_cls'] = acc_cls
            best_record['mean_iu'] = mean_iu
            best_record['fwavacc'] = fwavacc
            best_record['f1'] = f1

            plot_confusion_matrix(cm=con_matrix_n, classes=args.features, save_path=args.save_path)
            return True
        else:
            return False
    elif args.update_mode == 2:
        if mean_iu > best_record['mean_iu'] or f1 > best_record['f1']:
            best_record['epoch'] = epoch + 1
            best_record['val_loss'] = val_loss
            best_record['acc'] = acc
            best_record['acc_cls'] = acc_cls
            best_record['mean_iu'] = mean_iu
            best_record['fwavacc'] = fwavacc
            best_record['f1'] = f1

            plot_confusion_matrix(cm=con_matrix_n, classes=args.features, save_path=args.save_path)
            return True
        else:
            return False


# ---------------------------------------------------------#
#   分类混淆矩阵绘制
# ---------------------------------------------------------#
def plot_confusion_matrix(cm, classes, save_path, title='Confusion Matrix'):
    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    check_mkdir(save_path)
    plt.savefig(os.path.join(save_path, 'Confusion_Matrix.png'), format='png')


# ---------------------------------------------------------#
#   模型评价指标记录与可视化(自定义)
# ---------------------------------------------------------#
class LossHistory():
    def __init__(self, log_dir, val_loss_flag=True):
        import datetime
        self.time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        self.save_path = os.path.join(log_dir, "loss_" + str(self.time_str))
        self.val_loss_flag = val_loss_flag

        self.losses = []
        if self.val_loss_flag:
            self.val_loss = []

        os.makedirs(self.save_path)

    def append_loss(self, loss, val_loss=0):
        self.losses.append(loss)
        with open(os.path.join(self.save_path, "epoch_loss_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")

        if self.val_loss_flag:
            self.val_loss.append(val_loss)
            with open(os.path.join(self.save_path, "epoch_val_loss_" + str(self.time_str) + ".txt"), 'a') as f:
                f.write(str(val_loss))
                f.write("\n")
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()

        plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')
        try:
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, 5 if len(self.losses) < 25 else 15, 3), 'green',
                     linestyle='--', linewidth=2, label='smooth train loss')
        except:
            pass

        if self.val_loss_flag:
            plt.plot(iters, self.val_loss, 'coral', linewidth=2, label='val loss')
            try:
                plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, 5 if len(self.losses) < 25 else 15, 3),
                         '#8B4513', linestyle='--', linewidth=2, label='smooth val loss')
            except:
                pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.save_path, "epoch_loss_" + str(self.time_str) + ".png"))

        plt.cla()
        plt.close("all")


class AccHistory():
    def __init__(self, log_dir, val_acc_flag=True):
        import datetime
        self.time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        self.save_path = os.path.join(log_dir, "acc_" + str(self.time_str))
        self.val_acc_flag = val_acc_flag

        self.acces = []
        if self.val_acc_flag:
            self.val_acc = []

        os.makedirs(self.save_path)

    def append_acc(self, acc, val_acc=0):
        self.acces.append(acc)
        with open(os.path.join(self.save_path, "epoch_acc_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(acc))
            f.write("\n")

        if self.val_acc_flag:
            self.val_acc.append(val_acc)
            with open(os.path.join(self.save_path, "epoch_val_acc_" + str(self.time_str) + ".txt"), 'a') as f:
                f.write(str(val_acc))
                f.write("\n")
        self.acc_plot()

    def acc_plot(self):
        iters = range(len(self.acces))

        plt.figure()

        plt.plot(iters, self.acces, 'red', linewidth=2, label='train acc')
        try:
            plt.plot(iters, scipy.signal.savgol_filter(self.acces, 5 if len(self.acces) < 25 else 15, 3), 'green',
                     linestyle='--', linewidth=2, label='smooth train acc')
        except:
            pass

        if self.val_acc_flag:
            plt.plot(iters, self.val_acc, 'coral', linewidth=2, label='val acc')
            try:
                plt.plot(iters, scipy.signal.savgol_filter(self.val_acc, 5 if len(self.acces) < 25 else 15, 3),
                         '#8B4513', linestyle='--', linewidth=2, label='smooth val acc')
            except:
                pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Acc')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.save_path, "epoch_acc_" + str(self.time_str) + ".png"))

        plt.cla()
        plt.close("all")


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

# if __name__ == '__main__':
#     scheduler = lr_scheduler()
#     print(scheduler.scheduler())


