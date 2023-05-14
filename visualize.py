# modified from https://github.com/raoyongming/DynamicViT and https://github.com/facebookresearch/deit
import argparse
import numpy as np
import torch.backends.cudnn as cudnn

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from datasets import build_dataset2, get_post_process

import utils

from timm.utils import accuracy, ModelEma
from torchvision import utils as vutils

import torch
from torchvision import transforms

from PIL import Image
import os

from deit import evo_deit_vis

from torchvision import transforms
from torchvision.transforms import InterpolationMode

def get_transform(input_size):
    t = []
    resize_im = (input_size != 224)
    if resize_im:
        # 这段代码的作用是计算输入图像的大小。它使用了一个公式，其中args.input_size是输入图像的高或宽（假设输入图像是正方形），256是ImageNet数据集中图像的最大边长，
        # 224是ImageNet数据集中图像的标准边长。
        # 这个公式的目的是将输入图像的大小按比例缩放，以便与ImageNet数据集中的图像大小相匹配。最终，size变量将包含输入图像的新大小。
        size = int((256 / 224) * args.input_size)
        # size = 224
        t.append(
            # transforms.Resize是PyTorch中的一个图像变换函数，用于对图像进行缩放操作。在这段代码中，使用transforms.Resize对图像进行缩放操作，其中的size参数是一个整数，表示缩放后的短边大小。
            # interpolation参数表示缩放时使用的插值方法，这里设置为3，表示使用双三次插值。双三次插值是一种常用的插值方法，可以在缩放图像时保持图像的平滑性和细节信息。
            # transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
            transforms.Resize(size, interpolation=InterpolationMode.BICUBIC)
        )
        t.append(transforms.CenterCrop(args.input_size))
        t.append(transforms.ToTensor())
    else:
        t.append(transforms.ToTensor())

    return transforms.Compose(t)


def get_keep_indices(decisions):
    keep_indices = []
    for i in range(3):
        if i == 0:
            keep_indices.append(decisions[i])
        else:
            keep_indices.append(keep_indices[-1][decisions[i]])
    return keep_indices


def gen_masked_tokens(tokens, indices, alpha=0.3):
    indices = [i for i in range(196) if i not in indices]
    tokens = tokens.copy()
    # 三个255接近白色,之所以这么处理,是为了可视化的效果,使得原来的图像隐约可见
    tokens[indices] = alpha * tokens[indices] + (1 - alpha) * 255
    return tokens


def recover_image(tokens):
    # image: (C, 196, 16, 16)
    # 将掩码后的图像块恢复为可视化图像，该函数会将每个小块reshape为14x16x3的形状，然后将所有小块reshape为原始图像的形状。
    image = tokens.reshape(14, 14, 16, 16, 3).swapaxes(1, 2).reshape(224, 224, 3)
    return image

# 这段代码用于可视化模型的中间层输出，其中输入参数包括原始图像和要可视化的特征图的索引列表。
# 这个过程的目的是将掩码后的图像块恢复为原始图像的形状，并生成一个可视化图像。
# 这个过程的目的是突出显示特征图中的重要区域，以便更好地理解模型的预测结果。
def gen_visualization(image, keep_indices):
    # keep_indices = get_keep_indices(decisions)
    # 具体来说，首先将原始图像reshape为14x16x14x16x3的形状，然后使用swapaxes(1, 2)函数交换第1维和第2维的位置，将其reshape为196x16x16x3的形状。
    # 这个过程的目的是将原始图像划分为196个小块，以便对每个小块进行可视化。
    image_tokens = image.reshape(14, 16, 14, 16, 3).swapaxes(1, 2).reshape(196, 16, 16, 3)

    # 接下来，使用gen_masked_tokens函数生成掩码后的图像块，该函数接受原始图像块和要保留的像素索引列表作为输入，并返回一个掩码后的图像块。
    # 最后，使用recover_image函数将掩码后的图像块恢复为可视化图像，该函数会将每个小块reshape为14x16x3的形状，然后将所有小块reshape为原始图像的形状。
    viz = recover_image(gen_masked_tokens(image_tokens, keep_indices))
    return viz


def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)

    # Model parameters
    # parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL',
    #                     help='Name of model to train')
    parser.add_argument('--model', default='evo_deit_tiny_vis_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher-path', type=str, default='')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')

    # Dataset parameters
    # parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', type=str,
    #                     help='dataset path')
    parser.add_argument('--data-path', default='/data/ML_document/imagenette2/', type=str,
                        help='dataset path')                        
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='./test_img/', help='path where to save')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_false', help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--excel_filename', type=str, default='attention_matrix_cls', help='filename of saving excel')

    # visualization
    parser.add_argument('--img-path', default='', type=str,
                        help='path to images to be visualized. Set '' to visualize batch images in imagenet val.')
    parser.add_argument('--save-name', default='', type=str,
                        help='name to save when visualizing a single image. Set '' to save name as the original image.')
    parser.add_argument('--layer-wise-prune', action='store_true',
                        help='set true when visualize a model trained without layer to stage training strategy')
    return parser


def save_image_tensor(input_tensor: torch.Tensor, filename):
    """
    将tensor保存为图片
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    """
    assert ((len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1) or len(input_tensor.shape) == 3)
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 反归一化
    # input_tensor = unnormalize(input_tensor)
    vutils.save_image(input_tensor, filename)


@torch.no_grad()
def visualize_single_img(img_input, model, device, transform, post_process, save_name):
    # 这段代码用于在模型上进行推理，并获取可视化字典vis_dict。
    # 首先，使用model.eval()将模型设置为评估模式，这意味着模型的参数不会被更新，并且在前向传递过程中不会使用dropout等随机性操作。
    # 这是因为在评估模式下，我们只关心模型的输出结果，而不需要进行反向传播和参数更新。
    model.eval()
    # set stage_wise_prune = True if the trained model is under layer-to-stage training strategy
    model.stage_wise_prune = not args.layer_wise_prune

    # img: 1, 3, H, W
    image_raw = transform(img_input)
    save_image_tensor(image_raw, Path(args.output_dir, '{}.jpg'.format(save_name)))
    images = post_process(image_raw)
    images = images.unsqueeze(0)
    images = images.to(device, non_blocking=True)
    print(images.shape)
    # compute output
    # 然后，使用torch.cuda.amp.autocast()上下文管理器将输入数据转换为半精度浮点数，并在模型上进行前向传递，得到输出结果output。
    # 在这个上下文管理器中，所有的计算都会自动转换为半精度浮点数，从而减少了内存的使用和计算的时间。
    with torch.cuda.amp.autocast():
        output = model(images)
        # 使用model.get_vis_dict()获取可视化字典vis_dict，用于可视化模型的中间层输出。
        # 这个函数在模型的前向传递过程中记录中间层的输出，并将其保存在一个字典中。
        vis_dict = model.get_vis_dict()
    # 这段代码用于将PyTorch张量转换为NumPy数组，并将其还原为原始图像。首先，将image_raw乘以255，将其还原为原始像素值。
    # 然后，使用squeeze(0)函数将image_raw张量的第0维（批次维）去除，因为在这里我们只处理单张图像。
    # 接下来，使用permute(1, 2, 0)函数将image_raw张量的维度从[3, H, W]转换为[H, W, 3]，以便将其转换为NumPy数组。
    # 最后，使用cpu()函数将张量从GPU内存中移动到CPU内存中，并使用numpy()函数将其转换为NumPy数组。
    # 这样，就将PyTorch张量转换为了NumPy数组，并将其还原为原始图像。这个过程通常用于可视化模型的输出结果，以便更好地理解模型的预测结果。
    image_raw = image_raw * 255
    image_raw = image_raw.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # 这段代码用于生成可视化图像，并将其保存到指定的输出目录中。首先，使用for循环遍历可视化字典vis_dict中的每个键k，其中每个键对应一个中间层的输出。
    # 然后，使用gen_visualization函数生成可视化图像，该函数接受原始图像和要可视化的特征图的索引列表作为输入，并返回一个可视化图像的NumPy数组。
    # 这个函数的具体实现可能因模型而异，但通常会使用某种可视化技术（如CAM）来生成可视化图像。
    for k in vis_dict:
        keep_indices = vis_dict[k]
        viz = gen_visualization(image_raw, keep_indices)

        # 接下来，将生成的可视化图像转换为PyTorch张量，并使用permute(2, 0, 1)函数将其维度从[H, W, 3]转换为[3, H, W]，以便将其保存为图像文件。
        viz = torch.from_numpy(viz).permute(2, 0, 1)
        # 然后，将可视化图像的像素值除以255，将其归一化到[0, 1]范围内。
        viz = viz / 255

        # 最后，使用save_image_tensor函数将可视化图像保存到指定的输出目录中，其中文件名包括原始图像的名称和中间层的名称。
        save_image_tensor(viz,
                          Path(args.output_dir, '{}_{}.jpg'.format(save_name, k)))
    print("Visualization finished")

# 这是一个名为visualize的函数，它接受data_loader、model、device和post_process作为输入。该函数使用给定的模型对输入数据执行可视化。
@torch.no_grad()
def visualize(data_loader, model, device, post_process):
    criterion = torch.nn.CrossEntropyLoss()

    # 该函数为日志记录设置一个度量记录器和标头。
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # 它将模型设置为评估模式，并根据是否提供layer_wise_prune参数来调整修剪策略。
    # switch to evaluation mode
    model.eval()

    # set stage_wise_prune = True if the trained model is under layer-to-stage training strategy
    model.stage_wise_prune = not args.layer_wise_prune

    # 对于数据加载器中的每一批图像和目标，该函数对图像进行预处理，保存原始图像，并计算模型的输出。
    # 这行代码遍历来自数据加载器的一批数据，其中' images_raw_full '是输入图像，' target_full '是相应的目标标签。
    # 它还使用' metric_logger '记录测试期间模型的性能指标，并且每10次迭代记录一次。' header '是用作日志输出前缀的字符串。
    # 此代码使用' MetricLogger '对象来记录评估过程中的指标。具体来说，该对象的' log_every '方法被调用来记录每10次' data_loader '迭代的度量。
    # ' header '变量用于打印日志的标题。从' log_every '方法返回的值是' images_raw_full '和' target_full '，它们是模型评估的输入图像和相应的目标。
    for images_raw_full, target_full in metric_logger.log_every(data_loader, 8, header):   # change 10 to 8
        B = images_raw_full.shape[0]
        # print(B)  # 默认的batch_size为64
        # 按照B对批次中的每个样本单独处理可视化输出
        for index in range(B):
            images_raw = images_raw_full[index:index + 1]
            target = target_full[index:index + 1]
            assert images_raw.shape[0] == 1
            images = post_process(images_raw)

            name = 'label{}_seed{}_index{}.jpg'.format(str(target.item()), int(args.seed), index)
            save_image_tensor(images_raw, Path(args.output_dir, name))
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast():
                output = model(images)
            vis_dict = model.get_vis_dict()
            loss = criterion(output, target)

            images_raw = images_raw * 255
            images_raw = images_raw.squeeze(0).permute(1, 2, 0).cpu().numpy()
            # if np.max(images_raw) > 3:
            #     images_raw = images_raw / 255

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            if acc1 == 0:
                judger = 'wrong'
            elif acc1 == 100:
                judger = 'right'
            else:
                raise ValueError('xxxx')

            # 计算模型输出的损失和准确性，更新度量记录器，并根据某些标准保存输入图像的可视化。           
            for k in vis_dict:
                keep_indices = vis_dict[k]
                viz = gen_visualization(images_raw, keep_indices)
                viz = torch.from_numpy(viz).permute(2, 0, 1)
                viz = viz / 255

                name = 'label{}_seed{}_{}_index{}_{}.jpg'.format(
                    str(target.item()),
                    int(args.seed), k, index, judger)
                # 保存图片，这个是我自己加的，原来程序不输出图片,看源代码是有的，可能是被我不小心删掉了
                save_image_tensor(viz, Path(args.output_dir, name))

            batch_size = images.shape[0]
            # 这段代码使用当前批处理的损失、accuracy@1和accuracy@5的当前值更新度量记录器。
            # ' metric_logger.update() '使用' loss.item() '值更新当前epoch的损失平均值。
            # ' metric_logger.meters['acc1'].update() '使用当前批次的'acc1 .item() '值更新当前epoch的accuracy@1平均值，同时考虑到批次' n=batch_size '中的样本数量。
            # ' metric_logger.meters['acc5'].update() '使用当前批次的'acc5 .item() '值更新当前epoch的accuracy@5平均值，同时考虑到批次' n=batch_size '中的样本数量。
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        print("Visualization finished")
        break

    # gather the stats from all processes
    # 上面的代码同步多个进程之间的度量，并打印整个数据集的平均精度和损失值。然后，它返回一个字典，其中包含在评估过程中收集的所有仪表(指标)的全局平均值。
    # 具体来说，' synchronize_between_processes() '方法在所有进程中同步度量记录器的状态，以确保最终的度量值在所有进程中保持一致。
    metric_logger.synchronize_between_processes()
    # ' print '语句使用' metric_logger '中相应仪表的' global_avg '属性，格式化并打印整个数据集的top-1和top-5的平均精度以及损失。
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    # 最后，该函数返回一个字典，该字典将每个指标名称映射到它的全局平均值，该全局平均值是通过调用相应指标的“global_avg”属性获得的。
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def vis_single(args):
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    # 这段代码是用于启用CuDNN的自动调整功能，以提高训练速度的。CuDNN是NVIDIA提供的一个针对深度神经网络的加速库，它可以在GPU上加速神经网络的训练和推理。
    # 启用自动调整功能后，CuDNN会根据输入数据的大小和形状自动选择最优的卷积算法，以提高训练速度。但是，这可能会导致一些额外的开销，因此只有在输入数据的大小和形状不变时才应该启用此功能。
    cudnn.benchmark = True

    transform = get_transform(input_size=224)  # set input_size to other value if the test image is not 224*224
    post_process = get_post_process()

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=1000,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )

    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            # 这段代码是用于加载预训练模型的。它使用PyTorch的torch.load函数从指定的文件中加载模型的参数。参数args.resume指定了模型参数文件的路径。
            # map_location参数指定了将模型参数加载到哪个设备上。
            # 在这里，它被设置为'cpu'，这意味着模型参数将被加载到CPU内存中。如果模型参数是在GPU上训练的，则需要将其加载到GPU上，以便在GPU上进行推理或继续训练。
            checkpoint = torch.load(args.resume, map_location='cpu')

        model.load_state_dict(checkpoint['model'])

    # 在命令行中使用--img-path指定的参数会被解析为一个字符串类型的值，并存储在args.img_path中。
    img_input = Image.open(args.img_path)
    if args.save_name == '':
        save_name = os.path.basename(args.img_path).split('.')[0]
    else:
        save_name = args.save_name
    if args.eval:
        test_stats = visualize_single_img(img_input, model, device, transform, post_process, save_name=save_name)
        return


def vis_batch(args):
    utils.init_distributed_mode(args)
    print(args)

    if args.distillation_type != 'none' and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    # dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, args.nb_classes = build_dataset2(is_train=False, args=args)
    post_process = get_post_process()

    if True:  # args.distributed:

        # 这段代码用于获取当前进程的全局排名和总进程数。具体来说，utils.get_world_size()函数返回当前进程组中的总进程数，而utils.get_rank()函数返回当前进程在进程组中的排名。
        # 这些函数通常在分布式训练中使用，用于确定当前进程在整个训练过程中的位置，从而方便地进行数据并行和模型并行。
        # 在分布式训练中，通常会将训练数据划分为多个子集，并将每个子集分配给不同的进程进行处理。每个进程都会独立地计算梯度，并将其发送给其他进程进行聚合。
        # 在这个过程中，需要确保每个进程都能够访问全局排名和总进程数，以便进行数据划分和梯度聚合。因此，获取全局排名和总进程数是分布式训练中的一个重要步骤。
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        # if args.repeated_aug:
        #     sampler_train = RASampler(
        #         dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        #     )
        # else:
        #     sampler_train = torch.utils.data.DistributedSampler(
        #         dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        #     )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            # sampler_val = torch.utils.data.DistributedSampler(
            #     dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)

            # 这段代码用于创建一个分布式采样器，用于在验证集上进行分布式训练。具体来说，torch.utils.data.DistributedSampler()函数接受四个参数：dataset，num_replicas，rank和shuffle。
            # 其中，dataset表示要采样的数据集，num_replicas表示进程组中的总进程数，rank表示当前进程在进程组中的排名，shuffle表示是否对数据进行随机重排。
            # 分布式采样器的作用是将数据集划分为多个子集，并将每个子集分配给不同的进程进行处理。在每个进程中，采样器会根据进程的排名和总进程数计算出该进程需要处理的数据子集，
            # 并返回该子集的索引。这个过程可以保证每个进程都能够访问到整个数据集，并且每个数据样本只会被处理一次。
            # 在这段代码中，我们使用torch.utils.data.DistributedSampler()函数创建了一个分布式采样器sampler_val，用于在验证集上进行分布式训练。其中，dataset_val表示要采样的验证集，
            # num_replicas表示进程组中的总进程数，即num_tasks，rank表示当前进程在进程组中的排名，即global_rank，shuffle表示是否对数据进行随机重排，这里设置为True，表示对数据进行随机重排。
            # 创建完采样器后，我们可以将其传递给DataLoader，用于在验证集上进行分布式训练。

            # 在分布式训练中，如果数据集的大小不能被进程组中的总进程数整除，那么有些进程可能会处理比其他进程更多或更少的数据。
            # 为了解决这个问题，通常会采用一些技术来确保每个进程处理的数据集大小相同。
            # 一种常见的方法是在数据集中添加一些重复的数据样本，以确保每个进程处理的数据集大小相同。具体来说，可以将数据集的大小扩展到能够被进程组中的总进程数整除的最小值，
            # 并将多余的数据样本复制到数据集的末尾。这样做可以确保每个进程处理的数据集大小相同，并且每个数据样本只会被处理一次。
            # 例如，假设我们有一个大小为10的数据集，要在4个进程中进行分布式训练。由于10不能被4整除，因此我们需要将数据集的大小扩展到12，以确保每个进程处理的数据集大小相同。
            # 为了实现这一点，我们可以将数据集中的最后两个数据样本复制到数据集的末尾，这样数据集的大小就变成了12。然后，我们可以将数据集划分为4个子集，每个子集包含3个数据样本。
            # 在分布式训练中，每个进程将处理一个子集，因此每个数据样本只会被处理一次。
            # 需要注意的是，如果我们将多余的数据样本复制到数据集的末尾，那么这些数据样本可能会被处理多次。
            # 例如，在上面的例子中，最后两个数据样本会被复制到数据集的末尾，因此它们可能会被处理两次。
            # 为了避免这种情况，我们可以在创建分布式采样器时，将参数shuffle设置为False，这样可以确保每个进程处理的数据集是固定的，并且每个数据样本只会被处理一次。
            # 设置shuffle设置为True之后,每个进程中样本不重复，但是总的数据集中是有样本重复的.
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        else:
            # sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            sampler_val = torch.utils.data.RandomSampler(dataset_val)
    else:
        # sampler_train = torch.utils.data.RandomSampler(dataset_train)
        # sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        # 创建了一个名为sampler_val的采样器，它是一个RandomSampler类型的对象。
        # RandomSampler是PyTorch中的一个采样器类，它可以随机地从给定的数据集中选择样本，用于创建数据加载器。
        # 在这个例子中，RandomSampler被用于创建验证数据集的数据加载器.
        sampler_val = torch.utils.data.RandomSampler(dataset_val)
    # data_loader_train = torch.utils.data.DataLoader(
    #     dataset_train, sampler=sampler_train,
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers,
    #     pin_memory=args.pin_mem,
    #     drop_last=True,
    # )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    print("Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )

    # 这段代码用于加载预训练模型的权重，并将其应用于当前模型。
    # 由于预训练模型和当前模型的结构可能不同，因此需要对它们进行适当的匹配。
    # 在这个例子中，如果预训练模型和当前模型的某些层的形状不匹配，则需要将预训练模型中的这些层删除。
    # 具体来说，如果预训练模型中的`head.weight`、`head.bias`、`head_dist.weight`或`head_dist.bias`层的形状不匹配，则将其从预训练模型中删除。
    # 具体来说，如果`args.finetune`参数不为空，则表示需要加载预训练模型的权重。
    if args.finetune:
        # 如果`args.finetune`是一个URL，则使用`torch.hub.load_state_dict_from_url`函数从URL中加载预训练模型的权重。
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        # 如果`args.finetune`是一个本地文件路径，则使用`torch.load`函数从本地文件中加载预训练模型的权重。
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        # 加载预训练模型的权重后，将其应用于当前模型。具体来说，将预训练模型的权重复制到当前模型的对应层中。
        # 在这个例子中，预训练模型的权重被保存在`checkpoint`变量中，其中`checkpoint['model']`是一个字典，包含了预训练模型的权重。
        # `state_dict`是当前模型的权重，它是一个字典，包含了当前模型的所有层的权重。
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print("Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # 这段代码用于将预训练模型的位置编码（positional embedding）应用于当前模型。
        # 位置编码是一种用于将序列中每个位置的信息编码为向量的技术，它在许多自然语言处理和计算机视觉任务中都得到了广泛的应用。
        # 具体来说，这段代码首先从预训练模型中获取位置编码`pos_embed_checkpoint`，并计算出当前模型中位置编码的大小和形状。
        # 然后，将预训练模型中的位置编码分为额外的令牌（extra_tokens）和位置令牌（pos_tokens）。额外的令牌是当前模型中多余的令牌，而位置令牌是需要插值的令牌。
        # 接下来，将位置令牌重新排列成一个四维张量，并使用双三次插值将其调整为当前模型中位置编码的大小和形状。
        # 最后，将额外的令牌和插值后的位置令牌连接起来，得到新的位置编码`new_pos_embed`。将新的位置编码应用于当前模型，以更新模型的权重。
        # 需要注意的是，位置编码是一个非常重要的组件，它可以帮助模型更好地理解序列中每个位置的信息。
        # 在这个例子中，通过将预训练模型的位置编码应用于当前模型，可以帮助我们更快地训练模型，并提高模型的性能和泛化能力。
        # interpolate position embedding
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model['pos_embed'] = new_pos_embed

        model.load_state_dict(checkpoint_model, strict=False)

    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    # 这段代码是在PyTorch中使用分布式训练时的常见写法。在分布式训练中，模型需要在多个GPU上进行并行计算，以加快训练速度。
    # PyTorch提供了`torch.nn.parallel.DistributedDataParallel`模块来实现分布式训练。
    # 这段代码的作用是将模型包装成`DistributedDataParallel`模块，并将其赋值给`model`变量。`DistributedDataParallel`模块可以将模型复制到多个GPU上，
    # 并在每个GPU上计算不同的数据子集，最后将结果汇总。这样可以加快训练速度，并且可以处理大型数据集。
    # 同时，这段代码还将未经过`DistributedDataParallel`包装的模型赋值给`model_without_ddp`变量。这是因为在训练过程中，有时需要对模型进行一些操作，
    # 如保存或加载模型。如果直接对`DistributedDataParallel`包装后的模型进行操作，可能会导致一些问题。因此，将未经过包装的模型保存在`model_without_ddp`变量中，可以避免这些问题。
    # 最后，如果在命令行中指定了`--distributed`参数，说明要进行分布式训练，此时将模型包装成`DistributedDataParallel`模块。
    # 如果没有指定`--distributed`参数，则不进行分布式训练，此时`model`和`model_without_ddp`变量指向同一个模型。
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # 以下代码主要涉及到分布式训练中的学习率调整和优化器设置
    # 用来进行分布式训练下的学习率调整，其中args.lr表示原始的学习率，args.batch_size表示每个batch的大小，utils.get_world_size()返回当前分布式训练中的进程数，
    # 512.0是一个默认的参考 batch size。这里采用的是线性学习率缩放方法，将原始学习率按照批大小和进程数量进行线性缩放。
    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    # 创建一个优化器，其中args包含了优化器的相关参数，model_without_ddp表示没有进行分布式数据并行的模型。
    # `create_optimizer`函数返回的优化器类型取决于`args.optimizer`参数的值。`args.optimizer`通常在训练脚本的命令行或配置文件中指定，可以是以下常见的优化器之一：
    # - SGD（随机梯度下降法）- Adam（Adaptive Moment Estimation）- Adadelta - AdamW等
    # 因此，具体使用哪种优化器取决于`args.optimizer`参数的值和其他参数的配置,本文默认采用的是AdamW
    optimizer = create_optimizer(args, model_without_ddp)
    # 定义了一个NativeScaler来进行混合精度训练，该对象将用于缩放梯度以保持数值稳定性。
    loss_scaler = NativeScaler()

    # 建一个学习率调度器，其中args包含了学习率调度器的相关参数，optimizer是优化器对象。
    # 返回的lr_scheduler将在训练过程中被用来更新学习率。另外，这里用一个下划线_来占位符代表第二个返回值，因为create_scheduler函数返回的是一个元组，而这里只需要第一个元素。
    lr_scheduler, _ = create_scheduler(args, optimizer)

    if args.distillation_type != 'none':
        assert args.teacher_path, 'need to specify teacher-path when using distillation'
        print("Creating teacher model: {args.teacher_model}")
        teacher_model = create_model(
            args.teacher_model,
            pretrained=False,
            num_classes=args.nb_classes,
            global_pool='avg',
        )
        if args.teacher_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.teacher_path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.teacher_path, map_location='cpu')
        teacher_model.load_state_dict(checkpoint['model'])
        teacher_model.to(device)
        teacher_model.eval()

    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])

    if args.eval:
        test_stats = visualize(data_loader_val, model, device, post_process=post_process)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.eval = True

    #  ArgumentParser 会通过接受第一个长选项字符串并去掉开头的 -- 字符串来生成 dest 的值。 
    # 如果没有提供长选项字符串，则 dest 将通过接受第一个短选项字符串并去掉开头的 - 字符来获得。 
    # 任何内部的 - 字符都将被转换为 _ 字符以确保字符串是有效的属性名称。
    # --img-path --> img_path属性,见https://docs.python.org/zh-cn/3/library/argparse.html
    if args.img_path == '':
        # To visualize batch images of imagenet val, please run this:
        vis_batch(args)
    else:
        # To visualize a single image, please run this:
        vis_single(args)
