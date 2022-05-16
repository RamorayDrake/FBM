import argparse
import os
import random
import shutil
import time
import logging
import warnings
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle

import torch
import nncf  # Important - should be imported directly after torch

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from dataset_loader import dataloader
from tensorboardX import SummaryWriter
# torchvision for imagenet, resnet_cifar for cifar models
import torchvision.models as tvmodels
import utils.models.resnet_cifar as resnet_cifar

#from clearml import Task
#task = Task.init(project_name="Quantization", task_name="latency_based")
import datetime

from nncf import NNCFConfig
from nncf.torch import create_compressed_model, register_default_init_args
from collections import OrderedDict



from utils.hessians import EF_Hessian
from bit_config import *
from utils import *
from usage_extractor import *
from Latency_loss import Quantize_scores
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--ds', type=str,default='imagenet',
                    help='imagenet / cifar10 / cifar100')
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    help='model architecture')
parser.add_argument('--teacher-arch',
                    type=str,
                    default='resnet101',
                    help='teacher network used to do distillation')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-qf', '--quant-freq', default=10, type=int,
                    metavar='N', help='quant print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--save-path',
                    type=str,
                    default='checkpoints/imagenet/test/',
                    help='path to save the quantized model')
parser.add_argument('--data-percentage',
                    type=float,
                    default=1,
                    help='data percentage of training data')
parser.add_argument('--checkpoint-iter',
                    type=int,
                    default=-1,
                    help='the iteration that we save all the featuremap for analysis')
parser.add_argument('--resume-quantize',
                    action='store_true',
                    help='if True map the checkpoint to a quantized model,'
                         'otherwise map the checkpoint to an ordinary model and then quantize')
parser.add_argument('--resume-metadata',
                    action='store_true',
                    help='if True resume metadata')
parser.add_argument('--distill-method',
                    type=str,
                    default='None',
                    help='you can choose None or KD_naive')
parser.add_argument('--distill-alpha',
                    type=float,
                    default=0.95,
                    help='how large is the ratio of normal loss and teacher loss')
parser.add_argument('--temperature',
                    type=float,
                    default=6,
                    help='how large is the temperature factor for distillation')
parser.add_argument('--create_table',
                    action='store_true',
                    help='flag to create latency statistics file')
parser.add_argument('--alpha', #5
                    type=float,
                    default=1.,
                    help='CE coefficient')
parser.add_argument('--beta',
                    type=float,
                    default=10., #70
                    help='Latency coefficient')
parser.add_argument('-T', default=1., type=float,
                    help='temperture for the latency loss')
parser.add_argument('--inter_rep_update', action='store_true',
                    help='enable inter_rep_update')
parser.add_argument('--hessian_aware',
                    type=float,
                    default=0.0,
                    help='How much of the hessian we mask out')
parser.add_argument('--retrain-quant-model',
                    action='store_true',
                    help='if True take a quantized model and continue training it')
best_net_lat = 1e5
best_net_score = 0
best_net_acc = 0



args = parser.parse_args()
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

datatime_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
args.save_path = os.path.join(args.save_path, datatime_str)
args.save_path = args.save_path + '_al_' + str(args.alpha) + '_be_' + str(args.beta) + '_T_' + str(args.T)
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

hook_counter = args.checkpoint_iter
hook_keys = []
hook_keys_counter = 0
filename_log = os.path.join(args.save_path, 'log.log')
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S', filename=filename_log)
logging.getLogger().setLevel(logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler())

logging.info(args)

#how to measure the best model
calc_preformence = lambda ACC,LAT:   (args.alpha * ACC) / (args.beta * LAT)

def main():
    print(f'alpha: {args.alpha},beta: {args.beta},T: {args.T} ')
    global writer
    writer = SummaryWriter()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen a seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably!')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_net_acc
    global best_net_lat
    global best_net_score
    args.gpu = gpu

    if args.gpu is not None:
        logging.info("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # create model
    pretrained = True if args.pretrained else False
    if args.ds == "cifar10":
        logging.info(f"=> pytorch model {args.arch} for CIFAR10 pretrained: {pretrained}")

        if args.arch== 'resnet20':
            model = resnet_cifar.ResNet20()
        else:
            model = resnet_cifar.ResNet18()
        if args.distill_method != 'None':
            print('teacher is ResNet18')
            teacher = resnet_cifar.ResNet18()
            file = './resnet18_fp/model_best.pth.tar'
            logging.info("KD => loading pretrained Teacher from'{}'".format(file))
            checkpoint = torch.load(file)['state_dict']
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            teacher.load_state_dict(new_state_dict)
        if args.pretrained:
            if args.arch== 'resnet20':
                file= './resnet20_fp/resnet20-12fca82f.th'
                if os.path.isfile(file):
                    checkpoint = torch.load(file)
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    logging.info("=> no checkpoint found")
            else:
                file = './resnet18_fp/model_best.pth.tar'
                if os.path.isfile(file):
                    logging.info("=> loading pretrained cifar model '{}'".format(file))
                    checkpoint = torch.load(file)['state_dict']
                    new_state_dict = OrderedDict()
                    for k, v in checkpoint.items():
                        name = k[7:]  # remove `module.`
                        new_state_dict[name] = v
                    # load params
                    model.load_state_dict(new_state_dict)
                else:
                    logging.info("=> no checkpoint found")
    else:
        logging.info(f"=> pytorch model {args.arch} pretrained: {pretrained}")
        model = tvmodels.__dict__[args.arch](pretrained=pretrained)
        if args.distill_method != 'None':
            teacher = tvmodels.__dict__[args.arch](pretrained=True)

    train_loader, val_loader, train_sampler, ds_length = dataloader(args, download=True)
    #nncf_config = NNCFConfig.from_json(str(args.arch)+"_config.json")
    conf_name = f'{args.arch}_{args.ds}_manual_config.json'
    nncf_config = NNCFConfig.from_json(conf_name)
    #nncf_config = NNCFConfig.from_json(str(args.arch) + "_" + str(args.ds) + "_manual_config.json")

    #nncf_config = allocation2nncf_config(nncf_config, bit_allocation, args.arch, quant_act=False)
    cudnn.benchmark = True
    nncf_config = register_default_init_args(nncf_config, train_loader, val_loader=val_loader)


    ######
    check_model_size(model)

    compression_ctrl, model = create_compressed_model(model, nncf_config, dump_graphs=False)

    if args.create_table:
        create_model_latency_scheme(model, args.arch,ds=args.ds, device=device, reps=100)

    quant_layers_list= layers_for_quant_list(args.arch)

    bit_allocation = assign_bit_allocation({},q_layers_list=quant_layers_list,init=8)
    ref_lat = eval_inference(args.arch, bit_allocation)
    #logging.info(model)

    latencies = [] #initiate as HAWQ

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            if args.distill_method != 'None':
                teacher.cuda(args.gpu)
                teacher = torch.nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
            if args.distill_method != 'None':
                teacher.cuda()
                teacher = torch.nn.parallel.DistributedDataParallel(teacher)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        #model = model.cuda(args.gpu)
        model = torch.nn.DataParallel(model).cuda(args.gpu)
        if args.distill_method != 'None':
            teacher = teacher.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()
        if args.distill_method != 'None':
            teacher = torch.nn.DataParallel(teacher).cuda()

    # Resume training from checkpoint
    if args.resume and not args.resume_quantize:
        if os.path.isfile(args.resume):
            logging.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)['state_dict']
            model.load_state_dict(checkpoint)
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))
    if args.resume and args.resume_quantize:
        args.retrain_quant_model = True
        if os.path.isfile(args.resume):
            logging.info("=> loading quantized checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            bit_allocation = checkpoint['bit_allocation']
            best_net_acc = checkpoint['best_net_acc']
            best_net_lat = checkpoint['best_net_lat']
            if args.gpu is not None:
                best_net_acc = best_net_acc.to(args.gpu)
                # best_net_lat = best_net_lat.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            assign_bit_allocation2(model,bit_allocation)
            latencies.append(eval_inference(args.arch, bit_allocation)/ref_lat)
            logging.info("=> accuracy '{}', latency '{}', epoch '{}'".format(checkpoint['best_net_acc'],
                                                                             checkpoint['best_net_lat'],
                                                                             checkpoint['epoch']))
        else:
            logging.info("=> no quantized checkpoint found at '{}'".format(args.resume))
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    conv_weight_parameters = []
    for pname, p in model.named_parameters():
        if 'conv' in pname and 'weight' in pname:
            # conv_weight_parameters.append({pname[:-7]:p})
            conv_weight_parameters.append({'params':p,'name':pname[:-7]})
        else:
            conv_weight_parameters.append({'params':p,'name':'noname'})
    optimizer = torch.optim.SGD(conv_weight_parameters, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs)

    # optionally resume optimizer and meta information from a checkpoint
    if args.resume_metadata:
        if os.path.isfile(args.resume):
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_net_acc = checkpoint['best_net_acc']
            best_net_lat = checkpoint['best_net_lat']
            if args.gpu is not None:
                best_net_acc = best_net_acc.to(args.gpu)
            #     best_net_lat = best_net_lat.to(args.gpu)
            optimizer.load_state_dict(checkpoint['optimizer'])
            logging.info("=> loaded optimizer and meta information from checkpoint '{}' (epoch {})".
                         format(args.resume, checkpoint['epoch']))
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))

    latencies.append(eval_inference(args.arch, bit_allocation)/ref_lat)

    Hessian = EF_Hessian(model, args.arch, train_loader,criterion=criterion,Full=False,name='8 bit',device=args.gpu)

    Q_scores = Quantize_scores(n_layers=len(bit_allocation), bits_rep=[2, 3, 4, 8], T=args.T, alpha=args.alpha,
                               beta=args.beta,max_lat=ref_lat,Hessian=Hessian)
    logging.info(f'Latencty: {latencies[-1]}')
    logging.info(f'bit_allocation = {bit_allocation}')
    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    best_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, bit_allocation, args)

        if args.distill_method != 'None':
            CE_loss = train(train_loader, model, criterion, optimizer, epoch, args,teacher)
        else:
            CE_loss = train(train_loader, model, criterion, optimizer, epoch, args)
        #scheduler.step()
        acc1 = validate(val_loader, model, criterion, args)
        curr_bit_allocation = bit_allocation
        if epoch % args.quant_freq == 0 and epoch != 0 and not args.retrain_quant_model:
            #print(model)
            logging.info(f'last bit_allocation = {bit_allocation}')
            latencies.append(eval_inference(args.arch, bit_allocation)/ref_lat)
            Q_scores.update_quant_scores(CE_loss, latencies[-1],epoch,hessian_aware=args.hessian_aware)
            #bit_allocation = assign_bit_allocation(bit_allocation, quant_layers_list, Q_scores.select_quantization(stochastic=True))
            ##TODO: delete!!!!
            if epoch == 40:
                bit_allocation = assign_bit_allocation({}, q_layers_list=quant_layers_list, init=2)
            if epoch == 100:
                exit(0)

            assign_bit_allocation2(model,bit_allocation)
            #print('our quant model:')
            #print(model)
            #Q_scores.plot_scores(save_name=args.arch+'_'+str(epoch))

        net_score = calc_preformence(acc1, latencies[-1])
        if args.retrain_quant_model:
            net_score = acc1
        logging.info(f' * Score [{net_score}] Acc@1 [{acc1}] Latency [{latencies[-1]}]')
        logging.info(f'                       scaled CE: {args.alpha * CE_loss} |scaled lat: {args.beta * latencies[-1]} ')


        # remember best acc@1 and save checkpoint
        is_best = net_score >= best_net_score #todo: use tradeoff loss
        if acc1 > best_net_acc+.5 and best_net_lat + 0.05 > latencies[-1]:
            logging.info('** replaced best by small margin **')
            is_best = True
        best_net_score = max(net_score, best_net_score)
        writer.add_scalar("Best_net_score", best_net_score, epoch)

        if is_best:
            best_epoch = epoch
            best_net_acc = acc1 # the network that achive the best score
            best_net_lat = latencies[-1]
        logging.info(f'==Best net==: epoch: {best_epoch}: [{best_net_score}] accuracy [{best_net_acc}] latency [{best_net_lat}]')

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_net_acc': best_net_acc,
                'best_net_lat': best_net_lat,
                'best_score': best_net_score,
                'optimizer': optimizer.state_dict(),
                "bit_allocation": curr_bit_allocation,
            }, is_best, args.save_path)
    #plot_latency(latencies)
    writer.export_scalars_to_json("./tensorboard.json")
    writer.close()

def train(train_loader, model, criterion, optimizer, epoch, args,teacher=None):
    batch_time = AverageMeter('Time', ':6.3f')
    #data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    model.train()
    if teacher:
        teacher.eval()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        if args.distill_method != 'None':
            with torch.no_grad():
                teacher_output = teacher(images)
            loss = loss_kd(output, target, teacher_output, args)
        else:
            loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        writer.add_scalar("Train_Loss", loss.item(), epoch)
        writer.add_scalar("Train_Acc1", top1.avg, epoch)
        writer.add_scalar("Train_Acc5", top5.avg, epoch)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    return losses.avg


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    #freeze_model(model)
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        #logging.info(' * Acc@1 [{top1.avg:.3f}] Acc@5 [{top5.avg:.3f}]'.format(top1=top1, top5=top5))

    """torch.save({'convbn_scaling_factor': {k: v for k, v in model.state_dict().items() if 'convbn_scaling_factor' in k},
                'fc_scaling_factor': {k: v for k, v in model.state_dict().items() if 'fc_scaling_factor' in k},
                'weight_integer': {k: v for k, v in model.state_dict().items() if 'weight_integer' in k},
                'bias_integer': {k: v for k, v in model.state_dict().items() if 'bias_integer' in k},
                'act_scaling_factor': {k: v for k, v in model.state_dict().items() if 'act_scaling_factor' in k},
                }, os.path.join(args.save_path, 'quantized_checkpoint.pth.tar'))"""

    #unfreeze_model(model)

    return top1.avg


def save_checkpoint(state, is_best, path=None):
    filename = os.path.join(path ,'checkpoint.pth.tar')
    torch.save(state, filename )
    if is_best:
        shutil.copyfile(filename, os.path.join(path, 'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def loss_kd(output, target, teacher_output, args):
    """
    Compute the knowledge-distillation (KD) loss given outputs and labels.
    "Hyperparameters": temperature and alpha
    The KL Divergence for PyTorch comparing the softmaxs of teacher and student.
    The KL Divergence expects the input tensor to be log probabilities.
    """
    alpha = args.distill_alpha
    T = args.temperature
    KD_loss = F.kl_div(F.log_softmax(output / T, dim=1), F.softmax(teacher_output / T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(output, target) * (1. - alpha)

    return KD_loss


def adjust_learning_rate(optimizer, epoch, bit_allocation, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    #print('lr = ', lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        if len(param_group['name']) > 10:
            name = param_group['name'][7:]
        #if name in bit_allocation:
        #    if bit_allocation.get(name) == 2:
        #        param_group['lr'] = lr / 10

def plot_latency(latencies,save_name='A graph of latency'):
    plt.clf()
    plt.plot(range(len(latencies)),latencies,'g^')
    plt.ylabel('latency')
    plt.xlabel('epoch')
    if save_name:
        plt.savefig('imgs/' + save_name + '.png')

if __name__ == '__main__':
    main()
