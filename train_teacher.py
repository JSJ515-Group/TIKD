from __future__ import print_function

import os
import argparse
import time

import torch
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn
import torch.backends.cudnn as cudnn
# import tensorboard_logger as tb_logger

from teacher_models import model_dict
from dataset.cifar100 import get_cifar100_dataloaders
from helper.util import save_dict_to_json, reduce_tensor, adjust_learning_rate_cifar
from helper.loops import train_vanilla as train, validate
from torch.utils.tensorboard import SummaryWriter


# This function parses the input arguments
def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # Basic settings
    parser.add_argument('--print-freq', type=int, default=200, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size for training')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers to run tasks')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--gpu_id', type=str, default='0', help='ID for CUDA_VISIBLE_DEVICES')

    # Optimization strategy
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='epochs to decrease learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='learning rate decay rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # Dataset
    parser.add_argument('--model', type=str, default='resnet110',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'ResNet18', 'ResNet34',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'vgg8_imagenet', 'shufflenet_v2_x0_5',
                                 'ShuffleV2_0_5', 'MobileNetV2_Imagenet',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2', 'ResNet50', 'ResNet18', 'ShuffleV2_Imagenet',
                                 'ResNet34x2', 'resnet32x4', 'ResNet101'])
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100'], help='dataset')
    parser.add_argument('-t', '--trial', type=str, default='0', help='trial id')
    parser.add_argument('--dali', type=str, choices=['cpu', 'gpu'], default=None)
    parser.add_argument('--data_path', type=str, default='./dataset/cifar-100-python', help='path to dataset')
    parser.add_argument('--resume', type=str, default='', help='path to resume checkpoint for continued training')

    # Multi-processing training
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='use multi-processing distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:23451', type=str,
                        help='URL used to set up distributed training')
    # Whether to return image features
    parser.add_argument('--return_feat', action='store_true', help='whether to return image features')
    parser.add_argument('--output_dir', type=str, default='./output_teacher', help='directory to save output files')


    opt = parser.parse_args()

    # Set different learning rates for specific models
    if opt.model in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    # Set model name
    opt.model_name = '{}_{}_lr_{}_decay_{}_trial_{}'.format(opt.model, opt.dataset, opt.learning_rate,
                                                            opt.weight_decay, opt.trial)
    
    opt.model_path = os.path.join(opt.output_dir, 'models')
    opt.tb_path = os.path.join(opt.output_dir, 'tensorboard_logs')
    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    if opt.dali is not None:
        opt.model_name += '_dali:' + opt.dali

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


best_acc = 0  # Best accuracy
total_time = time.time()  # Total runtime


def main():
    opt = parse_option()

    # Set GPU IDs to use
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id

    ngpus_per_node = torch.cuda.device_count()
    opt.ngpus_per_node = ngpus_per_node
    if opt.multiprocessing_distributed:
        # Calculate world size
        world_size = 1
        opt.world_size = ngpus_per_node * world_size
        # Launch distributed training processes using torch.multiprocessing.spawn
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, opt))
    else:
        main_worker(None if ngpus_per_node > 1 else opt.gpu_id, ngpus_per_node, opt)


def main_worker(gpu, ngpus_per_node, opt):
    writer = SummaryWriter(log_dir=opt.tb_folder)
    
    global best_acc, total_time
    opt.gpu = int(gpu)
    opt.gpu_id = int(gpu)

    if opt.gpu is not None:
        print("Use GPU: {} for training".format(opt.gpu))

    if opt.multiprocessing_distributed:
        # Set rank for distributed training
        opt.rank = int(gpu)
        dist_backend = 'nccl'
        dist.init_process_group(backend=dist_backend, init_method=opt.dist_url,
                                 world_size=opt.world_size, rank=opt.rank)

    # Model building
    n_cls = {
        'cifar100': 100,
    }.get(opt.dataset, None)

    model = model_dict[opt.model](num_classes=n_cls)

    # Print total number of model parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")

    # Load checkpoint
    start_epoch = 1
    if opt.resume:
        if os.path.isfile(opt.resume):
            print(f"=> loading checkpoint '{opt.resume}'")
            checkpoint = torch.load(opt.resume, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            # Optimizer state dict is also loaded here, assuming it exists in the checkpoint
            # If not, this line might cause an error if optimizer is not yet defined.
            # It's usually good practice to define optimizer before loading its state.
            optimizer.load_state_dict(checkpoint['optimizer']) 
            best_acc = checkpoint['best_acc']
            start_epoch = checkpoint['epoch'] + 1
            print(f"=> loaded checkpoint '{opt.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> no checkpoint found at '{opt.resume}'")

    # Optimizer
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        # Handle multi-processing distributed training
        if opt.multiprocessing_distributed:
            if opt.gpu is not None:
                torch.cuda.set_device(opt.gpu)
                model = model.cuda(opt.gpu)
                criterion = criterion.cuda(opt.gpu)
                # Adjust batch size
                opt.batch_size = int(opt.batch_size / ngpus_per_node)
                opt.num_workers = int((opt.num_workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.gpu])
            else:
                print('multiprocessing_distributed must be with a specific gpu id')
        else:
            criterion = criterion.cuda()
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model).cuda()
            else:
                model = model.cuda()

    cudnn.benchmark = True  # Optimize work mode

    # Data input
    if opt.dataset == 'cifar100':
        train_loader, val_loader = get_cifar100_dataloaders(data_dir=opt.data_path, batch_size=opt.batch_size, num_workers=opt.num_workers)
    else:
        raise NotImplementedError(opt.dataset)

    total_throughput = 0  # Accumulated image throughput
    total_epochs = 0  # Number of training epochs, used for average calculation

    # Training epoch loop
    for epoch in range(start_epoch, opt.epochs + 1):

        if opt.dataset in ['cifar100']:
            adjust_learning_rate_cifar(optimizer, epoch, opt)
        print("==> training...")

        time1 = time.time()
        train_acc, train_acc_top5, train_loss = train(epoch, train_loader, model, criterion, optimizer, opt,
                                                      opt.return_feat)
        time2 = time.time()

        # Calculate throughput (images/s)
        throughput = (opt.batch_size * len(train_loader)) / (time2 - time1)
        # Update accumulated throughput and number of training epochs
        total_throughput += throughput
        total_epochs += 1

        # Log training loss and accuracy to TensorBoard
        writer.add_scalar('Train/Acc@1', train_acc, epoch)
        writer.add_scalar('Train/Acc@5', train_acc_top5, epoch)
        writer.add_scalar('Train/Loss', train_loss, epoch)

        if opt.multiprocessing_distributed:
            metrics = torch.tensor([train_acc, train_acc_top5, train_loss]).cuda(opt.gpu, non_blocking=True)
            reduced = reduce_tensor(metrics, opt.world_size if 'world_size' in opt else 1)
            train_acc, train_acc_top5, train_loss = reduced.tolist()

        if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
            print(' * Epoch {}, Acc@1 {:.3f}, Acc@5 {:.3f}, Time {:.2f}'.format(epoch, train_acc, train_acc_top5,
                                                                                 time2 - time1))

        test_acc, test_acc_top5, test_loss = validate(val_loader, model, criterion, opt, opt.return_feat)
        
        # Log validation loss and accuracy to TensorBoard
        writer.add_scalar('Validation/Acc@1', test_acc, epoch)
        writer.add_scalar('Validation/Acc@5', test_acc_top5, epoch)
        writer.add_scalar('Validation/Loss', test_loss, epoch)

        if opt.dali is not None:
            train_loader.reset()
            val_loader.reset()

        if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
            print(' ** Acc@1 {:.3f}, Acc@5 {:.3f}'.format(test_acc, test_acc_top5))

            # Save the best model
            if test_acc > best_acc:
                best_acc = test_acc
                state = {
                    'epoch': epoch,
                    'model': model.module.state_dict() if opt.multiprocessing_distributed else model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                }
                save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model))

                test_metrics = {'test_loss': float('%.2f' % test_loss),
                                'test_acc': float('%.2f' % test_acc),
                                'test_acc_top5': float('%.2f' % test_acc_top5),
                                'epoch': epoch}

                save_dict_to_json(test_metrics, os.path.join(opt.save_folder, "test_best_metrics.json"))

                print('saving the best model!')
                torch.save(state, save_file)

    # Calculate average throughput after training
    avg_throughput = total_throughput / total_epochs
    print(f"Average Image Throughput: {avg_throughput:.2f} images/sec")

    # Save average throughput
    with open(os.path.join(opt.save_folder, "average_throughput.txt"), "w") as f:
        f.write(f"Average Image Throughput: {avg_throughput:.2f} images/sec")


    if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
        # Only for printing best accuracy
        print('best accuracy:', best_acc)

        # Save parameters
        state = {k: v for k, v in opt._get_kwargs()}

        # Number of parameters (millions)
        num_params = (sum(p.numel() for p in model.parameters()) / 1000000.0)
        state['Total params'] = num_params
        state['Total time'] = float('%.2f' % ((time.time() - total_time) / 3600.0))
        params_json_path = os.path.join(opt.save_folder, "parameters.json")
        save_dict_to_json(state, params_json_path)

        writer.close()


if __name__ == '__main__':
    main()