# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional
import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils
import time


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('base_loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('distillation_loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('alignment_loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))  # Add alignment loss

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    # Record epoch start time
    epoch_start_time = time.time()

    # Check if cosub is used
    if args and args.cosub:
        bce_criterion = torch.nn.BCEWithLogitsLoss()

    epoch_throughput = 0  # To record throughput for the current epoch
    total_samples = 0  # To record total samples processed in the current epoch

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        batch_start_time = time.time()  # Start time for each batch

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if args and args.cosub:
            samples = torch.cat((samples, samples), dim=0)

        if args and args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)


        # Forward pass
        with torch.cuda.amp.autocast():
            outputs = model(samples)

            # Use DistillationLoss to calculate total loss
            if not args or not args.cosub:
                total_loss, base_loss, distillation_loss, alignment_loss = criterion(samples, outputs, targets)
            else:
                # When cosub is enabled, process outputs in two parts
                outputs = torch.split(outputs, outputs.shape[0] // 2, dim=0)
                # Calculate cross-entropy loss
                loss = 0.25 * criterion(outputs[0], targets)[0]
                loss = loss + 0.25 * criterion(outputs[1], targets)[0]
                loss = loss + 0.25 * criterion(outputs[0], outputs[1].detach().sigmoid())[0]
                loss = loss + 0.25 * criterion(outputs[1], outputs[0].detach().sigmoid())[0]
                total_loss = loss


        if isinstance(total_loss, torch.Tensor):
            total_loss_value = total_loss.item()
        else:
            total_loss_value = total_loss


        if not math.isfinite(total_loss_value):
            print(f"Loss is {total_loss_value}, stopping training")
            sys.exit(1)

        optimizer.zero_grad()

        # Check if it's a second-order optimizer
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(total_loss, optimizer, clip_grad=max_norm, parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(total_loss=total_loss_value)
        metric_logger.update(base_loss=base_loss.item())
        metric_logger.update(distillation_loss=distillation_loss.item())
        metric_logger.update(alignment_loss=alignment_loss.item())  # Update the alignment loss

        # Calculate throughput
        batch_end_time = time.time()
        batch_time = batch_end_time - batch_start_time
        throughput = samples.shape[0] / batch_time  # Images processed per second

        epoch_throughput += throughput  # Accumulate throughput for the current epoch
        total_samples += samples.shape[0]  # Accumulate total samples processed in the current epoch

    # Gather stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # Print average loss for each epoch
    avg_base_loss = metric_logger.meters['base_loss'].global_avg
    avg_distillation_loss = metric_logger.meters['distillation_loss'].global_avg
    avg_alignment_loss = metric_logger.meters['alignment_loss'].global_avg  # Get alignment loss average

    print(f"Epoch {epoch} - Avg Base Loss: {avg_base_loss:.4f}, Avg Distillation Loss: {avg_distillation_loss:.4f}, Avg Alignment Loss: {avg_alignment_loss:.4f}")

    # Record epoch end time and calculate total duration
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time

    # Calculate and return throughput for the current epoch
    if total_samples > 0:
        epoch_avg_throughput = epoch_throughput / len(data_loader)  # Average throughput for current epoch
    else:
        epoch_avg_throughput = 0  # Avoid division by zero error
    print(f"Epoch {epoch} - Average Throughput: {epoch_avg_throughput:.2f} images/sec")

    # Return results including throughput
    return {
        **{k: meter.global_avg for k, meter in metric_logger.meters.items()},
        'throughput': epoch_avg_throughput  # Add throughput to the return value
    }


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch_idx, (images, target) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)

            # If the model returns a tuple, extract the classification result
            if isinstance(output, tuple):
                output = output[0]  # Take the first element (x, classification output)

            loss = criterion(output, target)


        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}