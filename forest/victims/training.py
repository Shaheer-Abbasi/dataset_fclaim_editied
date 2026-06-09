"""Repeatable code parts concerning optimization and training schedules."""


import torch
from torchvision import transforms

import datetime
from .utils import print_and_save_stats, pgd_step

from ..consts import NON_BLOCKING, BENCHMARK, DEBUG_TRAINING
torch.backends.cudnn.benchmark = BENCHMARK


# from utils.subset import get_coreset, get_random_subset
import numpy as np
import copy


import torch.nn.functional as F
import torch.nn as nn



def get_optimizers(model, args, defs):
    """Construct optimizer as given in defs."""
    if defs.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=defs.lr, momentum=0.9,
                                    weight_decay=defs.weight_decay, nesterov=True)
    elif defs.optimizer == 'SGD-basic':
        optimizer = torch.optim.SGD(model.parameters(), lr=defs.lr, momentum=0.0,
                                    weight_decay=defs.weight_decay, nesterov=False)
    elif defs.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=defs.lr, weight_decay=defs.weight_decay)

    if defs.scheduler == 'cyclic':
        effective_batches = (50_000 // defs.batch_size) * defs.epochs
        print(f'Optimization will run over {effective_batches} effective batches in a 1-cycle policy.')
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=defs.lr / 100, max_lr=defs.lr,
                                                      step_size_up=effective_batches // 2,
                                                      cycle_momentum=True if defs.optimizer in ['SGD'] else False)
    elif defs.scheduler == 'linear':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[defs.epochs // 2.667, defs.epochs // 1.6,
                                                                     defs.epochs // 1.142], gamma=0.1)
    elif defs.scheduler == 'none':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[10_000, 15_000, 25_000], gamma=1)

        # Example: epochs=160 leads to drops at 60, 100, 140.
    return optimizer, scheduler


def run_step(kettle, poison_delta, loss_fn, epoch, stats, model, defs, criterion, optimizer, scheduler, ablation=True):

    epoch_loss, total_preds, correct_preds = 0, 0, 0
    if DEBUG_TRAINING:
        data_timer_start = torch.cuda.Event(enable_timing=True)
        data_timer_end = torch.cuda.Event(enable_timing=True)
        forward_timer_start = torch.cuda.Event(enable_timing=True)
        forward_timer_end = torch.cuda.Event(enable_timing=True)
        backward_timer_start = torch.cuda.Event(enable_timing=True)
        backward_timer_end = torch.cuda.Event(enable_timing=True)

        stats['data_time'] = 0
        stats['forward_time'] = 0
        stats['backward_time'] = 0

        data_timer_start.record()

    if kettle.args.ablation < 1.0:
        # run ablation on a subset of the training set
        loader = kettle.partialloader
    else:
        loader = kettle.trainloader

    # use epic to remove effective poisoned data
    if True:
        for batch, (inputs, labels) in enumerate(loader):
            # Prep Mini-Batch
            model.train()
            optimizer.zero_grad()

            # if poison_delta == 'epic':
            #     kettle.args.times_selected[ids] += 1

            # Transfer to GPU
            inputs = inputs.to(**kettle.setup)
            labels = labels.to(dtype=torch.long, device=kettle.setup['device'], non_blocking=NON_BLOCKING)

            if DEBUG_TRAINING:
                data_timer_end.record()
                forward_timer_start.record()


            # Add data augmentation
            if defs.augmentations:  # defs.augmentations is actually a string, but it is False if --noaugment
                inputs = kettle.augment(inputs)

            # Does adversarial training help against poisoning?
            for _ in range(defs.adversarial_steps):
                inputs = pgd_step(inputs, labels, model, loss_fn, kettle.dm, kettle.ds,
                                    eps=kettle.args.eps, tau=kettle.args.tau)

            # Get loss
            outputs = model(inputs)
            loss = loss_fn(model, outputs, labels)
            if DEBUG_TRAINING:
                forward_timer_end.record()
                backward_timer_start.record()

            loss.backward()

            # Enforce batch-wise privacy if necessary
            # This is a defense discussed in Hong et al., 2020
            # We enforce privacy on mini batches instead of instances to cope with effects on batch normalization
            # This is reasonble as Hong et al. discuss that defense against poisoning mostly arises from the addition
            # of noise to the gradient signal
            with torch.no_grad():
                if defs.privacy['clip'] is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), defs.privacy['clip'])
                if defs.privacy['noise'] is not None:
                    # generator = torch.distributions.laplace.Laplace(torch.as_tensor(0.0).to(**kettle.setup),
                    #                                                 kettle.defs.privacy['noise'])
                    for param in model.parameters():
                        # param.grad += generator.sample(param.shape)
                        noise_sample = torch.randn_like(param) * defs.privacy['clip'] * defs.privacy['noise']
                        param.grad += noise_sample

            optimizer.step()

            predictions = torch.argmax(outputs.data, dim=1)
            total_preds += labels.size(0)
            correct_preds += (predictions == labels).sum().item()
            epoch_loss += loss.item()

            if DEBUG_TRAINING:
                backward_timer_end.record()
                torch.cuda.synchronize()
                stats['data_time'] += data_timer_start.elapsed_time(data_timer_end)
                stats['forward_time'] += forward_timer_start.elapsed_time(forward_timer_end)
                stats['backward_time'] += backward_timer_start.elapsed_time(backward_timer_end)

                data_timer_start.record()

            if defs.scheduler == 'cyclic':
                scheduler.step()
            if kettle.args.dryrun:
                break

        if defs.scheduler == 'linear':
            scheduler.step()

    if epoch % defs.validate == 0 or epoch == (defs.epochs - 1):
        valid_acc, valid_loss = run_validation(model, criterion, kettle.validloader, kettle.setup, kettle.args.dryrun)
    else:
        valid_acc, valid_loss = None, None

    current_lr = optimizer.param_groups[0]['lr']
    print_and_save_stats(epoch, stats, current_lr, epoch_loss / (batch + 1), correct_preds / total_preds,
                         valid_acc, valid_loss)

    if DEBUG_TRAINING:
        print(f"Data processing: {datetime.timedelta(milliseconds=stats['data_time'])}, "
              f"Forward pass: {datetime.timedelta(milliseconds=stats['forward_time'])}, "
              f"Backward Pass and Gradient Step: {datetime.timedelta(milliseconds=stats['backward_time'])}")
        stats['data_time'] = 0
        stats['forward_time'] = 0
        stats['backward_time'] = 0


def run_validation(model, criterion, dataloader, setup, dryrun=False):
    """Get accuracy of model relative to dataloader."""
    model.eval()
    correct = 0
    total = 0

    loss = 0
    with torch.no_grad():
        for i, (inputs, targets, _) in enumerate(dataloader):
            inputs = inputs.to(**setup)
            targets = targets.to(device=setup['device'], dtype=torch.long, non_blocking=NON_BLOCKING)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            loss += criterion(outputs, targets).item()
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            if dryrun:
                break
                
    accuracy = correct / total
    loss_avg = loss / (i + 1)
    return accuracy, loss_avg

