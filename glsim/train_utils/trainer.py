import os.path as osp
import sys
import statistics
from contextlib import suppress

import wandb
import numpy as np
import matplotlib.pyplot as plt
from einops import repeat, reduce, rearrange
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from timm.models import model_parameters

from .misc_utils import AverageMeter, accuracy, count_params_single
from .dist_utils import reduce_tensor, distribute_bn
from .mix import cutmix_data, mixup_data, mixup_criterion
from .scaler import NativeScaler
from .contrastive_loss import cont_loss, multi_cont_loss


class Trainer():
    def __init__(self, args, model, criterion, optimizer, lr_scheduler,
                 train_loader, val_loader, test_loader):
        self.args = args
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.model = model
        self.saved = False
        self.curr_iter = 0

        self.amp_autocast = torch.cuda.amp.autocast if args.fp16 else suppress
        self.loss_scaler = NativeScaler() if args.fp16 else None

    def train(self):
        val_acc = 0
        self.best_acc = 0
        self.best_epoch = 0
        self.max_memory = 0
        self.no_params = 0
        self.class_deviation = 0
        self.lr_scheduler.step(0)

        for epoch in range(self.args.epochs):
            self.epoch = epoch + 1

            if self.args.distributed or self.args.ra > 1:
                self.train_loader.sampler.set_epoch(epoch)

            train_acc, train_loss = self.train_epoch()

            if self.args.local_rank == 0 and \
                    ((self.epoch % self.args.eval_freq == 0) or (self.epoch == self.args.epochs)):
                val_acc, val_loss = self.validate_epoch(self.val_loader)

                if self.args.debugging:
                    return None, None, None, None, None

                self.epoch_end_routine(train_acc, train_loss, val_acc, val_loss)

        if self.args.local_rank == 0:
            self.train_end_routine(val_acc)

        return self.best_acc, self.best_epoch, self.max_memory, self.no_params, self.class_deviation

    def prepare_batch(self, batch):
        images, targets = batch
        if self.args.distributed:
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
        else:
            images = images.to(self.args.device, non_blocking=True)
            targets = targets.to(self.args.device, non_blocking=True)
        return images, targets

    def predict(self, images, targets, train=True):
        if self.args.cm or self.args.mu:
            r = np.random.rand(1)
            if r < self.args.mix_prob and train:
                images, y_a, y_b, lam = self.prepare_mix(images, targets)

        image_size = self.args.image_size

        if (self.args.save_images and train and (self.curr_iter % self.args.save_images == 0)):
            self.save_images(images, osp.join(self.args.results_dir, f'{self.curr_iter}.png'), image_size)

        with self.amp_autocast():
            if self.args.model_name == 'cal':
                output, loss, images_crops = self.model(images, targets, train=train)
                if (self.args.save_images and train and (self.curr_iter % self.args.save_images == 0)):
                    self.save_images(
                        images_crops, osp.join(self.args.results_dir, f'{self.curr_iter}_crops.png'),
                        self.args.image_size)
                elif self.args.save_images and not train and not self.saved:
                    self.save_images(images, osp.join(self.args.results_dir, f'test.png'), image_size)
                    self.save_images(
                        images_crops, osp.join(self.args.results_dir, f'test_crops.png'), self.args.image_size)
                    if not self.args.debugging and not self.args.vis_errors and not self.args.test_offline:
                        wandb.log({'crops': wandb.Image(images_crops)})
                    self.saved = True
                return output, loss

            output = self.model(images, targets, train=train)

            if self.args.classifier_aux:
                output, output_aux = output
            elif self.args.anchor_size and isinstance(output, tuple):
                crops = True
                output, images_crops = output
            else:
                crops = False

            if self.args.num_anchors:
                r = self.args.seq_len_post_reducer

            if (self.args.cm or self.args.mu) and r < self.args.mix_prob and train:
                loss = mixup_criterion(self.criterion, output, y_a, y_b, lam)
            else:
                loss = self.criterion(output, targets)

            if self.args.classifier_aux:
                if self.args.classifier_aux == 'cont':
                    loss_aux = cont_loss(output_aux, targets)
                elif self.args.classifier_aux == 'multi_cont':
                    if self.args.supcon:
                        loss_aux = multi_cont_loss(output_aux, targets, norm_ind=self.args.norm_ind)
                    else:
                        loss_aux = multi_cont_loss(output_aux, norm_ind=self.args.norm_ind)
                loss = loss + (self.args.loss_aux_weight * loss_aux)

        if self.args.anchor_size and crops:
            resize_size = self.args.anchor_resize_size if self.args.anchor_resize_size else self.args.image_size
            if (self.args.save_images and train and (self.curr_iter % self.args.save_images == 0)):
                self.save_images(
                    images_crops, osp.join(self.args.results_dir, f'{self.curr_iter}_crops.png'), resize_size)
            elif self.args.save_images and not train and not self.saved:
                self.save_images(images, osp.join(self.args.results_dir, f'test.png'), image_size)
                self.save_images(images_crops, osp.join(self.args.results_dir, f'test_crops.png'), resize_size)
                if not self.args.debugging and not self.args.vis_errors and not self.args.test_offline:
                    wandb.log({'crops': wandb.Image(images_crops)})
                self.saved = True
        elif self.args.save_images and not train and not self.saved:
            self.save_images(images, osp.join(self.args.results_dir, f'test.png'), image_size)
            if not self.args.debugging and not self.args.vis_errors and not self.args.test_offline:
                wandb.log({'crops': wandb.Image(images)})
            self.saved = True

        return output, loss

    def prepare_mix(self, images, targets):
        # cutmix and mixup
        if self.args.cm and self.args.mu:
            switching_prob = np.random.rand(1)
            # Cutmix
            if switching_prob < 0.5:
                slicing_idx, y_a, y_b, lam, sliced = cutmix_data(images, targets, self.args)
                images[:, :, slicing_idx[0]:slicing_idx[2], slicing_idx[1]:slicing_idx[3]] = sliced
            # Mixup
            else:
                images, y_a, y_b, lam = mixup_data(images, targets, self.args)
        # cutmix only
        elif self.args.cm:
            slicing_idx, y_a, y_b, lam, sliced = cutmix_data(images, targets, self.args)
            images[:, :, slicing_idx[0]:slicing_idx[2], slicing_idx[1]:slicing_idx[3]] = sliced
        # mixup only
        elif self.args.mu:
            images, y_a, y_b, lam = mixup_data(images, targets, self.args)
        return images, y_a, y_b, lam

    def train_epoch(self):
        """vanilla training"""
        self.model.train()

        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        for idx, batch in enumerate(self.train_loader):
            images, targets = self.prepare_batch(batch)
            output, loss = self.predict(images, targets, train=True)

            acc1, acc5 = accuracy(output, targets, topk=(1, 5))

            # ===================backward=====================
            if self.args.gradient_accumulation_steps > 1:
                with self.amp_autocast():
                    loss = loss / self.args.gradient_accumulation_steps
            if self.loss_scaler is not None:
                self.loss_scaler.scale_loss(loss)

            if (idx + 1) % self.args.gradient_accumulation_steps == 0:
                if self.loss_scaler is not None:
                    self.loss_scaler(self.optimizer, clip_grad=self.args.clip_grad,
                                     parameters=model_parameters(self.model))
                else:
                    loss.backward()
                    self.optimizer.step()
                self.optimizer.zero_grad()
                self.lr_scheduler.step_update(num_updates=self.curr_iter)

            # ===================meters=====================
            torch.cuda.synchronize()

            if self.args.distributed:
                reduced_loss = reduce_tensor(loss.data, self.args.world_size)
                acc1 = reduce_tensor(acc1, self.args.world_size)
                acc5 = reduce_tensor(acc5, self.args.world_size)
            else:
                reduced_loss = loss.data

            losses.update(reduced_loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            self.curr_iter += 1

            # print info
            if idx % self.args.log_freq == 0 and self.args.local_rank == 0:
                lr_curr = self.optimizer.param_groups[0]['lr']
                print(
                    'Epoch: [{0}/{1}][{2}/{3}]\t'
                    'LR: {4}\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        self.epoch, self.args.epochs, idx, len(self.train_loader), lr_curr,
                        loss=losses, top1=top1, top5=top5))
                sys.stdout.flush()

            if self.args.debugging:
                return None, None

        if self.args.local_rank == 0:
            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

        if self.args.distributed:
            distribute_bn(self.model, self.args.world_size, True)

        self.lr_scheduler.step(self.epoch)

        return round(top1.avg, 2), round(losses.avg, 3)

    def validate_epoch(self, val_loader):
        """validation"""
        # switch to evaluate mode
        self.model.eval()

        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        if self.epoch == self.args.epochs:
            self.curr_img = 0
            class_correct = [0 for _ in range(self.args.num_classes)]
            class_total = [0 for _ in range(self.args.num_classes)]

        with torch.no_grad():
            for idx, batch in enumerate(val_loader):
                images, targets = self.prepare_batch(batch)
                output, loss = self.predict(images, targets, train=False)

                acc1, acc5 = accuracy(output, targets, topk=(1, 5))

                torch.cuda.synchronize()

                if self.epoch == self.args.epochs:
                    self.calc_per_class_correct(output, targets, class_correct, class_total, images)

                reduced_loss = loss.data

                losses.update(reduced_loss.item(), images.size(0))
                top1.update(acc1.item(), images.size(0))
                top5.update(acc5.item(), images.size(0))

                if idx % self.args.log_freq == 0 and self.args.local_rank == 0:
                    print('Val: [{0}/{1}]\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                              idx, len(val_loader),
                              loss=losses, top1=top1, top5=top5))

                if self.args.debugging:
                    return None, None

        if self.epoch == self.args.epochs:
            self.calc_class_deviation(class_correct, class_total)

        if self.args.local_rank == 0:
            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

        return round(top1.avg, 2), round(losses.avg, 3)

    def epoch_end_routine(self, train_acc, train_loss, val_acc, val_loss):
        lr_curr = self.optimizer.param_groups[0]['lr']
        print("Training...Epoch: {} | LR: {}".format(self.epoch, lr_curr))
        log_dic = {'epoch': self.epoch, 'lr': lr_curr,
                   'train_acc': train_acc, 'train_loss': train_loss,
                   'val_acc': val_acc, 'val_loss': val_loss}
        # if hasattr(self, 'samples'):
        #    log_dic['samples'] = wandb.Image(self.samples)
        wandb.log(log_dic)

        # save the best model
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            self.best_epoch = self.epoch
            self.save_model(self.best_epoch, val_acc, mode='best')
        # regular saving
        if self.epoch % self.args.save_freq == 0:
            self.save_model(self.epoch, val_acc, mode='epoch')

    def train_end_routine(self, val_acc):
        # save last
        self.save_model(self.epoch, val_acc, mode='last')
        # VRAM and No. of params
        self.computation_stats()

    def computation_stats(self):
        # VRAM memory consumption
        self.max_memory = torch.cuda.max_memory_reserved() / (1024 ** 3)
        # summary stats
        self.no_params = count_params_single(self.model)

    def save_model(self, epoch, acc, mode):
        state = {
            'config': self.args,
            'epoch': epoch,
            'model': self.model.state_dict(),
            'accuracy': acc,
            'optimizer': self.optimizer.state_dict(),
        }

        if mode == 'best':
            save_file = osp.join(self.args.results_dir, f'{self.args.model_name}_best.pth')
            print('Saving the best model!')
            torch.save(state, save_file)
        elif mode == 'epoch':
            save_file = osp.join(self.args.results_dir, f'ckpt_epoch_{epoch}.pth')
            print('==> Saving each {} epochs...'.format(self.args.save_freq))
            torch.save(state, save_file)
        elif mode == 'last':
            save_file = osp.join(self.args.results_dir, f'{self.args.model_name}_last.pth')
            print('Saving last epoch')
            torch.save(state, save_file)

    def test(self):
        print(f'Evaluation on test dataloader: ')
        self.epoch = self.args.epochs
        test_acc, _ = self.validate_epoch(self.test_loader)
        self.computation_stats()

        if self.args.test_multiple:
            self.epoch = 0
            for i in range(self.args.test_multiple):
                print(f'Testing multiple times: {i}/{self.args.test_multiple}')
                test_acc, _ = self.validate_epoch(self.test_loader)

        if self.args.debugging:
            return None, None, None, None
        return test_acc, self.max_memory, self.no_params, self.class_deviation

    def save_images(self, images, fp, image_size):
        with torch.no_grad():
            samples = (images.reshape(-1, 3, image_size, image_size).data + 1) / 2.0
            save_image(samples, fp, nrow=int(np.sqrt(samples.shape[0])))
        return 0

    def calc_per_class_correct(self, output, targets, class_correct, class_total, images):
        _, predicted = torch.max(output.data, 1)
        c = (predicted == targets)
        for i, target in enumerate(targets):
            class_correct[target] += c[i].item()
            class_total[target] += 1

            if self.args.vis_errors and not c[i].item():
                prob = torch.softmax(output, -1)[i, predicted[i]].item() * 100
                title = f'Current image: {self.curr_img}\
                          Prediction: {predicted[i].item()} ({prob:.2f}%)\
                          Correct: {target.item()}'
                print(title)
                self.imshow(images, title=title)

            self.curr_img += 1

        return 0

    def calc_class_deviation(self, class_correct, class_total):
        # per class accuracy
        per_class_accuracy = []
        for i in range(self.args.num_classes):
            per_class_accuracy.append(100 * class_correct[i] / class_total[i])

        class_mean = round(statistics.mean(per_class_accuracy), 2)
        self.class_deviation = round(statistics.stdev(per_class_accuracy), 2)
        print(f'Per-class mean accuracy: {class_mean}%\nClass deviation: {self.class_deviation}%')
        return 0

    def imshow(self, img, title=None):
        img = rearrange(img, '1 c h w -> h w c')
        img = (img.data + 1) / 2.0
        img = np.uint8(np.clip(img.to('cpu').numpy(), 0, 1) * 255)

        fig = plt.figure()
        axs = fig.add_subplot(111)

        axs.imshow(img)
        if title is not None:
            axs.set_title(title, fontsize=8, wrap=True)

        axs.axis('off')
        fig.tight_layout()

        if self.args.vis_errors_save:
            fp = osp.join(self.args.results_dir, f'test_{self.curr_img}.png')
            fig.savefig(fp, dpi=300)
        else:
            plt.show(block=False)

            inp = input("input 'exit' to stop visualizing: ")
            if inp == 'exit':
                sys.exit()

        plt.close()
        return 0
