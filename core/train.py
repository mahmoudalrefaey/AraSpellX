from pathlib import Path
from typing import Union, List
from core.args import get_train_args
from core.callback import TermCallback, get_callback
from data.data import get_train_test_loaders
from core.interfaces import ILogger
from core.logger import get_logger
from models.loss import get_criterion
from models.models import get_model
from models.optimizer import get_optimizer
from data.tokenizer import get_tokenizer
from tqdm import tqdm
import torch
import os
import glob
from core.utils import load_state
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast


class DistTrainer:
    _train_loss_key = 'train_loss'
    _acc_train_loss_key = 'acc_train_loss'
    _test_loss_key = 'test_loss'
    _acc_test_loss_key = 'acc_test_loss'

    def __init__(
            self,
            train_loader,
            test_loader,
            model,
            criterion,
            optimizer,
            epochs: int,
            callback: TermCallback,
            logger: ILogger,
            outdir: Union[str, Path],
            rank: int = 0,
            clip_grad: bool = False,
            grad_norm=None,
            ckpt=None,
            mixed_precision: bool = False,
            grad_accum_steps: int = 1,
            ckpt_interval_steps: int = 5000,
            max_checkpoints: int = 5
            ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required for training, but CUDA is not available on this machine.')
        self.callback = callback
        self.logger = logger
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.rank = rank
        self.device = torch.device('cuda:0')
        self.model.to(self.device)
        self.__counter = 0
        self.outdir = Path(outdir)
        self.last_epoch = 0
        self.logger.set_rank(self.rank)
        if ckpt is not None:
            self._set_state(ckpt)
        self.history = dict()
        self.grad_norm = grad_norm
        self.clip_grad = clip_grad
        self.mixed_precision = mixed_precision
        self.scaler = GradScaler() if mixed_precision else None
        self.grad_accum_steps = max(1, grad_accum_steps)
        self.ckpt_interval_steps = ckpt_interval_steps
        self.max_checkpoints = max_checkpoints
        self.global_step = 0

    def _set_state(self, ckpt_path):
        model, optimizer, epoch, steps = load_state(ckpt_path)
        self.model.load_state_dict(model)
        self.optimizer.load_state_dict(optimizer, steps)
        self.last_epoch = epoch + 1

    @property
    def is_master(self):
        return self.rank == 0

    def log_results(self, epoch: int):
        if self._train_loss_key in self.history:
            self.logger.log(
                key=self._acc_train_loss_key,
                value=self.history[self._train_loss_key][-1],
                step=epoch,
                end=''
            )
        if self._test_loss_key in self.history:
            self.logger.log(
                key=self._acc_test_loss_key,
                value=self.history[self._test_loss_key][-1]
            )

    def set_train_mode(self) -> None:
        self.model = self.model.train()

    def set_test_mode(self) -> None:
        self.model = self.model.eval()

    def _get_ckpt_state(self, epoch: int):
        return {
            'model': self.model.state_dict(),
            'epoch': epoch,
            'optimizer': self.optimizer.state_dict(),
            'steps': self.optimizer.counter,
            'scaler': self.scaler.state_dict() if self.scaler else None
        }

    def _get_checkpoint_files(self) -> List[Path]:
        """Get all checkpoint files sorted by modification time (oldest first)."""
        pattern = self.outdir / 'checkpoint_epoch_*.pt'
        files = sorted(glob.glob(str(pattern)), key=os.path.getmtime)
        return [Path(f) for f in files]

    def _cleanup_old_checkpoints(self) -> None:
        """Remove oldest checkpoints if we exceed max_checkpoints."""
        checkpoint_files = self._get_checkpoint_files()
        while len(checkpoint_files) > self.max_checkpoints:
            oldest = checkpoint_files.pop(0)
            try:
                oldest.unlink()
                print(f'Removed old checkpoint: {oldest}')
            except OSError as e:
                print(f'Warning: Could not remove {oldest}: {e}')

    def save_ckpt(self, epoch: int) -> None:
        state = self._get_ckpt_state(epoch)
        path = self.outdir / f'checkpoint_epoch_{epoch}_step_{self.global_step}.pt'
        torch.save(state, path)
        print(f'Checkpoint saved: {path}')
        self._cleanup_old_checkpoints()

    def test_and_log(self, epoch):
        if self.is_master:
            self.test()
            self.log_results(epoch)
            save_ckpt, terminate = self.callback(
                self.history[self._test_loss_key][-1]
                )
            if epoch == -1:
                return
            if save_ckpt is True:
                self.save_ckpt(epoch)
            if terminate is True:
                print('The model is not improving any more!')
                print('terminated!')
                exit()

    def fit(self, *args, **kwargs):
        self.test_and_log(-1)
        for epoch in range(self.last_epoch, self.epochs):
            self.last_epoch = epoch
            self.train()
            self.test_and_log(epoch)

    @torch.no_grad()
    def test(self):
        total_loss = []
        self.set_test_mode()
        for batch in tqdm(self.test_loader):
            self.__counter += 1
            (enc_inp, dec_inp, enc_mask, dec_mask) = batch
            enc_inp = enc_inp.to(self.device, non_blocking=True)
            dec_inp = dec_inp.to(self.device, non_blocking=True)
            enc_mask = enc_mask.to(self.device, non_blocking=True)
            dec_mask = dec_mask.to(self.device, non_blocking=True)
            preds, att = self.model(enc_inp, dec_inp, enc_mask, dec_mask, need_weights=True)
            loss = self.criterion(preds, dec_inp, dec_mask)
            total_loss.append(loss.item())
        total_loss = sum(total_loss)
        total_loss /= len(self.test_loader)
        if self._test_loss_key in self.history:
            self.history[self._test_loss_key].append(total_loss)
        else:
            self.history[self._test_loss_key] = [total_loss]
        h = att.shape[0] // dec_inp.shape[0]
        self.logger.log_img('enc_dec_att', att[:h, ...])

    def train(self):
        total_loss = 0
        self.set_train_mode()
        accum_step = 0
        
        # Create progress bar that updates every step
        pbar = tqdm(self.train_loader, total=len(self.train_loader), 
                    desc=f'Epoch {self.last_epoch}', unit='step',
                    dynamic_ncols=True, leave=False)
        
        for batch in pbar:
            (enc_inp, dec_inp, enc_mask, dec_mask) = batch
            enc_inp = enc_inp.to(self.device, non_blocking=True)
            dec_inp = dec_inp.to(self.device, non_blocking=True)
            enc_mask = enc_mask.to(self.device, non_blocking=True)
            dec_mask = dec_mask.to(self.device, non_blocking=True)
            
            if self.mixed_precision:
                with autocast():
                    preds, att = self.model(enc_inp, dec_inp, enc_mask, dec_mask, need_weights=False)
                    loss = self.criterion(preds, dec_inp, dec_mask)
                    loss = loss / self.grad_accum_steps
                self.scaler.scale(loss).backward()
            else:
                preds, att = self.model(enc_inp, dec_inp, enc_mask, dec_mask, need_weights=False)
                loss = self.criterion(preds, dec_inp, dec_mask)
                loss = loss / self.grad_accum_steps
                loss.backward()
            
            accum_step += 1
            
            if accum_step % self.grad_accum_steps == 0:
                if self.mixed_precision:
                    if self.clip_grad:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    if self.clip_grad:
                        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_norm)
                    self.optimizer.step()
                self.optimizer.zero_grad()
            
            # Log the unscaled loss
            log_loss = loss.item() * self.grad_accum_steps
            self.logger.log_step(self._train_loss_key, log_loss)
            total_loss += log_loss
            
            # Increment global step and update progress bar
            self.global_step += 1
            pbar.set_postfix({
                'loss': f'{log_loss:.4f}',
                'avg_loss': f'{total_loss / (pbar.n + 1):.4f}',
                'step': self.global_step,
                'epoch': self.last_epoch
            })
            
            # Checkpoint every ckpt_interval_steps
            if self.global_step % self.ckpt_interval_steps == 0 and self.is_master:
                self.save_ckpt(self.last_epoch)
        
        pbar.close()
        
        if self.rank == 0:
            total_loss /= len(self.train_loader)
            if self._train_loss_key in self.history:
                self.history[self._train_loss_key].append(total_loss)
            else:
                self.history[self._train_loss_key] = [total_loss]


def get_trainer(rank: int, args):
    callback = get_callback(args)
    logger = get_logger(args)
    tokenizer = get_tokenizer(args)
    vocab_size = tokenizer.vocab_size
    train_loader, test_loader = get_train_test_loaders(args, rank, tokenizer)
    model = get_model(
        args,
        rank,
        vocab_size,
        pad_idx=tokenizer.special_tokens.pad_id
        )
    criterion = get_criterion(args, vocab_size)
    optimizer = get_optimizer(args, model.parameters())
    return DistTrainer(
        train_loader=train_loader,
        test_loader=test_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        epochs=args.epochs,
        callback=callback,
        logger=logger,
        outdir=args.outdir,
        rank=rank,
        ckpt=args.pre_trained_path,
        grad_norm=args.grad_norm,
        clip_grad=args.clip_grad,
        mixed_precision=getattr(args, 'mixed_precision', False),
        grad_accum_steps=getattr(args, 'grad_accum_steps', 1),
        ckpt_interval_steps=getattr(args, 'ckpt_interval_steps', 5000),
        max_checkpoints=getattr(args, 'max_checkpoints', 5)
    )


def main(args):
    if args.n_gpus != 1:
        raise ValueError('This project supports single-GPU CUDA training only.')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for training, but CUDA is not available on this machine.')
    trainer = get_trainer(rank=0, args=args)
    trainer.fit()