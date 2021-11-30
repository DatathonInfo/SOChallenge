
from torch.nn.utils.clip_grad import clip_grad_norm_
import torch
import torch.nn as nn 
import numpy as np
from tqdm import tqdm 
from collections import OrderedDict
from src.utils import AverageMeter, OCRLabelConverter


class OCRTrainer(object):
    def __init__(self, opt):
        super(OCRTrainer, self).__init__()
        self.data_train = opt.data_train
        self.data_val = opt.data_val
        self.model = opt.model
        self.criterion = opt.criterion
        self.optimizer = opt.optimizer
        self.converter = OCRLabelConverter(opt.alphabet)
        self.batch_size = opt.batch_size
        self.count = opt.epoch
        self.epochs = opt.epochs
        self.cuda = opt.cuda
        self.collate_fn = opt.collate_fn
        self.init_meters()

    def init_meters(self):
        self.avgTrainLoss = AverageMeter("Train loss")
        self.avgValLoss = AverageMeter("Validation loss")

    def forward(self, x):
        logits = self.model(x)
        return logits.transpose(1, 0)

    def loss_fn(self, logits, targets, pred_sizes, target_sizes):
        loss = self.criterion(logits, targets, pred_sizes, target_sizes)
        return loss

    def step(self):
        self.max_grad_norm = 0.05
        clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
    

    def _run_batch(self, batch, report_accuracy=False, validation=False):
        input_, targets = batch['img'].cuda(), batch['label']
        targets, lengths = self.converter.encode(targets)
        logits = self.forward(input_)
        logits = logits.contiguous().cpu()
        logits = torch.nn.functional.log_softmax(logits, 2)

        T, B, H = logits.size()
        pred_sizes = torch.LongTensor([T for i in range(B)])
        targets= targets.view(-1).contiguous()
        loss = self.loss_fn(logits, targets, pred_sizes, lengths)
        if report_accuracy:
            probs, preds = logits.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
        return loss 

    def _run_epoch(self, validation=False):
        if not validation:
            loader = self.train_dataloader()
            pbar = tqdm(loader, desc='Epoch: [%d]/[%d] Training'%(self.count + 1, self.epochs), leave=True)
            self.model.train()
        else:
            loader = self.val_dataloader()
            pbar = tqdm(loader, desc='Validating', leave=True)
            self.model.eval()
        outputs = []
        for batch_nb, batch in enumerate(pbar):
            if not validation:
                output = self.training_step(batch)
            else:
                output = self.validation_step(batch)
            pbar.set_postfix(output)
            outputs.append(output)
        if not validation:
            result = self.train_end(outputs)
        else:
            result = self.validation_end(outputs)
        return result

    def training_step(self, batch):
        loss = self._run_batch(batch, report_accuracy=True)
        self.optimizer.zero_grad()
        
        try:
            loss.backward()
        except Exception as e:
            print(loss)
            return OrderedDict({
            'loss': 0,
            })
        self.step()
        output = OrderedDict({
            'loss': abs(loss.item()),
            })
        return output

    def validation_step(self, batch):
        loss = self._run_batch(batch, report_accuracy=True, validation=True)
        output = OrderedDict({
            'val_loss': abs(loss.item()),
            })
        return output

    def train_dataloader(self):
        loader = torch.utils.data.DataLoader(self.data_train,
                batch_size=self.batch_size,
                collate_fn=self.collate_fn,
                shuffle=True)
        return loader
        
    def val_dataloader(self):
        loader = torch.utils.data.DataLoader(self.data_val,
                batch_size=self.batch_size,
                collate_fn=self.collate_fn)
        return loader

    def train_end(self, outputs):
        for output in outputs:
            self.avgTrainLoss.add(output['loss'])
        train_loss_mean = abs(self.avgTrainLoss.compute())
        result = {'train_loss': train_loss_mean}
        return result

    def validation_end(self, outputs):
        for output in outputs:
            self.avgValLoss.add(output['val_loss'])
        val_loss_mean = abs(self.avgValLoss.compute())
        result = {'val_loss': val_loss_mean}
        return result