import os 
import random
import torch  
from torch.utils.data import random_split
from argparse import ArgumentParser
import numpy as np
from tqdm import *

from src.ocr_trainer import OCRTrainer
from src.model import CRNN
from src.dataset import CustomDataset, SynthCollator, CustomDataset_infer, SynthCollator_infer
from src.utils import OCRLabelConverter
from src.loss import CustomCTCLoss

import nsml
from nsml import DATASET_PATH


def bind_model(model):
    def load(filename, **kwargs):
        state = torch.load(os.path.join(filename, 'model.pt'))
        model.load_state_dict(state['model'])
        print('Model loaded')

    def save(filename, **kwargs):
        state = {
            'model': model.state_dict(),
        }
        torch.save(state, os.path.join(filename, 'model.pt'))

    def infer(data_path): # 해당부분은 test mode시 infer_func을 의미합니다

        data = CustomDataset_infer(data_path)
        loader = torch.utils.data.DataLoader(data,
            batch_size=8,
            collate_fn=SynthCollator_infer())
    
        net = model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        net.to(device)
        net.eval()
        math_list = []
        with open('train_label', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            math_list.extend(lines) 
        math_list = ''.join(math_list)
        alphabet = ''.join(set(math_list))
        alphabet = ''.join(sorted(alphabet))   # 훈련시 사용한  알파벳리스트와 동일해야 합니다

        converter = OCRLabelConverter(alphabet) 
        predictions = []
        for iteration, batch in enumerate(tqdm(loader)):
            input_ = batch['img'].to(device) 
            logits = net(input_).transpose(1, 0)
            logits = torch.nn.functional.log_softmax(logits, 2)
            logits = logits.contiguous().cpu()
            T, B, H = logits.size()
            pred_sizes = torch.LongTensor([T for i in range(B)])
            probs, pos = logits.max(2)
            pos = pos.transpose(1, 0).contiguous().view(-1)
            sim_preds = converter.decode(pos.data, pred_sizes.data, raw=False)
            predictions.extend(sim_preds)
            
        return predictions  # test시 사용합니다. ['word1', 'word2', 'word3'...] 순서, 양식 준수하셔야 합니다!

    nsml.bind(save=save, load=load, infer=infer)


class Learner(object):
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            self.model = self.model.cuda() 
        self.epoch = 0

    def fit(self, opt):
        opt.cuda = self.cuda
        opt.model = self.model
        opt.optimizer = self.optimizer
        opt.epoch = self.epoch
        trainer = OCRTrainer(opt)

        for epoch in range(opt.epoch, opt.epochs):
            train_result = trainer._run_epoch()
            val_result = trainer._run_epoch(validation=True)
            trainer.count = epoch + 1
            self.val_loss = val_result['val_loss']
            print('val_loss:', self.val_loss)

            nsml.report(
                summary=True,
                epoch=epoch,
                epoch_total=args.epochs,
                train_loss=train_result['train_loss'],
            )
            nsml.save(epoch)

if __name__ == "__main__":

    torch.backends.cudnn.deterministic = True 
    torch.manual_seed(71)
    np.random.seed(71)
    random.seed(71)

    parser = ArgumentParser() 
    if nsml.IS_ON_NSML:
        parser.add_argument("--path", type=str, default=DATASET_PATH)
        parser.add_argument("--train_label_path", type=str, default=DATASET_PATH + '/train/train_label')
    else:
        parser.add_argument("--path", type=str, default='./data')
        parser.add_argument("--train_label_path", type=str, default='./data/train/train_label')
    parser.add_argument("--name", type=str, default='exp1')
    parser.add_argument("--imgdir", type=str, default='train/train_data')  
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--imgH", type=int, default=32)
    parser.add_argument("--nHidden", type=int, default=256)
    parser.add_argument("--nChannels", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=2)  
    

#######################################################################
    # DONOTCHANGE: They are reserved for nsml
    parser.add_argument("--mode", type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.') 
    parser.add_argument("--pause", type=int, default=0)
#######################################################################


    args = parser.parse_args()

    if args.mode == 'test':
        args.imgdir = 'test/test_data'

    if nsml.IS_ON_NSML:
        train_label_path = DATASET_PATH + '/train/train_label'
    else:
        train_label_path = './data/train/train_label'

    data = CustomDataset(args)
    args.collate_fn = SynthCollator() 
    train_split = int(0.9*len(data))
    val_split = len(data) - train_split
    args.data_train, args.data_val = random_split(data, (train_split, val_split))
    print('Traininig Data Size:{}\nVal Data Size:{}'.format(
        len(args.data_train), len(args.data_val)))
    
    math_list = []
    with open('train_label', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        math_list.extend(lines)
    math_list = ''.join(math_list)
    alphabet = ''.join(set(math_list))
    alphabet = ''.join(sorted(alphabet)) # nClasses=84 

    args.alphabet = alphabet
    args.nClasses = len(args.alphabet)
    model = CRNN(args)
    args.criterion = CustomCTCLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    learner = Learner(model, optimizer)

    bind_model(model)
    if args.pause:
        nsml.paused(scope=locals()) 

    learner.fit(args)

