import os
import json
from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader

# baseline model
from src.model import BASELINE_MODEL
from src.utils import train, generate_dboxes, Encoder, BaseTransform
from src.loss import Loss
from src.dataset import collate_fn, Small_dataset, prepocessing

# nsml
import nsml
from nsml import DATASET_PATH

# only infer
def test_preprocessing(img, transform=None):
    # [참가자 TO-DO] inference를 위한 이미지 데이터 전처리
    if transform is not None:
        img = transform(img)
        img = img.unsqueeze(0)
    return img

def bind_model(model):
    def save(dir_path, **kwargs):
        checkpoint = {
            "model": model.state_dict()}
        torch.save(checkpoint, os.path.join(dir_path, 'model.pt'))
        print("model saved!")

    def load(dir_path):
        checkpoint = torch.load(os.path.join(dir_path, 'model.pt'))
        model.load_state_dict(checkpoint["model"])
        print('model loaded!')

    def infer(test_img_path_list): # data_loader에서 인자 받음
        '''
        반환 형식 준수해야 정상적으로 score가 기록됩니다.
        {'file_name':[[cls_num, x, y, w, h, conf]]}
        '''
        result_dict = {}

        # for baseline model ==============================
        import torchvision.transforms as transforms
        from PIL import Image
        from tqdm import tqdm

        infer_transforms = transforms.Compose([
                transforms.Resize((300,300)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        dboxes = generate_dboxes() 
        encoder = Encoder(dboxes) # inference시 박스 좌표로 후처리하는 모듈

        model.cuda()
        model.eval()

        for _, file_path in enumerate(tqdm(test_img_path_list)):
            file_name = file_path.split("/")[-1]
            img = Image.open(file_path)
            width, height = img.size

            img = test_preprocessing(img, infer_transforms)
            img = img.cuda()
            detections = []

            with torch.no_grad():
                ploc, plabel = model(img)
                ploc, plabel = ploc.float().detach().cpu(), plabel.float().detach().cpu()

                try:
                    result = encoder.decode_batch(ploc, plabel, 0.5, 100)[0]
                except:
                    print("No object detected : ", file_name)
                    continue

                loc, label, prob = [r.numpy() for r in result]
                for loc_, label_, prob_ in zip(loc, label, prob):
                    try:
                        '''
                        결과 기록 형식, 데이터 타입 준수해야 함
                        pred_cls, x, y, w, h, confidence
                        '''
                        detections.append([
                            int(label_)-1,
                            float( loc_[0] * width ), 
                            float( loc_[1] * height ), 
                            float( (loc_[2] - loc_[0]) * width ),
                            float( (loc_[3] - loc_[1]) * height ), 
                            float( prob_ )
                            ])
                    except:
                        continue

            result_dict[file_name] = detections # 반환 형식 준수해야 함
        return result_dict

    # DONOTCHANGE: They are reserved for nsml
    nsml.bind(save=save, load=load, infer=infer)

def get_args():
    parser = ArgumentParser(description="NSML BASELINE")
    parser.add_argument("--epochs", type=int, default=10, help="number of total epochs to run")
    parser.add_argument("--batch-size", type=int, default=8, help="number of samples for each iteration")
    parser.add_argument("--lr", type=float, default=0.001, help="initial learning rate")
    parser.add_argument("--nms-threshold", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)

    # DONOTCHANGE: They are reserved for nsml
    parser.add_argument("--pause", type=int, default=0)
    parser.add_argument('--mode', type=str, default='train', help='submit일때 test로 설정됩니다.')
    parser.add_argument('--iteration', type=str, default='0', help='fork 명령어를 입력할때의 체크포인트로 설정됩니다.')    
    args = parser.parse_args()
    return args

def main(opt):
    
    torch.manual_seed(123)
    num_class = 30 # 순수한 데이터셋 클래스 개수

    # baseline model
    dboxes = generate_dboxes()
    model = BASELINE_MODEL(num_classes=num_class+1) # 배경 class 포함 모델
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.937, 0.999))
    scheduler = None

    bind_model(model)

    if opt.pause:
        nsml.paused(scope=locals())
    else:
        # loss
        criterion = Loss(dboxes)

        # train data
        with open(os.path.join(DATASET_PATH, 'train', 'train_label'), 'r', encoding="utf-8") as f:
            train_data_dict = json.load(f)
            train_img_label = prepocessing(root_dir=os.path.join(DATASET_PATH, 'train', 'train_data'),\
                label_data=train_data_dict, input_size=(300,300))
        
        train_params = {"batch_size": opt.batch_size,
                        "shuffle": True,
                        "drop_last": False,
                        "num_workers": opt.num_workers,
                        "collate_fn": collate_fn}

        # data loader
        train_data = Small_dataset(train_img_label, num_class, BaseTransform(dboxes))
        train_loader = DataLoader(train_data, **train_params)

        model.cuda()
        criterion.cuda()

        for epoch in range(0, opt.epochs):
            train_loss = train(model, train_loader, epoch, criterion, optimizer, scheduler)
            nsml.report(
                epoch=epoch,
                epoch_total=opt.epochs,
                batch_size=opt.batch_size,
                train_loss=train_loss)
            nsml.save(epoch)

if __name__ == "__main__":
    opt = get_args()
    main(opt)