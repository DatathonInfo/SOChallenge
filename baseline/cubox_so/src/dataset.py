import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import os
import numpy as np
from PIL import Image
import pickle
from tqdm import tqdm

def collate_fn(batch):
    items = list(zip(*batch))
    items[0] = default_collate([i for i in items[0] if torch.is_tensor(i)])
    items[1] = list([i for i in items[1] if i])
    items[2] = list([i for i in items[2] if i])
    items[3] = default_collate([i for i in items[3] if torch.is_tensor(i)])
    items[4] = default_collate([i for i in items[4] if torch.is_tensor(i)])
    return items

# [참가자 TO-DO] custom dataset
class Small_dataset(Dataset):
    def __init__(self, label_data, num_classes=80, transform=None):
        self.label_data = label_data
        self.num_class = num_classes + 1 # 배경 포함 모델
        self.transform = transform

        self.img_list = self.label_data['data']
        self.label_list = self.label_data['label']

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        image = self.img_list[index].copy()
        label = self.label_list[index]

        height, width = label[0]
        labels = label[1].clone()
        boxes = label[2].clone()

        # [참가자 TO-DO] 모델에 맞는 전처리 코드로 채우면 됩니다. 
        if self.transform is not None:
            image, (height, width), boxes, labels =\
                self.transform(image, (height, width), boxes, labels)
        return image, int(index), (height, width), boxes, labels

# [참가자 TO-DO] 효율적인 훈련 시간을 위한 preprocessing / 사용하지 않아도 무방합니다.
def prepocessing(root_dir, label_data, input_size):
    img_file_list = list(label_data.keys())
    img_list = []
    label_list = []

    print("Starting Caching...")
    for idx, cur_file in enumerate(tqdm(img_file_list)):
        image = Image.open(os.path.join(root_dir, cur_file))
        width, height = image.size
        img_list.append(image.resize(input_size))

        # 원본 레이블 형식 list [cls, x, y, w, h]
        cur_label = np.array(label_data[cur_file])

        boxes = []
        labels = []
        
        for label in cur_label:
            # 네트워크에 따라 cls index를 1씩(background) 미룹니다.
            labels.append(int(label[0]+1))

            # model 형식에 맞게 변환 필요 (ex. xywh -> (normalized left,top,right,bottom)
            boxes.append([label[1] / width, label[2] / height, 
                (label[1] + label[3]) / width, (label[2]+label[4]) / height])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels)

        label_list.append(((height, width), labels, boxes))
    
    data_dict = {
        'data':img_list,
        'label':label_list  
    }

    return data_dict