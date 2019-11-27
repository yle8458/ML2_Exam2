import torch
import torch.nn as nn

import os
import torchvision.transforms as T
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
from torchvision import models
from torch import optim
import numpy as np

IMG_COL = 800
IMG_ROW = 600
batch_size = 8
N_LABEL = 7
resnet_cls = models.resnet50()

class AvgPool(nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, x.shape[2:])


class ResNet50(nn.Module):
    def __init__(self, num_outputs):
        super(ResNet50, self).__init__()
        self.resnet = resnet_cls
        layer4 = self.resnet.layer4
        self.resnet.layer4 = nn.Sequential(
            nn.Dropout(0.5),
            layer4
        )
        self.resnet.avgpool = AvgPool()
        self.resnet.fc = nn.Linear(2048, num_outputs)
        for param in self.resnet.parameters():
            param.requires_grad = False

        for param in self.resnet.layer4.parameters():
            param.requires_grad = True

        for param in self.resnet.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        out = self.resnet(x)
        return out

# dataset class
class exam_dataset(torch.utils.data.Dataset):
    def __init__(self, img_list, transforms = None):
        self.img_list = sorted(img_list)
        self.transforms = transforms
        self.n_imgs = len(self.img_list)


    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        img = Image.open(img_path).convert('RGB')
        #img = cv2.resize(img, (self.ncol, self.nrow), interpolation=cv2.INTER_AREA)
        #img = np.moveaxis(img, -1, 0)

        if self.transforms is not None:
            img = self.transforms(img)

        return img

    def __len__(self):
        return self.n_imgs

def get_transform(train):
    transforms = []
    transforms.append(T.Resize((600, 800), interpolation=Image.LANCZOS))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomVerticalFlip(0.5))
        transforms.append(T.RandomRotation(90))
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return T.Compose(transforms)

res50 = ResNet50(num_outputs=N_LABEL)

def predict(test_files):
    results = []

    res50.cuda()
    res50.load_state_dict(torch.load('model_ppzt.pt'))
    res50.eval()
    dataset = exam_dataset(test_files, get_transform(train = False))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = False, drop_last = False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for batch_img in dataloader:
        batch_img = batch_img.to(device)
        eval_out = res50(batch_img)
        activation = nn.Sigmoid()
        eval_out = activation(eval_out)
        eval_out = eval_out.cpu().detach()
        eval_out.view(len(batch_img),-1)
        #eval_out = torch.where(eval_out >= 0.5,
        #                       torch.ones(eval_out.shape),
        #                       torch.zeros(eval_out.shape))
        results.append(eval_out.float())
    results = torch.cat(results, dim=0)
    results = results.view(len(test_files), -1)
    return results
