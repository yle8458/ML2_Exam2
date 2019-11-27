# -----------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
from torchvision import models
from torch import optim
import numpy as np
import tqdm
import copy
from predict_ppzt import predict
import time
import os

# -----------------------------------------------------------------------------------
# Hyper Parameters
num_epochs = 10
batch_size = 12
learning_rate = 0.00005
IMG_COL = 800
IMG_ROW = 600
N_LABEL = 7
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
label_dict = {"red blood cell": 0,
              "difficult": 1,
              "gametocyte": 2,
              "trophozoite": 3,
              "ring": 4,
              "schizont": 5,
              "leukocyte": 6}


# dataset class
class exam_dataset(torch.utils.data.Dataset):
    def __init__(self, dir, transforms = None):
        self.dir = dir
        self.transforms = transforms
        self.img_filenames = sorted([file.name for file in os.scandir(dir) if file.name.endswith('png')])
        self.n_imgs = len(self.img_filenames)


    def __getitem__(self, idx):
        img_path = os.path.join(self.dir, self.img_filenames[idx])
        img = Image.open(img_path).convert('RGB')
        #img = cv2.resize(img, (self.ncol, self.nrow), interpolation=cv2.INTER_AREA)
        #img = np.moveaxis(img, -1, 0)
        label_fname = self.img_filenames[idx][:-4] + '.txt'
        label_path = os.path.join(self.dir, label_fname)
        label = [0] * N_LABEL
        with open(label_path, 'r') as label_file:
            line = label_file.readline()
            while line:
                line = line.rstrip('\n')
                if (line in label_dict.keys()):
                    label[label_dict[line]] = 1.
                line = label_file.readline()


        label = torch.as_tensor(label, dtype = torch.float32)
        if self.transforms is not None:
            img = self.transforms(img)

        return img, label

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


# -----------------------------------------------------------------------------------
# Resnet Model
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

        for param in self.resnet.layer4.parameters():
            param.requires_grad = True

        for param in self.resnet.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        out = self.resnet(x)
        return out


res50 = ResNet50(num_outputs=N_LABEL)
# -----------------------------------------------------------------------------------

train_set = exam_dataset( 'train', get_transform(train = True) )
test_set = exam_dataset( 'train', get_transform(train = False) )

# split the dataset in train and test set
indices = torch.randperm(len(train_set)).tolist()
test_size = int(0.1 * len(indices))
train_set = torch.utils.data.Subset(train_set, indices[:-test_size])
test_set = torch.utils.data.Subset(test_set, indices[-test_size:])

train_loader = torch.utils.data.DataLoader(train_set,
                                           batch_size = batch_size,
                                           shuffle = True,
                                           drop_last = False)
test_loader = torch.utils.data.DataLoader(test_set,
                                          batch_size = 1,
                                          shuffle = False,
                                          drop_last=False)

dataloaders_dict = {}
dataloaders_dict['train'] = train_loader
dataloaders_dict['val'] = test_loader
# -----------------------------------------------------------------------------------


res50.cuda()
# -----------------------------------------------------------------------------------
# Loss and Optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(res50.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience = 2)
best_loss = np.inf
# -----------------------------------------------------------------------------------
# Train the Model

min_val_loss = None
for epoch in range(num_epochs):
    for phase in ['train', 'val']:
        start_time = time.time()
        if phase == 'train':
            res50.train()
        else:
            res50.eval()

        running_loss = 0.0
        for images_batch, labels_batch in dataloaders_dict[phase]:
            images_batch = images_batch.to(device)
            labels_batch = labels_batch.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                pred_batch = res50(images_batch)
                loss = criterion(pred_batch, labels_batch)

            if phase == 'train':
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images_batch.size(0)
        epoch_loss = running_loss / len(dataloaders_dict[phase].dataset)

        if phase == 'val' and epoch_loss < best_loss:
            print("model val_loss Improved from {:.8f} to {:.8f}".format(best_loss, epoch_loss))
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(res50.state_dict())

        if phase == 'val':
            scheduler.step(epoch_loss)

        elapsed_time = time.time() - start_time
        print("Phase: {} | Epoch: {}/{} | {}_loss:{:.8f} | Time: {:.4f}s".format(phase,
                                                                         epoch + 1,
                                                                         num_epochs,
                                                                         phase,
                                                                         epoch_loss,
                                                                         elapsed_time))
res50.load_state_dict(best_model_wts)

# -----------------------------------------------------------------------------------
