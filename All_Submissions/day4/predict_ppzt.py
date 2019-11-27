import cv2
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

IMG_COL = 800
IMG_ROW = 600
batch_size = 16
N_LABEL = 7

# CNN Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv2d1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)) # in: 800 * 600 * 3, out: 400*300*16
        self.conv2d2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)) # in: 400*300*16, out: 200*150*32
        self.conv2d3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))  # in: 200*150*32, out: 100*75*64
        self.fc1 = nn.Sequential(
            nn.Linear(100*75*64, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(),
            nn.Dropout(0.2))
        self.fc2 = nn.Linear(400, N_LABEL)

    def forward(self, x):
        x = self.conv2d1(x)
        x = self.conv2d2(x)
        x = self.conv2d3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

def predict(test_files):
    results = []
    cnn = CNN()
    cnn.cuda()
    cnn.load_state_dict(torch.load('model_ppzt.pt'))
    cnn.eval()
    for file in test_files:
        img = cv2.imread(file, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_COL, IMG_ROW), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        img = np.moveaxis(img, -1, 0)
        img = torch.from_numpy(img / 255.0)
        img = Variable(img.float().cuda())
        img = img.unsqueeze(0)
        eval_out = cnn(img)
        activation = nn.Sigmoid()
        eval_out = activation(eval_out)
        eval_out = eval_out.cpu().detach()
        eval_out.view(1,-1)
        eval_out = torch.where(eval_out >= 0.5,
                               torch.ones(eval_out.shape),
                               torch.zeros(eval_out.shape))
        results.append(eval_out.float())
    results = torch.stack(results)
    results = results.view(len(test_files), -1)
    return results

