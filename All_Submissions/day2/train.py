# -----------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from predict_ppzt import predict
import os
import cv2
# -----------------------------------------------------------------------------------
# Hyper Parameters
num_epochs = 5
batch_size = 32
learning_rate = 0.00005
IMG_COL = 800
IMG_ROW = 600
N_LABEL = 7
# -----------------------------------------------------------------------------------
def loadData(folder_name):
    imgs = []
    labels = []
    img_name = []
    json_list = []
    label_dict = {"red blood cell" : 0,
                  "difficult" : 1,
                  "gametocyte" : 2,
                  "trophozoite" : 3,
                  "ring" : 4,
                  "schizont" : 5,
                  "leukocyte": 6}

    for file in os.scandir(folder_name):
        label = [0] * N_LABEL
        if (file.is_file() and file.name.endswith('png')):
            img = cv2.imread(os.path.join(folder_name, file.name), cv2.IMREAD_COLOR)
            img = cv2.resize(img, (IMG_COL, IMG_ROW), interpolation = cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            img = np.moveaxis(img, -1, 0)
            label_fname = file.name[:-4] + '.txt'
            img_name.append(os.path.join(folder_name, file.name))
            imgs.append(img)
            json_fname = file.name[:-4] + '.json'
            json_list.append(os.path.join(folder_name, json_fname))
            with open(os.path.join(folder_name, label_fname), 'r') as label_file:
                line = label_file.readline()
                while line:
                    line = line.rstrip('\n')
                    if (line in label_dict.keys()):
                        label[label_dict[line]] = 1.
                    line = label_file.readline()
                labels.append(label)
    return np.array(imgs), np.array(labels), img_name, json_list
print(torch.cuda.is_available())

# -----------------------------------------------------------------------------------
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
        self.fc = nn.Linear(100*75*64, N_LABEL)

    def forward(self, x):
        x = self.conv2d1(x)
        x = self.conv2d2(x)
        x = self.conv2d3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
# -----------------------------------------------------------------------------------
imgs, labels, img_name, json_list = loadData('train')
imgs = imgs / 255.
idx = np.array(range(len(labels)))

idx_train, idx_test, y_train, y_test = train_test_split(idx, labels, test_size=0.1, random_state=42)
X_train = imgs[idx_train]
X_test = imgs[idx_test]
test_files = np.array(img_name)[idx_test]

train_dataset = zip(X_train, y_train)
test_dataset = zip(test_files, y_test)
val_dataset = zip(X_test, y_test)
# Data Loader (Input Pipeline)

train_loader = torch.utils.data.DataLoader(dataset=list(train_dataset), batch_size=batch_size,shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=list(val_dataset), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=list(test_files), batch_size=batch_size, shuffle=False)

# -----------------------------------------------------------------------------------

cnn = CNN()
cnn.cuda()
# -----------------------------------------------------------------------------------
# Loss and Optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
# -----------------------------------------------------------------------------------
# Train the Model

min_val_loss = None
for epoch in range(num_epochs):
    cnn.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.float()
        images = Variable(images).cuda()
        labels = Variable(labels.float()).cuda()

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1,
                     len(X_train) // batch_size, loss.item()))
    cnn.eval()
    val_imgs, val_labels = next(iter(val_loader))
    val_imgs = val_imgs.float()
    val_imgs = Variable(val_imgs).cuda()
    val_labels = Variable(val_labels.float()).cuda()
    val_outputs = cnn(val_imgs)
    val_loss = criterion(val_outputs, val_labels)
    print('Epoch [%d/%d], Val_Loss: %.4f'
          % (epoch + 1, num_epochs, val_loss.item()))
    # Save the Trained Model
    if (min_val_loss == None or min_val_loss > val_loss.item()):
        torch.save(cnn.state_dict(), 'model_ppzt.pt')




# -----------------------------------------------------------------------------------

# Eval
y_pred = predict(test_files)
eval_criterion = nn.BCELoss()
val_loss = eval_criterion(y_pred.cuda(), torch.from_numpy(y_test).float().cuda())
print(val_loss.item())