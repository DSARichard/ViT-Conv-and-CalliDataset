import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from vit_pytorch import SimpleViT
from simple_vit_conv import SimpleViTConv

model_type = ["svc", "vit", "net"][0]
data_dim = [(256, None), (320, 64), (360, 40), (480, 32)][2]
num_epochs, batches, lr, lr_decay = 5, 32, 0.00001, 0.96
device = torch.device("cuda") if(torch.cuda.is_available()) else torch.device("cpu")

class LeNet(nn.Module):
  def __init__(self):
    super(LeNet, self).__init__()
    self.conv1 = nn.Conv2d(1, 6, 5)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.fc1 = nn.Linear(59536, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 2)
  
  def forward(self, x):
    x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
    x = F.max_pool2d(F.relu(self.conv2(x)), 2)
    x = torch.flatten(x, 1)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

if(model_type == "vit"):
  assert data_dim[1] is None, "Data dimensions must match model type"
  vit = SimpleViT(
    image_size = 256,
    patch_size = 16,
    num_classes = 2,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    channels = 1,
  )

elif(model_type == "svc"):
  assert data_dim[1] is not None, "Data dimensions must match model type"
  vit = SimpleViTConv(
    image_size = data_dim[0],
    patch_size = data_dim[1],
    num_classes = 2,
    conv_dim = 8,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    channels = 1,
    kernel_size = 5,
  )

else:
  assert data_dim[1] is None, "Data dimensions must match model type"
  vit = LeNet()

vit.to(device)

class CharDataset():
  def __init__(self, data, targets, transform = None):
    self.data = data
    self.targets = targets
    self.transform = transform
  def __len__(self):
    return len(self.data)
  def __getitem__(self, idx):
    img = np.expand_dims(self.data[idx], 0)
    label = np.int64(self.targets[idx])
    if(self.transform is not None):
      img = self.transform(img)
    return img, label

direc = "CalliDatasetNpy"
train_data = np.load(f"{direc}/train_data{data_dim[0]}.npy")
train_targets = np.load(f"{direc}/train_targets{data_dim[0]}.npy")
test_data = np.load(f"{direc}/test_data{data_dim[0]}.npy")
test_targets = np.load(f"{direc}/test_targets{data_dim[0]}.npy")
train_data, train_targets, test_data, test_targets = map(np.float32, (train_data, train_targets, test_data, test_targets))

char_trainset = CharDataset(
  data = train_data,
  targets = train_targets, transform = torch.from_numpy
)
char_testset = CharDataset(
  data = test_data,
  targets = test_targets, transform = torch.from_numpy
)
torch.manual_seed(668866)
data_loader_train = DataLoader(char_trainset, batch_size = batches, shuffle = True)
data_loader_test = DataLoader(char_testset, batch_size = 1, shuffle = False)

params = [p for p in vit.parameters() if(p.requires_grad)]
optimizer = torch.optim.Adam(params, lr = lr, weight_decay = 0.01)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = lr_decay)

print("train losses:")
vit.train()
train_losses = []
train_acc = [0]*num_epochs
for epoch in range(num_epochs):
  epoch_losses = []
  for imgs, labels in data_loader_train:
    imgs = imgs.to(device)
    labels = labels.to(device)
    preds = vit(imgs)
    train_loss = nn.CrossEntropyLoss()(preds, labels)
    print(train_loss)
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    for i in range(len(preds)):
      train_acc[epoch] += (preds[i, labels[i].item()] == max(preds[i])).to("cpu").item()
    epoch_losses.append(train_loss.to("cpu").item())
  train_losses.append(sum(epoch_losses)/len(epoch_losses))
  lr_scheduler.step()
train_acc = list(map(lambda acc: acc*100/len(epoch_losses)/batches, train_acc))

print("test losses:")
vit.eval()
test_losses = []
test_acc = 0
for imgs, labels in data_loader_test:
  imgs = imgs.to(device)
  labels = labels.to(device)
  preds = vit(imgs)
  test_acc += (preds[0, labels.item()] == max(preds[0])).to("cpu").item()
  test_loss = nn.CrossEntropyLoss()(preds, labels)
  print(test_loss)
  test_losses.append(test_loss.to("cpu").item())
test_acc = test_acc*100/len(test_losses)
test_losses = sum(test_losses)/len(test_losses)

print(f"average train losses:\n{' '.join(map(str, train_losses))}")
print(f"train accuracies:\n{' '.join(map(lambda x: f'{x}%', train_acc))}")
print(f"average test loss:\n{test_losses}")
print(f"test accuracy:\n{test_acc}%")
