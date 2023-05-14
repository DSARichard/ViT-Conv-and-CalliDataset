import cv2
import numpy as np
from overlap_patches import overlap_patches
import os
import gc

model_type = ["svc", "vit", "net"][0]
patch_size = [None, 64, 40, 32][2]
overlap_size = [None, 16, 13, 16][2]

i = 0
train_data, train_targets = [], []
for file in sorted(os.listdir("CalliDataset/trainset")):
  file = f"CalliDataset/trainset/{file}"
  img = cv2.imread(file, 0)
  if(model_type == "svc"):
    img = overlap_patches(img, patch_size, overlap_size)
  train_data.append(img)
  train_targets.append(0)
  train_data.append(img.T)
  train_targets.append(1)
  i += 1
  if(i%100 == 0):
    gc.collect()
train_data, train_targets = np.array(train_data), np.array(train_targets)
test_data, test_targets = [], []
for file in sorted(os.listdir("CalliDataset/testset")):
  file = f"CalliDataset/testset/{file}"
  img = cv2.imread(file, 0)
  if(model_type == "svc"):
    img = overlap_patches(img, patch_size, overlap_size)
  test_data.append(img)
  test_targets.append(0)
  test_data.append(img.T)
  test_targets.append(1)
  i += 1
  if(i%100 == 0):
    gc.collect()
test_data, test_targets = np.array(test_data), np.array(test_targets)

direc = "CalliDatasetNpy"
if(not os.path.isdir(direc)):
  os.makedirs(direc)
data_dim = [256, 320, 360, 480][2]
names = [f"{direc}/train_data{data_dim}.npy", f"{direc}/train_targets{data_dim}.npy", f"{direc}/test_data{data_dim}.npy", f"{direc}/test_targets{data_dim}.npy"]
print(train_data.shape, test_data.shape, names)
np.save(names[0], train_data)
np.save(names[1], train_targets)
np.save(names[2], test_data)
np.save(names[3], test_targets)
