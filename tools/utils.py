import matplotlib.pyplot as plt
import random
import cv2
import numpy as np
from PIL import Image
import os
import torch
import seaborn as sns
import pandas as pd
from metrics import calc_precision_cls, calc_recall_cls, calc_f1_score_cls
sns.set()
def show_batch(data_dir):

  ''' show random image from every class '''
  fig = plt.figure(figsize=(12, 12))
  cols, rows = 3, 3
  for i, cls in enumerate(os.listdir(data_dir)):
    img_path = random.choice(os.listdir(os.path.join(data_dir,cls)))
    fig.add_subplot(rows, cols, i + 1)
    plt.title(cls)
    plt.axis("off")
    img = cv2.imread(os.path.join(data_dir,cls,img_path))
    img = cv2.resize(img,(299,299))
    plt.imshow(img)
  plt.show()

def show_aug_batch(data_loader,classes):
  ''' show random images after transformations '''
  imgs, labels = next(iter(data_loader))
  fig = plt.figure(figsize=(12, 12))
  cols, rows = 3, 3
  for i in range(1, cols * rows + 1):
      fig.add_subplot(rows, cols, i)
      plt.title(classes[labels[i]])
      plt.axis("off")
      plt.imshow(imgs[i].permute(1, 2, 0).clip(0, 1)) ## clipping for matplotlib warning
  plt.show()
def plot_results(train, val, plot_type = 'accuracy'):
  plt.figure(figsize = (8,5))
  plt.plot(train, 'bo--', label = 'train_' + str(plot_type))
  plt.plot(val, 'ro--', label = 'val_' + str(plot_type))
  plt.title('train_' + str(plot_type) + ' vs ' + 'val_' + str(plot_type))
  plt.ylabel(plot_type)
  plt.xlabel('epochs')
  plt.legend()
  plt.show()
def load_img(img_path, transformations):
  img = Image.open(img_path)
  return transformations(img)[None]

def tensor_to_img(normalized_tensor, mn, sd):
  # from torch.tensor to numpy.ndarray
  normalized_img = normalized_tensor.detach().cpu().numpy()
  # from (bs, ch, h, w) to (h, w, ch)
  normalized_img = normalized_img.squeeze(axis = 0).transpose(1,2,0)
  # normalize eq is  (img - mean)/std
  img  = normalized_img  * sd + mn
  img  = np.uint8(img * 255.0)
  return img

def load_ckpts(ckpt_path, model, device, optimizer = None):
  ckpt    = torch.load(ckpt_path, map_location = device)
  history =  ckpt['history']
  model.load_state_dict(ckpt['weights'])
  if optimizer is not None:
    optimizer.load_state_dict(ckpt['optimizer'])
    del ckpt
    return model, optimizer, history
  del ckpt
  return model, history
import torch

def preds_with_loaders (loader, model, device,classes):
  precision   = np.zeros((len(classes)))
  recall      = np.zeros((len(classes)))
  classes_idx = np.arange(0, len(classes))
  labelss ,predss = [], []
  with torch.no_grad():
    for images, labels in (loader):
      images, labels = images.to(device), labels.to(device)
      preds = model(images)
      predss.append(preds.argmax(dim = 1).detach().cpu())
      labelss.append(labels.detach().cpu())
  return torch.cat(predss, dim = 0), torch.cat(labelss, dim = 0)
  
def calc_precision_recall_cls(preds, labels, classes):
  classes_idx = np.arange(0, len(classes))
  precision = calc_precision_cls(preds , labels, classes_idx)
  recall    = calc_recall_cls(preds, labels, classes_idx)
  f1_scores = calc_f1_score_cls(precision, recall)
  print(" %-12s %12s %12s %12s" % ('classes','precision','recall','f1_score'),'\n')
  for i, cls in enumerate(classes):
    print(" %-12s %12.2f %12.2f %12.2f" % (cls, precision[i], recall[i],f1_scores[i]))
    
def plot_confusion_matrix(preds, labels, classes):
  matrix = []
  classes_idx = np.arange(0, len(classes))
  for cls in classes_idx:
    row = []
    for cls2 in classes_idx:
      row.append(((preds == cls2) & (labels  == cls)).sum().item())
    matrix.append(row)
  heat_map = pd.DataFrame(np.array(matrix), index = classes, columns = classes)
  plt.figure(figsize = (13,7))
  ax = sns.heatmap(heat_map, annot=True,cmap="Blues", fmt=',d')
  plt.xlabel("labels", labelpad=15,size=20 )
  plt.ylabel("pred", labelpad=15,size=20 )
  plt.xticks(rotation = -30)
  plt.show()
