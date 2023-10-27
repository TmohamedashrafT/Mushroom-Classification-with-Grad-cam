from tqdm import tqdm
import torch 
import torch.nn as nn
import os
from eval_ import eval_
from metrics import calc_accuracy,calc_recall_precision
from Grad_cam_utils import Generate_cam
from utils import load_ckpts
from Grad_cam import Grad_cam
from models import classification_model
from utils import  load_ckpts
from Config import cfg
import warnings
warnings.filterwarnings("ignore")

def train_(
            train_loader,
            val_loader,
            test_loader,
            model,
            grad_cam,
            optimizer,
            epochs,
            criterion,
            scheduler,
            device,
            last_ckpt_path,
            best_ckpt_path,
            early_stopping,
            history
            ):
    model.train()
    print('################ Start  Training ################')
    # choose random image to see changes in heatmap
    rand_img, true_class = next(iter(val_loader))
    rand_img, true_class = rand_img.to(device), true_class.to(device)
    rand_img, true_class = rand_img[0][None], true_class[0]
    st_epoch = history['epochs'][-1] if history['epochs'] else 0
    for epoch in range(st_epoch,epochs):
      model.train()
      total_loss = 0.0
      accuracy   = 0.0
      recall_precision = 0.0
      print(f'epoch [{epoch}/{epochs}]')
      train_loss, train_accuracy, train_recall_precision = 0, 0, 0
      for images, labels in tqdm(train_loader, ascii = True, desc ="Training"):
        optimizer.zero_grad()
        images, labels = images.to(device), labels.to(device)
        preds          = model(images)
        loss           = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        scheduler.get_last_lr()
        train_loss     += loss.item()
        train_accuracy += calc_accuracy(preds.argmax(dim = 1), labels)
        train_recall_precision += calc_recall_precision(preds.argmax(dim = 1), labels)
      val_loss, val_accuracy, val_recall_precision = eval_(val_loader, model, criterion, device)

      history['epochs'].append(epoch)
      history['lr'].append(scheduler.get_last_lr())
      history['train_loss'].append(train_loss / len(train_loader))
      history['val_loss'].append(val_loss / len(val_loader))
      history['train_accuracy'].append(train_accuracy / len(train_loader))
      history['val_accuracy'].append(val_accuracy / len(val_loader))
      history['train_recall_precision'].append(train_recall_precision / len(train_loader))
      history['val_recall_precision'].append(val_recall_precision / len(val_loader))
      ## save last weights
      ckpt ={'weights'  : model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'history'  : history,
            }
      torch.save(ckpt, last_ckpt_path)
      ## save best weights
      if history['val_loss'][-1] <= min(history['val_loss']):
        torch.save(ckpt, best_ckpt_path)
      del ckpt

      print('gpu_mem = ',round(torch.cuda.memory_reserved()/1E9,3),
            '\ntrain_loss = ',            round(history['train_loss'][-1],2),
            ' train_accuracy = ',         round(history['train_accuracy'][-1],2),
            ' train_recall_precision = ', round(history['train_recall_precision'][-1],2),
            '\nval_loss = ',              round(history['val_loss'][-1],2),
            ' val_accuracy = ',           round(history['val_accuracy'][-1],2),
            ' val_recall_precision = ',   round(history['val_recall_precision'][-1],2)
            )
      if early_stopping(history['val_loss'][-1]):
        print('Early stopping')
        break
      Generate_cam(rand_img, grad_cam, true_class)
      torch.cuda.empty_cache()
    print('################ Training Finished ################')
    print('Evaluating on test dataset')
    model,_ = load_ckpts(best_ckpt_path, model, device)
    test_loss, test_accuracy, test_recall_precision = eval_(test_loader, model, criterion, device)
    print('test_loss = ', train_loss, ' test_accuracy = ', test_accuracy, 'test_recall_precision = ', test_recall_precision)


def update_pretrained_model(pretrain_model,pretrained_weights):
    model = classification_model(pretrain_model,
                                 cfg.model.pretrained,
                                 cfg.model.num_classes).to(cfg.model.device)
    grad_cam  = Grad_cam(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer_(model, cfg.model.opt_name, cfg.model.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    early_stopping = Early_stopping ()
    history = {'epochs':[],'lr':[],'train_loss': [], 'train_accuracy': [],'train_recall_precision': [], 'val_loss': [], 'val_accuracy': [],'val_recall_precision': [],}
    if os.path.exists(pretrained_weights):
      model, optimizer, history = load_ckpts(pretrained_weights, model, cfg.model.device, optimizer)
    return model, grad_cam, criterion, optimizer, scheduler, early_stopping, history

def optimizer_(model, opt_name='RMSProp', lr=0.001, momentum=0.9, weight_decay=1e-5):
      if   opt_name=='SGD':
          optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,weight_decay=weight_decay)
      elif opt_name=='Adam':
          optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(momentum,0.999),weight_decay=weight_decay)
      elif opt_name=='RMSProp':
          optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, momentum=momentum,weight_decay=weight_decay)
      else:
          raise NotImplementedError(f'optimizer {opt_name} not implemented')
      return optimizer

class Early_stopping:
    def __init__(self, patience=5, small = True):
        self.best_scale = 1e5 if small else 0.0
        self.fipatience = patience
        self.patience   = patience
        self.small      = small
    def __call__(self, scale):
        if scale < self.best_scale and self.small:
            self.patience = self.fipatience
            self.best_scale = scale
        elif scale > self.best_scale:
            self.patience = self.fipatience
            self.best_scale = scale
        else :
            self.patience -= 1
        stop = True if not self.patience else False
        return stop


