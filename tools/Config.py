from easydict import EasyDict as edict
import torch
cfg   = edict()



cfg.model = edict()
cfg.model.train_dir        = '/content/drive/MyDrive/mushrooms-classification/train/'
cfg.model.val_dir          = '/content/drive/MyDrive/mushrooms-classification/val/'
cfg.model.test_dir         = '/content/drive/MyDrive/mushrooms-classification/test/'
cfg.model.classes          = ['Amanita','Boletus','Cortinarius','Hygrocybe','Entoloma','Russula','Suillus','Agaricus','Lactarius']
cfg.model.num_classes      = 9
cfg.model.device           ='cuda' if torch.cuda.is_available() else 'cpu'
cfg.model.pretrained       = True
cfg.model.batch_size       = 32
cfg.model.shuffle          = True
cfg.model.opt_name         = 'Adam'
cfg.model.lr               = 0.001
cfg.model.epochs           = 100
cfg.model.pretrained_models = ['inceptionv3','resnet50','convnext','alxnet']

cfg.model.last_ckpt_path   = ['/content/drive/MyDrive/mushrooms-classification/last_inceptionv3.pt',
                              '/content/drive/MyDrive/mushrooms-classification/last_resnet50.pt',
                              '/content/drive/MyDrive/mushrooms-classification/last_convnext.pt',
                              '/content/drive/MyDrive/mushrooms-classification/last_alxnet.pt']
cfg.model.best_ckpt_path   = ['/content/drive/MyDrive/mushrooms-classification/best_inceptionv3.pt',
                              '/content/drive/MyDrive/mushrooms-classification/best_resnet50.pt',
                              '/content/drive/MyDrive/mushrooms-classification/best_convnext.pt',
                              '/content/drive/MyDrive/mushrooms-classification/best_alxnet.pt']
## augmentation details
cfg_aug = edict()

cfg_aug.mn ,cfg_aug.sd = ([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]) if cfg.model.pretrained  else([0.5,0.5,0.5] , [0.5,0.5,0.5])
cfg_aug.img_size       = 299

