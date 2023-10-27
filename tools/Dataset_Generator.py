import torch
from torchvision import transforms as T
from torch.utils.data import DataLoader,Dataset
from PIL import Image
import os
from Config import cfg_aug
aug_imgs = {'train':T.Compose([
           T.Resize((cfg_aug.img_size, cfg_aug.img_size)),
           T.RandomHorizontalFlip(),
           T.RandomVerticalFlip(),
           T.ToTensor(),
           T.RandomRotation(15),
           T.Normalize(mean=cfg_aug.mn, std=cfg_aug.sd)
           ]),
          'val':T.Compose([
           T.Resize((cfg_aug.img_size, cfg_aug.img_size)),
           T.ToTensor(),
           T.Normalize(mean=cfg_aug.mn, std=cfg_aug.sd)
           ])}

class Dataset_Generator(Dataset):
  def __init__(self,folder_path,transform,classes):
    super().__init__()
    self.classes   = classes
    self.img_files = self.load_files(folder_path)
    self.transform = transform

  def __len__(self):
    return len(self.img_files)

  def __getitem__(self,index):
    try:
      img_path, label = self.img_files[index]
      img = Image.open(img_path).convert('RGB')
      assert  img is not None ,f'{img_path} not founded'
      img = self.transform(img)
    except:
      print(img_path)
    return img , torch.tensor(label)

  def load_files(self,folder_path):
    img_files = []
    for cls in os.listdir(folder_path):
      try:
        img_files += [[os.path.join(folder_path,cls,img_path),self.classes.index(cls)] for img_path in os.listdir(os.path.join(folder_path , cls))]
      except:
        print('There is a class that does not exist  in our classes')
    return img_files
def get_loaders(
        train_dir,
        val_dir,
        test_dir,
        classes,
        batch_size,
        shuffle
                ):
    train_dataset = Dataset_Generator(train_dir,aug_imgs['train'],classes)

    train_loader  = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle
    )
    val_dataset = Dataset_Generator(val_dir,aug_imgs['val'],classes)
    val_loader  = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=shuffle
    )
    test_dataset = Dataset_Generator(test_dir,aug_imgs['val'],classes)
    test_loader  = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=shuffle
    )
    return train_loader,val_loader,test_loader
