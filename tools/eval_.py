import torch 
from metrics import calc_accuracy, calc_recall_precision
def eval_(val_loader, model, criterion, device):
  model.eval()
  val_loss, val_accuracy, val_recall_precision = 0, 0, 0
  with torch.no_grad():
    for images, labels in (val_loader):
      images, labels = images.to(device), labels.to(device)
      preds = model(images)
      loss  = criterion(preds, labels)
      val_loss      += loss.item()
      val_accuracy  += calc_accuracy(preds.argmax(dim = 1), labels)
      val_recall_precision += calc_recall_precision(preds.argmax(dim = 1), labels)

  return val_loss, val_accuracy, val_recall_precision