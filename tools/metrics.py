'''
    inputs : torch tensor
    output : scaler
    for all metrics
'''
import numpy as np

def calc_accuracy(pred, true):
  return (round(((pred == true).sum() / len(pred)).item(),2))

def calc_recall_precision(pred, true):
  tp  = (pred == true).sum()
  bot = (tp + (pred != true).sum())
  return round((tp / bot).item(),3)

def calc_precision_cls(pred, true, classes):
  precisions = []
  for cls in classes:
    tp = ((pred == cls) & (true  == cls)).sum()
    precisions.append(round((tp / (1e-7 + tp + ((pred == cls) & (true !=cls)).sum())).item(),3))
  return precisions

def calc_recall_cls(pred, true, classes):
  recalls = []
  for cls in classes:
    tp = ((pred == cls) & (true  == cls)).sum()
    recalls.append(round((tp / ((true == cls).sum() + 1e-7)).item(),3))
  return recalls
def calc_f1_score_cls(precisions, recalls):
  return ( 2 * np.multiply(precisions , recalls) ) /( np.add(precisions , recalls ))