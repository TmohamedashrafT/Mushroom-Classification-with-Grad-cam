from utils import tensor_to_img, load_img
import matplotlib.pyplot as plt
import cv2
from Config import cfg_aug
import numpy as np
def Generate_cam(img, grad_cam, true_class, transformations = None, plot = True, save = False, save_path = 'cam_img'):
  if  isinstance(img,str):
    img = load_img(img,transformations)

  heatmap, predicted_class = grad_cam.forward(img)
  array_img           = tensor_to_img(img, cfg_aug.mn, cfg_aug.sd)
  applied_img,heatmap = apply_cam(array_img, heatmap)

  if plot:
    plot_cam(array_img, heatmap, applied_img, true_class, predicted_class, save, save_path)
  else:
    return array_img, heatmap, applied_img

def plot_cam(array_img, heatmap, applied_img, true_class, predicted_class, save, save_path):
  plt.figure(figsize=(15, 5))
  plt.subplot(1, 3, 1)
  plt.imshow(array_img)
  plt.title("Image, class = " + str(true_class.item()))
  plt.axis('off')


  plt.subplot(1, 3, 2)
  plt.imshow(heatmap)
  plt.title("Heatmap")
  plt.axis('off')

  plt.subplot(1, 3, 3)
  plt.imshow(applied_img)
  plt.title("Image with heatmap, predicted class = " + str(predicted_class.item()))
  plt.axis('off')
  if save:
    plt.savefig( save_path,bbox_inches = 'tight')
  plt.show()
  plt.close()
    
def apply_cam(img, heatmap):
  heatmap     = cv2.resize(heatmap,(img.shape[1],img.shape[0]))
  heatmap     = np.uint8(255 * heatmap)
  heatmap     = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
  heatmap     = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
  img         = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  applied_img = cv2.addWeighted(img, 0.7, heatmap, 0.3, 0)

  return applied_img, heatmap
