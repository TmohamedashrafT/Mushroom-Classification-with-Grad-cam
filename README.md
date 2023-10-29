# Mushroom-Classification-with-Grad-CAM
This repository aims to implement a mushroom type classifier using PyTorch, utilizing various models to enhance performance. Additionally, the project includes an analysis of the model's performance using Gradient-Class Activation Map (Grad-CAM) visualization.

## Dataset
The dataset was obtained from Kaggle, specifically from the "LOVE OF A LIFETIME" collection. It consists of nine classes of mushrooms, which were downloaded from Kaggle and then split into train (65%), validation (20%), and test (15%) sets. The split was done equally among the classes.<br />[Download and split.ipynb](https://github.com/TmohamedashrafT/Mushroom-Classification-with-Grad-cam/blob/main/Download%20and%20split.ipynb)<br /><br />
![image](https://github.com/TmohamedashrafT/Mushroom-Classification-with-Grad-cam/blob/main/readme_imgs/number%20of%20classes.png) <br />
### Data Preprocessing
- Resize to 299  
- Normalize channals as [pytorch recommendation](https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2)
  - mean = [0.485, 0.456, 0.406]
  - std = [0.229, 0.224, 0.225]
### Data augmentation
- RandomHorizontalFlip
- RandomVerticalFlip
- RandomRotation with maximum degrees 15
#### The data was loaded and augmented [Dataset_Generator.py](https://github.com/TmohamedashrafT/Mushroom-Classification-with-Grad-cam/blob/main/tools/Dataset_Generator.py)
## Models
[models.py](https://github.com/TmohamedashrafT/Mushroom-Classification-with-Grad-cam/blob/main/tools/models.py)
#### List of models:
- ResNet50
- Convnext
#### Each model is followed by:
- Linear (out_features, out_features //2)
- BatchNorm1d(out_features // 2)
- Relu()
- Dropout(0.3)
- Linear (out_features // 2, out_features //4)
- BatchNorm1d(out_features // 4)
- Relu()
- Dropout(0.2)
- Linear (out_features // 4, number of classes)
## Gradient-Class Activation Map (Grad-CAM)
### Grad-CAM Overview
Grad-CAM is a visualization technique that allows us to understand what the network focuses on when making decisions based on an image. It combines the concepts of a saliency map and a class activation map. Grad-CAM works by computing the gradients of the output of the network to determine which parts of the image contribute the most to the network assigning the highest probability to a specific class.

By utilizing Grad-CAM, we can generate informative heatmaps that highlight the regions in the input image that are most influential in the network's decision-making process. These heatmaps help us interpret and analyze the model's behavior by visualizing the areas that the network pays the most attention to when classifying mushroom types.<br />
![image](https://github.com/TmohamedashrafT/Mushroom-Classification-with-Grad-cam/blob/main/readme_imgs/Grad_cam_flow.png) 
### For more details 
- https://arxiv.org/pdf/1610.02391.pdf
- https://medium.com/@ninads79shukla/gradcam-73a752d368be
- https://towardsdatascience.com/understand-your-algorithm-with-grad-cam-d3b62fce353
#### The Grad-CAM was implemented in [Grad_cam.py](https://github.com/TmohamedashrafT/Mushroom-Classification-with-Grad-cam/blob/main/tools/Grad_cam.py)
[Grad_cam_utils.py](https://github.com/TmohamedashrafT/Mushroom-Classification-with-Grad-cam/blob/main/tools/Grad_cam_utils.py)
This file contains functions to generate heatmaps using Grad-CAM and plot them.
## Metrics
- Accuracy
- Recall
- Precision
- F1-score
#### The Metrics was implemented in [Metrics.py](https://github.com/TmohamedashrafT/Mushroom-Classification-with-Grad-cam/blob/main/tools/metrics.py)
## Utils Function
- show_batch     : show random images from each class
- show_aug_batch : show 9 random images after transformations
- plot_results   :  plot the results of the same metric for both the training and validation datasets
## Training settings
- Learning rate : 1e-3 with cosine annealing scheduler
- Optimizer : Adam
- Epochs : 100
- Loss : Cross entropy
- Freeze the weights of the backbone
# Results
After examining the ResNet notebook, it appears that the model is unable to effectively handle this particular dataset.
| Data | Loss | Accuracy |
| :---: | :---: | :---: |
| Train | 0.207 | 93.3% |
| Val   | 0.724 | 77.5% | 
| Test  | 0.656 | 77.8% |
Upon analyzing the dataset, it becomes apparent that mushrooms of the same class exhibit diverse shapes and colors. This variation poses a challenging task for humans to accurately classify the different types of mushrooms.

To overcome this complexity, a larger and more powerful model will be utilized. By employing a larger model, we aim to capture a broader range of features and patterns present in the mushroom images. This increased capacity will enhance the model's ability to differentiate between various types of mushrooms
![
