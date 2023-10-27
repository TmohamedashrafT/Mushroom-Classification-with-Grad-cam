import torchvision
import torch.nn as nn
from torch.nn import functional as F

class classification_model(nn.Module):
  def __init__(self, pretrained_model, pretrained, num_classes):
   super().__init__()
   pretrained_model = pretrained_model.lower()
   if  pretrained_model == 'resnet50':
    weights    = torchvision.models.ResNet50_Weights.DEFAULT if pretrained else None
    self.model = torchvision.models.resnet50(weights = weights)

   elif pretrained_model == 'convnext':
    weights    = torchvision.models.ConvNeXt_Base_Weights.DEFAULT if pretrained else None
    self.model = torchvision.models.convnext_base(weights = weights)

   elif pretrained_model == 'inceptionv3':
    weights    = torchvision.models.Inception_V3_Weights.DEFAULT if pretrained else None
    self.model = torchvision.models.inception_v3(weights = weights)
    self.model.aux_logits=False
   else:
    self.model = torchvision.models.alexnet(weights = None)
    pretrained = False
   if pretrained:
    for param in self.model.parameters():
            param.requires_grad = False
   in_features = 1024 if pretrained_model == 'convnext' else self.model.fc.in_features
   Sequential  =  nn.Sequential(
                    nn.Linear(in_features, in_features//2),
                    nn.BatchNorm1d(in_features//2),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(in_features//2, in_features//4),
                    nn.BatchNorm1d(in_features//4),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(in_features//4, num_classes),
                                  )
   if pretrained_model == 'convnext':
    self.model.classifier = nn.Sequential(
          LayerNorm2d((1024,), eps=1e-06, elementwise_affine=True),
          nn.Flatten(start_dim=1, end_dim=-1),
          Sequential
          )
   else:
    self.model.fc = Sequential
  def forward(self,x):
    return self.model(x)

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x