import torch
class Grad_cam():
  def __init__(self,model):
    self.model        = model
    self.gradients    = None
    self.feature_maps = None
    self.get_last_conv()

  def forward(self,img):
    self.model.zero_grad()
    img.requires_grad_()
    self.model.eval()
    predictions      = self.model(img)
    predicted_class  = predictions.argmax()
    # get the gradiants with respect  to the predicted class
    predictions[...,predicted_class].backward()
    ''' get the mean over hight and width in all feature maps
        shape from (batch_size, feature_maps, height, width) to ((batch_size, feature_maps, 1, 1))'''
    gradients  = torch.mean(self.gradients, dim=[2, 3],keepdim = True)
    # weight the feature maps by the gradients
    weighted_feature_maps  = gradients * self.feature_maps
    '''get the mean of all feature maps
       gradients shape is (batch_size, feature_maps, height, width)'''
    heatmap   = torch.mean(weighted_feature_maps, dim=1).squeeze()

    # The aim of clipping is to restrict or limit the values to only retain the positive influence.
    heatmap   = torch.clip(heatmap, 0)
    # normalize the heatmap
    heatmap  /= torch.max(heatmap)
    return heatmap.detach().cpu().numpy(), predicted_class

  def save_grads(self, module, input, grad_output):
    self.gradients = grad_output[0]

  def save_feature_maps(self,module, input, output):
    self.feature_maps = output
  def get_last_conv(self):
    # get last conv layer
    for _, module in self.model.named_modules():
      if isinstance(module, torch.nn.modules.conv.Conv2d):
        last_conv = module
    # save feature maps of last conv layer
    last_conv.register_forward_hook(self.save_feature_maps)
    # save the gradiants, from last linear layer to conv
    last_conv.register_backward_hook(self.save_grads)