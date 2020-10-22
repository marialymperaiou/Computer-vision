import cv2
import torch
from torch.autograd import Variable
import imageio
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd

# performs the detection
def detection(frame, net, transform): # image, neural network, transformations for compatibility
      height, width, n_channels = frame.shape  
      # or height, width = frame.shape[:2] if we focus only to the height and width parameters
      
      # Transformation 1
      # input ftame has right dimensions and colors
      frame_t = transform(frame)[0] # keep only the first element, which corresponds to the new frame
      
      # Transformation 2
      # Numpy array to torch tensor
      x = torch.from_numpy(frame_t)
      #RBG -> GRB for the neural network
      x = x.permute(2,0,1)
      
      # Transformation 3
      # Add fake dimension for batch for the neural network (NN accepts only batches of images)
      # and convert this batch into a torch variable
      x = Variable(x.unsqueeze(0)) # index of the 1st dimension corresponding to the batch
      
      # feed the batch of tensors to the neural network
      y = net(x)
      
      # get the target information (values of the output)
      detections = y.data
      
      # Create tensor (width,height,width,height) for normalizing the location of the objects detected in the image
      # upper left and lower right coordinates
      scale = torch.Tensor([width, height, width, height])
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      