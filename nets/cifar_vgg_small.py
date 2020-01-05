import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.quant_dorefa import *


class VGG_Cifar10(nn.Module):

  def __init__(self, ratio_code, wbit_code, abit_code, num_classes=10):
      super(VGG_Cifar10, self).__init__()
      in_channels = [3, 128, 128, 256, 256, 512]
      out_channels = [128, 128, 256, 256, 512, 512]
      for i in range(6):
          if i != 5:
              in_channels[i+1] = int(in_channels[i+1]*ratio_code[i])
          out_channels[i] = int(out_channels[i]*ratio_code[i])
      self.in_planes = int(512*4*4*ratio_code[5])
      self.features = nn.Sequential(
          nn.Conv2d(in_channels[0], out_channels[0], kernel_size=3, stride=1, padding=1,
                    bias=False),
          nn.BatchNorm2d(out_channels[0]),
          nn.ReLU(inplace=True),

          activation_quantize_fn(a_bit=abit_code[1]),
          Conv2d_Q(wbit_code[1],in_channels[1], out_channels[1], kernel_size=3, padding=1, bias=False),
          nn.BatchNorm2d(out_channels[1]),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2, stride=2),
          
          activation_quantize_fn(a_bit=abit_code[2]),
          Conv2d_Q(wbit_code[2],in_channels[2], out_channels[2], kernel_size=3, padding=1, bias=False),
          nn.BatchNorm2d(out_channels[2]),
          nn.ReLU(inplace=True),

          activation_quantize_fn(a_bit=abit_code[3]),
          Conv2d_Q(wbit_code[3],in_channels[3], out_channels[3], kernel_size=3, padding=1, bias=False),
          nn.BatchNorm2d(out_channels[3]),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2, stride=2),
          
          activation_quantize_fn(a_bit=abit_code[4]),
          Conv2d_Q(wbit_code[4],in_channels[4], out_channels[4], kernel_size=3, padding=1, bias=False),  
          nn.BatchNorm2d(out_channels[4]),
          nn.ReLU(inplace=True), 

          activation_quantize_fn(a_bit=abit_code[5]),
          Conv2d_Q(wbit_code[5],in_channels[5], out_channels[5], kernel_size=3, padding=1, bias=False),
          nn.BatchNorm2d(out_channels[5]),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2, stride=2),       
      )
      self.classifier = nn.Sequential(
          nn.Linear(self.in_planes, 10, bias=False),
      )


  def forward(self, x):
      x = self.features(x)
      x = x.view(-1, self.in_planes)
      x = self.classifier(x)
      return x


if __name__ == '__main__':
  import numpy as np
  import matplotlib.pyplot as plt

  net = VGG_Cifar10(ratio_code =[0.5, 2, 1, 1, 1, 0.5], 
                    wbit_code = [32,1,1,1,1,1], 
                    abit_code = [32,1,1,1,1,1], num_classes=10)

  a = torch.ones(1, 3, 32, 32)

  output1 = net(a)
  print (output1)

  pass
