import os
import time
import argparse
from datetime import datetime

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

cudnn.benchmark = True
import torchvision

from nets.cifar_vgg_small import *

from utils.preprocessing import *


pretrain_dir = './ckpt/vgg[111111][111111][111111]/checkpoint.t7'

def run(hyper_list):

  dataset = torchvision.datasets.CIFAR10
  train_dataset = dataset(root='../data', train=True, download=False,
                          transform=cifar_transform(is_training=True))
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True,
                                             num_workers=5)

  eval_dataset = dataset(root='../data', train=False, download=False,
                         transform=cifar_transform(is_training=False))
  eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=100, shuffle=False,
                                            num_workers=5)
  

  
  model = VGG_Cifar10(ratio_code = [1,1,1,1,1,1], 
                      wbit_code = hyper_list, 
                      abit_code = hyper_list, num_classes=10).cuda()

  optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
  criterion = torch.nn.CrossEntropyLoss().cuda()

  
  model.load_state_dict(torch.load(pretrain_dir))

  # Training
  def train(epoch):
    print('Epoch: %d' % epoch)
    model.train()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
      outputs = model(inputs.cuda())
      loss = criterion(outputs, targets.cuda())

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      # if batch_idx % 50 == 0:
      #   print('%s epoch: %d step: %d cls_loss= %.5f' %
      #         (datetime.now(), epoch, batch_idx, loss.item()))

  def test(epoch):
    model.eval()
    correct = 0
    for batch_idx, (inputs, targets) in enumerate(eval_loader):
      inputs, targets = inputs.cuda(), targets.cuda()

      outputs = model(inputs)
      _, predicted = torch.max(outputs.data, 1)
      correct += predicted.eq(targets.data).cpu().sum().item()

    acc = 100. * correct / len(eval_dataset)
    print('%s  ''Precision@1: %.2f%%' % (datetime.now(), acc))
    return acc

  acc_max = 0

  for epoch in range(3):
    train(epoch)
    acc = test(epoch)
    if acc > acc_max:
      acc_max = acc

  print("maximu acc is: ",acc_max)
  return acc_max

if __name__ == "__main__":
  run([1,2,1,1,1,1])
