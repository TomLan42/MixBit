import os
import time
import argparse
from datetime import datetime

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

cudnn.benchmark = True
import torchvision

from tensorboardX import SummaryWriter

from nets.cifar_resnet import *
from nets.cifar_vgg_small import *

from utils.preprocessing import *

# Training settings
parser = argparse.ArgumentParser(description='DoReFa-Net pytorch implementation')

parser.add_argument('--root_dir', type=str, default='./')
parser.add_argument('--data_dir', type=str, default='../data')
parser.add_argument('--log_name', type=str, default='resnet_w1a32')
parser.add_argument('--model_name', type=str, default='resnet')
parser.add_argument('--pretrain', action='store_true', default=False)
parser.add_argument('--pretrain_dir', type=str, default='./ckpt/resnet20_baseline')

parser.add_argument('--cifar', type=int, default=10)

parser.add_argument('--Wbits', type=int, default=1)
parser.add_argument('--Abits', type=int, default=32)

parser.add_argument('--wbit_code', nargs='+', default=[1,1,1,1,1,1])
parser.add_argument('--abit_code', nargs='+', default=[1,1,1,1,1,1])
parser.add_argument('--ratio_code', nargs='+', default=[1,1,1,1,1,1])

parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--wd', type=float, default=1e-4)

parser.add_argument('--train_batch_size', type=int, default=128)
parser.add_argument('--eval_batch_size', type=int, default=100)
parser.add_argument('--max_epochs', type=int, default=200)

parser.add_argument('--log_interval', type=int, default=10)
parser.add_argument('--use_gpu', type=str, default='0')
parser.add_argument('--num_workers', type=int, default=5)

parser.add_argument('--cluster', action='store_true', default=False)

cfg = parser.parse_args()

cfg.wbit_code = list(map(int,cfg.wbit_code))
cfg.abit_code = list(map(int,cfg.abit_code))
cfg.ratio_code = list(map(float,cfg.ratio_code))

print('wbit_code: ',cfg.wbit_code)
print('abit_code: ',cfg.abit_code)
print('expansion_ratio: ',cfg.ratio_code)

cfg.log_name = 'vgg' + str(cfg.wbit_code).replace(', ','') \
                + str(cfg.abit_code).replace(', ','') \
                + str(cfg.ratio_code).replace(', ','')

if cfg.pretrain:
  cfg.log_name += 'pretrained'

cfg.log_dir = os.path.join(cfg.root_dir, 'logs', cfg.log_name)
cfg.ckpt_dir = os.path.join(cfg.root_dir, 'ckpt', cfg.log_name)

os.makedirs(cfg.log_dir, exist_ok=True)
os.makedirs(cfg.ckpt_dir, exist_ok=True)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.use_gpu


def main():
  if cfg.cifar == 10:
    print('training CIFAR-10 !')
    dataset = torchvision.datasets.CIFAR10
  elif cfg.cifar == 100:
    print('training CIFAR-100 !')
    dataset = torchvision.datasets.CIFAR100
  else:
    assert False, 'dataset unknown !'

  print('==> Preparing data ..')
  train_dataset = dataset(root=cfg.data_dir, train=True, download=True,
                          transform=cifar_transform(is_training=True))
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.train_batch_size, shuffle=True,
                                             num_workers=cfg.num_workers)

  eval_dataset = dataset(root=cfg.data_dir, train=False, download=True,
                         transform=cifar_transform(is_training=False))
  eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=cfg.eval_batch_size, shuffle=False,
                                            num_workers=cfg.num_workers)
  
  if cfg.model_name == 'resnet':
    print('==> Building ResNet..')
    model = resnet20(wbits=cfg.Wbits, abits=cfg.Abits).cuda()
  if cfg.model_name == 'vgg':
    print('==> Building VGGSMALL..')
    model = VGG_Cifar10(ratio_code = cfg.ratio_code, 
                        wbit_code = cfg.wbit_code, 
                        abit_code = cfg.abit_code, num_classes=10).cuda()

  optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.wd)
  lr_schedu = optim.lr_scheduler.MultiStepLR(optimizer, [60, 120, 180], gamma=0.1)
  criterion = torch.nn.CrossEntropyLoss().cuda()
  summary_writer = SummaryWriter(cfg.log_dir)

  if cfg.pretrain:
    model.load_state_dict(torch.load(cfg.pretrain_dir))

  # Training
  def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()

    start_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
      outputs = model(inputs.cuda())
      loss = criterion(outputs, targets.cuda())

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if batch_idx % cfg.log_interval == 0:
        step = len(train_loader) * epoch + batch_idx
        duration = time.time() - start_time

        print('%s epoch: %d step: %d cls_loss= %.5f (%d samples/sec)' %
              (datetime.now(), epoch, batch_idx, loss.item(),
               cfg.train_batch_size * cfg.log_interval / duration))

        start_time = time.time()
        summary_writer.add_scalar('cls_loss', loss.item(), step)
        summary_writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], step)

  def test(epoch):
    # pass
    model.eval()
    correct = 0
    for batch_idx, (inputs, targets) in enumerate(eval_loader):
      inputs, targets = inputs.cuda(), targets.cuda()

      outputs = model(inputs)
      _, predicted = torch.max(outputs.data, 1)
      correct += predicted.eq(targets.data).cpu().sum().item()

    acc = 100. * correct / len(eval_dataset)
    print('%s------------------------------------------------------ '
          'Precision@1: %.2f%% \n' % (datetime.now(), acc))
    summary_writer.add_scalar('Precision@1', acc, global_step=epoch)

  for epoch in range(cfg.max_epochs):
    lr_schedu.step(epoch)
    train(epoch)
    test(epoch)
    torch.save(model.state_dict(), os.path.join(cfg.ckpt_dir, 'checkpoint.t7'))

  summary_writer.close()


if __name__ == '__main__':
  main()
