import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


out_np = np.zeros((1, 21, 3, 3))
out_np[0, 0, 0, 0] = 0
out_np[0, 1, 1, 0] = 1
out_np[0, 2, 2, 0] = 1
out_np[0, 3, 0, 1] = 1
out_np[0, 4, 1, 1] = 1
out_np[0, 5, 2, 1] = 1
out_np[0, 6, 0, 2] = 1
out_np[0, 7, 1, 2] = 1
out_np[0, 8, 2, 2] = 1

target_np = np.zeros((1, 3, 3))
target_np[0, 0, 0] = 0
target_np[0, 1, 0] = 1
target_np[0, 2, 0] = 2
target_np[0, 0, 1] = 3
target_np[0, 1, 1] = 4
target_np[0, 2, 1] = 5
target_np[0, 0, 2] = 6
target_np[0, 1, 2] = 7
target_np[0, 2, 2] = 8


#out = out_np[0].argmax(axis=0)
#print(out)
#print(out.shape)
#print(target_np)
output = Variable(torch.FloatTensor(out_np))
target = Variable(torch.LongTensor(target_np))
criterion = nn.NLLLoss(size_average=False, reduce=False)
loss = criterion(output, target)
