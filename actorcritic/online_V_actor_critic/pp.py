import torch
import torch.nn as nn


print("name", __name__)
m = nn.LogSoftmax(dim=1)
s = nn.Softmax(dim=1)
T =torch.Tensor([[ 35647.1836,   -277.2057,   5510.0200,   1375.9424, -15350.5791,-7701.7388, -14281.4453,  -9288.6318]])
T =torch.Tensor([[-0.1611, -0.1116, -0.2468, -0.2568, -0.2881, -0.0146,  0.1602, -0.1064]])
print("logsoftmax", m(T))
print("softmax",s(T), "sum", s(T).sum())

