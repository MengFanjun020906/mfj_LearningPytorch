import torch
from torch import nn


class MFJ(nn.Module):
    def __init__(self) :
        super().__init__()

    def forward(self,input):
        output=input+1
        return output


mfj=MFJ()
x=torch.tensor(1.0)
output=mfj(x)
print(output)
