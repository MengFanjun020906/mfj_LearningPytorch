from torch import nn
import torch
class NetWork(nn.Module):
    def __init__(self):
        super(NetWork, self).__init__()
        self.model=nn.Sequential(
            nn.Conv2d(3,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4,64),
            nn.Linear(64,10)
        )#在序列中去填网络结构

    def forward(self,x):
        x=self.model(x)
        return x



if __name__ == '__main__':
     network=NetWork()
     input=torch.ones((64,3,32,32))
     output=network(input)
     print(output.shape)