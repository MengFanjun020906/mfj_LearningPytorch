import torchvision.datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data=torchvision.datasets.CIFAR10("./dataset2",train=False,transform=torchvision.transforms.ToTensor())

test_loader=DataLoader(dataset=test_data,batch_size=64,shuffle=True,num_workers=0,drop_last=True)

#测试数据集中第一张图片及target
img,target=test_data[0]
print(img.shape)

print(target)

writer=SummaryWriter("dataloader")
step=0
for data in test_loader:
    imgs,targets=data
    writer.add_images("test_data_drop_last",imgs,step)
    step=step+1

writer.close()