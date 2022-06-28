
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *
##准备数据集
train_data=torchvision.datasets.CIFAR10(root="../data",train=True,transform=torchvision.transforms.ToTensor(),
                                        download=True)##数据集的目录，是否训练，将PIL型变为Tensor型，需要下载

test_data=torchvision.datasets.CIFAR10(root="../data",train=False,transform=torchvision.transforms.ToTensor(),
                                        download=True)
#length 长度
train_data_size=len(train_data)
test_data_size=len(test_data)
#如果train_data_size为10，训练集的长度为:10
print("训练数据集长度为：{}".format(train_data_size))#{},会被替换为后面的值，把他变成字符串了
print("测试数据集长度为：{}".format(test_data_size))#{},会被替换为后面的值，把他变成字符串了


#利用dataloader来加载数据集
train_dataloader=DataLoader(train_data,batch_size=64)#一次训练所抓取的数据样本数量为64
test_dataloader=DataLoader(test_data,batch_size=64)

#创建网络模型
network=NetWork()

#损失函数
loss_fn=nn.CrossEntropyLoss()#交叉熵

#优化器
learning_rate=0.01#学习速率
optimizer=torch.optim.SGD(network.parameters(),lr=learning_rate)#随机梯度下降,先填网络模型，然后是学习速率


#设置训练网络的一些参数
total_train_step=0#记录训练次数
#记录测试的次数
total_test_step=0
#训练的轮数
epoch=10

#添加tensorboard
writer=SummaryWriter("logs_train")

for i in range(epoch):
    print("----------------第{}轮训练开始-------------".format(i+1))

    #训练步骤开始
    for data in train_dataloader:
        imgs,targets=data
        outputs=network(imgs)
        loss=loss_fn(outputs,targets)
        #优化器优化模型
        optimizer.zero_grad()#优化器梯度清零
        loss.backward()#反向传播
        optimizer.step()#优化器进行优化

        total_train_step=total_train_step+1 #记录训练次数
        if total_train_step %100==0:#每100次才打印
            print("训练次数：{},Loss:{}".format(total_train_step,loss.item()))
            writer.add_scalar("train_loss",loss.item(),total_train_step)##标题，loss值，训练次数
    #测试步骤开始
    total_test_loss=0
    total_accuracy=0#整体正确率
    with torch.no_grad():
        for data in test_dataloader:
            imgs,targets =data
            outputs =network(imgs)#输入放到网络当中我们可以得到对应输出
            loss =loss_fn(outputs,targets)#计算误差
            total_test_loss=total_test_loss+loss.item()
            accuracy=(outputs.argmax(1)==targets).sum()#横向准确率
            total_accuracy=total_accuracy+accuracy
    print("整体测试集上的Loss:{}".format(total_test_loss))
    print("整体测试集上的正确率:{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss",total_test_loss,total_test_step)##标题，测试误差值，测试次数
    writer.add_scalar("test_accuracy",total_accuracy/test_data_size,total_test_step)
    total_test_step=total_test_step+1


    torch.save(network,"network_{}.pth".format(i))#保存每一轮的训练结果
    print("模型已保存")




writer.close()







