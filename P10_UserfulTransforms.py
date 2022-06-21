from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer=SummaryWriter("logs")
img=Image.open("image_user/mfj1.jpg")
print(img)

#Tornsor
trans_totensor=transforms.ToTensor()#调用对象
img_tensor=trans_totensor(img)#给对象一个返回值
writer.add_image("Totensor",img_tensor)


# Normalize
print(img_tensor[0][0][0])
trans_norm=transforms.Normalize([1,2,3],[4,3,6])
img_norm=trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalize",img_norm,2)

#Resize
print(img.size)
trans_resize=transforms.Resize((512,512))
img_resize=trans_resize(img)
img_resize=trans_totensor(img_resize)
writer.add_image("Resize",img_resize,0)
print(img_resize)

#Compose -resize-2
trans_resize_2=transforms.Resize(512)
trans_compose=transforms.Compose([trans_resize_2,trans_totensor])
img_resize_2=trans_compose(img)
writer.add_image("Resize",img_resize_2,1)


writer.close()
