from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image


writer =SummaryWriter("logs")
image_path="data/train/ants_image/0013035.jpg"
img_PIL=Image.open(image_path)
img_array=np.array(img_PIL)
print(type(img_array))
print(img_array.shape)

writer.add_image("test",img_array,1,dataformats='HWC')##从PIL到numpy,需要进行换dataformats的操作
for i in range(100):
    writer.add_scalar("y=2x",2*i,i)
writer.close()

