from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image


img_path="dataset/train/ants/0013035.jpg"
img=Image.open(img_path)

writer=SummaryWriter("logs")

tensor_trans=transforms.ToTensor()
tensor_img=tensor_trans(img)

print(tensor_img)


writer.add_image("Tensor_img",tensor_img)

writer.close()