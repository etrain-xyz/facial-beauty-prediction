import torch
import torchvision.transforms as transforms

model_arch = 'resnet18'
epochs = 50

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

transform = transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

data_root = './SCUT-FBP5500_v2'

models_dir = './models'