import torch
import torcheras
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
from PIL import Image

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu:0')

model = TaskModel(params).to(device)

test_img_path = 'data/blob_test_image_data/'

model = torcheras.Model(model, 'log')
model.load_model('', epoch=5)
model = model.model
model.eval()

with torch.no_grad():
    for img_path in os.listdir(test_img_path):
        img = np.array(Image.open(test_img_path+img_path), dtype=np.float32) / 255.
        img = torch.Tensor(img).to(device)
        y_pred = model(img)
        y_pred = F.sigmoid(y_pred)

        print(y_pred)