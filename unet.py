# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 21:15:33 2021

@author: josef
"""
#! pip install albumentations==0.4.6
#from albumentations.pytorch import ToTensorV2
#Libaries
import torch
import numpy as np
import matplotlib.pyplot as plt

#For UNET architecture
import torch.nn as nn
import torchvision.transforms.functional as TF

#dataset de torch
from torch.utils.data import DataLoader, Dataset

##Para lectura de archivos desde github
from skimage import io
import requests
from bs4 import BeautifulSoup
import re

#Data augmentation
import albumentations as A

#Barra de progreso
from tqdm import tqdm

#Training and validation
from sklearn.model_selection import train_test_split


##url de imágenes y máscaras para lectura
inputs='https://raw.githubusercontent.com/pepetonof/unet_hu/main/input/'
masks= 'https://raw.githubusercontent.com/pepetonof/unet_hu/main/target/'

##Muestra de una de las '24' imágenes recabadas hasta ahora
# new_root = inputs+'input24.png'
# image = io.imread(new_root)
# io.imshow(image)
# io.show()

##List files on github repository
github_url_inp = 'https://github.com/pepetonof/unet_hu/tree/main/input'
github_url_msk = 'https://github.com/pepetonof/unet_hu/tree/main/target'

#Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
NUM_EPOCHS = 35
NUM_WORKERS = 0
IMAGE_HEIGHT = 40
IMAGE_WIDTH = 60
PIN_MEMORY = True

#Obtiene los nombres de los archivos de la liga a repositorio
def get_filenames(github_url:str):
    result = requests.get(github_url)
    soup = BeautifulSoup(result.text, 'html.parser')
    pngfiles = soup.find_all(title=re.compile("\.png$"))
    filename = []
    for i in pngfiles:
        filename.append(i.extract().get_text())
    
    return filename

#Determina el indice del titulo de las imágenes
def get_ind(string:str):
    num=""
    for i in string:
        if i.isdigit():
            num=num+i
    
    return int(num)

#Obtiene lista y la ordena con base en el indice de la imagen    
filenames_inp=get_filenames(github_url_inp)
filenames_msk=get_filenames(github_url_msk)
filenames_inp.sort(key=get_ind)
filenames_msk.sort(key=get_ind)

#Concatena con dirección de repositorio de imagen y máscara
def concat_git(url,lst:list):
    for i in range(len(lst)):
        lst[i]=url+lst[i]

#Determina los url de las imágenes
concat_git(inputs,filenames_inp)
concat_git(masks, filenames_msk)        

##Dataset
class PlacentaDataSet(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir= image_dir
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_dir)
    
    def __getitem__(self, index:int):
        img_id  =self.image_dir[index]
        mask_id =self.mask_dir[index]
    
        image=io.imread(img_id, as_gray=True).astype(np.float32)
        mask =io.imread(mask_id).astype(np.float32)
        mask[mask==255.0]=1.0
        
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image=augmentations["image"]
            mask=augmentations["mask"]
        
        #Convert into a Tensor
        image=torch.from_numpy(image)
        mask=torch.from_numpy(mask)
        image = image.unsqueeze(dim=0)
        #mask = mask.unsqueeze(dim=0)
        
        return image, mask
    
# ##Clase para convertir un objeto numpy HWC o HW en un tensor CHW
# class ToTensor(object):
#     def __call__(self, image, mask):
#         #Grayscale images
#         if len(image.shape)==2:
#             #tensor = torch.from_numpy(sample)        
#             #tensor = tensor.unsqueeze(dim=0)
            
#             img_tensor=torch.from_numpy(image)
#             msk_tensor=torch.from_numpy(mask)
#             img_tensor = img_tensor.unsqueeze(dim=0)
#             msk_tensor = msk_tensor.unsqueeze(dim=0)
        
#         # #In case input image RGB
#         # elif len(sample.shape)==3:
#         #     sample=sample.transpose(2,0,1)
#         #     tensor=torch.from_numpy(sample)
        
#         # else:
#         #     print('Error Input Dimension')
        
#         return img_tensor, msk_tensor

##Transformations for Data Augmentation and preprocessing
transform=A.Compose(
    [
     A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
     A.Normalize(
         mean=[0.0],
         std=[1.0],
         max_pixel_value=255.00,
         ),
    ]
    )

##Split into train and valid data to 70% and 30%
train_size=0.7
inputs_train, inputs_val, targets_train, targets_val = train_test_split(
    filenames_inp,
    filenames_msk,
    train_size=train_size,
    #random_state=random_seed,
    shuffle=True
)

##Function to get DataLoaders
def get_loaders(
        train_dir,
        train_maskdir,
        
        val_dir,
        val_maskdir,
        
        batch_size,
        transform,
        num_workers=1,
        pin_memory=True
        ):
    
    train_ds=PlacentaDataSet(image_dir=train_dir, mask_dir=train_maskdir, transform=transform)
    valid_ds=PlacentaDataSet(image_dir=val_dir, mask_dir=val_maskdir, transform=transform)
    
    train_loader=DataLoader(train_ds,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                            shuffle=True,)
    
    val_loader=DataLoader(valid_ds,
                          batch_size=8,#batch_size,
                          num_workers=num_workers,
                          pin_memory=pin_memory,
                          shuffle=False,)
    
    return train_loader, val_loader, train_ds, valid_ds

##Get DataLoaders
train_dl, valid_dl, train_ds, valid_ds = get_loaders(
                                 inputs_train,
                                 targets_train,
                                 inputs_val,
                                 targets_val,
                                 
                                 BATCH_SIZE,
                                 transform,
                                 NUM_WORKERS,
                                 )

##############################################################################
####Arquitectura UNET
#Doble convolución
class DoubleConv(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(DoubleConv, self).__init__()

    self.conv=nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),

        nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
      )
  
  def forward(self, x):
    return self.conv(x)

#Unet
class UNET(nn.Module):
    def __init__( 
        self, in_channels=1, out_channels=1, features=[64, 128, 256, 512],
      ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
        #Down part of UNET
        for feature in features:
          self.downs.append(DoubleConv(in_channels, feature))
          in_channels = feature
        
        #Uppart of UNET
        for feature in reversed(features):
          self.ups.append(
              nn.ConvTranspose2d(
              feature*2, feature, kernel_size=2, stride=2,
              )
          )
          self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck= DoubleConv(features[-1], features[-1]*2)
        self.final_conv= nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self,x):
      skip_connections =[]
      for down in self.downs:
        x=down(x)
        skip_connections.append(x)
        x = self.pool(x)
      
      x=self.bottleneck(x)
      skip_connections=skip_connections[::-1]
      for idx in range(0, len(self.ups), 2):
        x= self.ups[idx](x)
        skip_connection = skip_connections[idx//2]

        if x.shape != skip_connection.shape:
          x=TF.resize(x, size=skip_connection.shape[2:])

        concat_skip=torch.cat((skip_connection,x),dim=1)
        x=self.ups[idx+1](concat_skip)

      return self.final_conv(x)
###############################################################################

##Modelo, función de costo para entrenamiento y optimizador Adam para aprendizaje
model = UNET(in_channels=1, out_channels=1).to(DEVICE)
loss_fn = nn.BCEWithLogitsLoss() #cross entropy loss
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

##train function to 1 epoch
def train_fn(loader, model, optimizer, loss_fn, scaler):
  loop = tqdm(loader)
  for batch_idx, (data, targets) in enumerate(loop):
    data=data.to(device=DEVICE)
    #print('data_shape',data.shape)
    
    #float for Binary Cross Entropy Loss
    #its already float
    targets=targets.float().unsqueeze(1).to(device=DEVICE)
    #print('targets_shape',targets.shape)
    
    #forward
    if torch.cuda.is_available():
        with torch.cuda.amp.autocast():
          predictions=model(data)
          loss = loss_fn(predictions, targets) #.type(torch.int64))
    else:
        predictions=model(data)
        loss = loss_fn(predictions, targets)

    #backward
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    #update tqdm loop
    loop.set_postfix(loss=loss.item())


def check_accuracy(loader, model, device="cuda"):
  num_correct=0
  num_pix=0
  dice_score=0
  model.eval()
  with torch.no_grad():
     for x,y in loader:
       x=x.to(device)
       y=y.to(device)
       preds = torch.sigmoid(model(x))
       preds = (preds > 0.5).float()
      
       num_correct += (preds==y).sum()
       num_pix += torch.numel(preds)
       dice_score += (2*(preds*y).sum()) / ((preds+y).sum() + 1e-8)
       
  print('\n')
  print(f"Got {num_correct} / {num_pix} with acc {num_correct/num_pix*100:.2f}")
  print(f"Dice score: {dice_score/len(loader)}")
  
  model.train()

check_accuracy(valid_dl, model, device=DEVICE)
scaler=torch.cuda.amp.GradScaler()

#Training
for epoch in range (NUM_EPOCHS):
  train_fn(train_dl, model, optimizer, loss_fn, scaler)
  #check accuracy
  check_accuracy(valid_dl, model, device=DEVICE)
  

#Convert the idx element from tensor "xb" to image
def batch_to_img(xb, idx):
    img = np.array(xb[idx,0:3])
    img = img.transpose((1,2,0))
    img = np.squeeze(img)
    return img

##Compare predictions against masks and original images
model.eval() 
xb, yb = next(iter(valid_dl))
xb=xb.to(DEVICE)
yb=yb.to(DEVICE) 
preds = torch.sigmoid(model(xb))
preds = (preds > 0.5).float()

bs = 1
fig, ax = plt.subplots(bs, 3, figsize=(15,bs*5))
for i in range(bs):
    ax[0].imshow(batch_to_img(xb, i),cmap='gray')
    ax[1].imshow(yb[i].cpu(),cmap='gray')
    ax[2].imshow(batch_to_img(preds, i),cmap='gray')
    ax[0].set_title("Image")
    ax[1].set_title("Ground truth mask")
    ax[2].set_title("Predicted")