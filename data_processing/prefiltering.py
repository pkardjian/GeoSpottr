import torch
import os
import csv
import torch.nn as nn
import torch.utils.data
import torchvision
import torchvision.models as models
from torchvision import transforms as trn
from torch.autograd import Variable as V
from torch.nn import functional as F
from PIL import Image



def main():
  use_cuda = True

  img_data = torchvision.datasets.ImageFolder(r'/Users/pascalkardjian/Downloads/output')

  # LOAD MODEL
  arch = 'resnet50'
  model_file = r'/Users/pascalkardjian/Downloads/resnet50_places365.pth.tar' 

  model = models.__dict__[arch](num_classes=365)
  checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
  state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
  model.load_state_dict(state_dict)
  model.eval()


  # LOAD CLASSES AND IMAGE TRANSORMATIONS
  centre_crop = trn.Compose([
          trn.Resize((256,256)),
          trn.CenterCrop(224),
          trn.ToTensor(),
          # trn.Lambda(lambda x: x[:3,:,:]),
          trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  ])
  centre_crop_og = trn.Compose([
          trn.ToTensor() 
  ])

  # Load the class labels
  file_name = r'/Users/pascalkardjian/Downloads/categories_places365.txt'
  classes = list()
  with open(file_name) as class_file:
      for line in class_file:
          classes.append(line.strip().split(' ')[0][3:])
  classes = tuple(classes)

  # Create directories to hold accepted images
  map = {
        0:"amsterdam",
        1:"berlin",
        2:"bogota",
        3:"buenosaires",
        4:"capetown",
        5:"chicago",
        6:"losangeles",
        7:"melbourne",
        8:"milan",
        9:"montevideo",
        10:"newyork",
        11:"riodejaneiro",
        12:"santiago",
        13:"seoul",
        14:"singapore",
        15:"taipei",
        16:"telaviv",
        17:"tokyo",
        18:"toronto",
        19:"vancouver",
        20:"vienna",
        21:"zurich",
        }

  for key, value in map.items():
      os.mkdir(r'/Users/pascalkardjian/Downloads/filtered' + f'/{value}')

  # Create image folder to hold original images as tensors (with no transformations applied)
  image_dataset_og = torchvision.datasets.ImageFolder(r'/Users/pascalkardjian/Downloads/output', transform = centre_crop_og)

  # List of acceptable classes from Places365 class list (these are the classes of images that will help in training our model)
  acceptable_classes = r'/Users/pascalkardjian/Downloads/Acceptable_Classes.txt'
  list_of_acceptable_classes = []

  with open(acceptable_classes) as csv_file:
      for line in csv_file:
        list_of_acceptable_classes.append(line[2:-3])


  # Create dataset to hold tranformed images to be used with classifier
  image_dataset = torchvision.datasets.ImageFolder(r'/Users/pascalkardjian/Downloads/output', transform = centre_crop)

  # FILTERING PROCESS
  data_loader = torch.utils.data.DataLoader(image_dataset, batch_size=1,num_workers=1)

  index = 0
  accepted = 0


  for i, data in enumerate(data_loader):
    input_img, labels = data
    input_img = V(input_img)
    logit = model.forward(input_img)
    h_x = F.softmax(logit, 1).data.squeeze()
    probs, idx = h_x.sort(0, True)

    sum_acceptable_classes = 0
    # Get sum of probs of acceptable classes in top 5 predicted probs
    for i in range(0,5):
      if classes[idx[i]] in list_of_acceptable_classes:
        sum_acceptable_classes += probs[i]

    if sum_acceptable_classes >= 0.5:
      accepted += 1
      to_pil_image = trn.ToPILImage()

      # Convert original tensor to PIL Image
      pil_image = to_pil_image(image_dataset_og[index][0])

      # Save the PIL Image as JPEG
      pil_image.save('/Users/pascalkardjian/Downloads/filtered/' + map[labels.item()] + '/image' + str(index) + '.jpg')

    index += 1
    if index % 100 == 0:
      print(f"Finished Processing {index} images.")
      print(f'{accepted} images accepted.\n')

if __name__ == '__main__':
   main()
