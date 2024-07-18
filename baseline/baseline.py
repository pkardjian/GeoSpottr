import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.autograd import Variable as V
from torchvision import transforms
from torch.nn import functional as F
import matplotlib.pyplot as plt
import pandas
use_cuda = True


transformations = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()]) 
img_data = torchvision.datasets.ImageFolder(r'/Users/pascalkardjian/Downloads/small/', transform = transformations) #PATH WHERE FILTERED IMAGES ARE LOCATED

train_data, val_data, test_data = torch.utils.data.random_split(img_data,[0.6,0.2,0.2])

class BaselineCNN(nn.Module):
    def __init__(self):
        super(BaselineCNN, self).__init__()
        self.name = "BaselineCNN"
        self.conv1 = nn.Conv2d(3, 5, kernel_size=5) # s=1 & p=0; gives 220x220 output feature maps
        self.pool = nn.MaxPool2d(2, 2) # s=2, p=0 & kernel_size=2; gives 110x110 output feature maps
        self.conv2 = nn.Conv2d(5, 5, kernel_size=5) # s=1, p=0 & kernel_size=5; gives 106x106 then 53x53 output feature maps AFTER POOLING
        self.fc1 = nn.Linear(5*53*53, 32) # output size = 32 
        self.fc2 = nn.Linear(32, 4) # 21 possible outputs

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 5*53*53)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.squeeze(1)
        return x


def train(model, train_data, val_data, batch_size=32, num_epochs=10, learn_rate=0.001):
    train_load = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    val_load = torch.utils.data.DataLoader(val_data, batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)

    iters, losses, train_acc, val_acc = [], [], [], []

    # training
    n = 0 # the number of iterations
    for epoch in range(num_epochs):
        for imgs, labels in iter(train_load):

            #############################################
            #To Enable GPU Usage
            if use_cuda and torch.cuda.is_available():
              imgs = imgs.cuda()
              labels = labels.cuda()
            #############################################
              
            out = model(imgs)             # forward pass

            loss = criterion(out, labels) # compute the total loss
            loss.backward()               # backward pass (compute parameter updates)
            optimizer.step()              # make the updates for each parameter
            optimizer.zero_grad()         # a clean up step for PyTorch

            # save the current training information
            iters.append(n)
            losses.append(float(loss)/batch_size)             # compute *average* loss

        train_acc.append(get_accuracy(model, train_data, val_data, batch_size=batch_size, train=True)) # compute training accuracy 
        val_acc.append(get_accuracy(model, train_data, val_data, batch_size=batch_size, train=False))  # compute validation accuracy


        n += 1

        print(("Epoch {}: Train Accuracy: {} |"+
               " Validation Accuracy: {} ").format(
                   epoch + 1,
                   train_acc[epoch],
                   val_acc[epoch]))

    

    # plotting
    plt.title("Training Curve")
    plt.plot(iters, losses, label="Train")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()

    plt.title("Training Curve (Accuracy)")
    plt.plot(range(1 ,num_epochs+1), train_acc, label="Train")
    plt.plot(range(1 ,num_epochs+1), val_acc, label="Validation")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.show()

    print("Final Training Accuracy: {}".format(train_acc[-1]))
    print("Final Validation Accuracy: {}".format(val_acc[-1]))


def get_accuracy(model, train_data, val_data, batch_size=32, train=False):
    if train:
        data = train_data
    else:
        data = val_data

    correct = 0
    total = 0
    for imgs, labels in torch.utils.data.DataLoader(data, batch_size=batch_size):
        
        
        #############################################
        #To Enable GPU Usage
        if use_cuda and torch.cuda.is_available():
          imgs = imgs.cuda()
          labels = labels.cuda()
        #############################################
        
        
        output = model(imgs)
        
        #select index with maximum prediction score
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += imgs.shape[0]
    return correct / total

use_cuda = True

cnn = BaselineCNN()

if use_cuda and torch.cuda.is_available():
  cnn.cuda()
  print('CUDA is available!  Training on GPU ...')
else:
  print('CUDA is not available.  Training on CPU ...')

train(cnn, train_data, val_data)
torch.save(cnn.state_dict(), 'model_parameters')
