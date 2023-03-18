import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
from torch.autograd import Variable as V
from torchvision import transforms as trn
from torch.nn import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time
use_cuda = True

#region HELPER FUNCTIONS

#calculate output size of a convolution
def calc_output_size(image_size, kernal_size, stride=1, padding = 0):
  output = ((image_size + 2*padding - kernal_size)/ stride) + 1
  return (output)

#calculate padding needed to keep image same size
def calc_pad(kernel_size):
  return (kernel_size - 1) // 2

def get_model_name(name, batch_size, learning_rate, epoch):
    """ Generate a name for the model consisting of all the hyperparameter values

    Args:
        config: Configuration object containing the hyperparameters
    Returns:
        path: A string with the hyperparameter name and value concatenated
    """
    path = "model_{0}_bs{1}_lr{2}_epoch{3}".format(name,
                                                   batch_size,
                                                   learning_rate,
                                                   epoch)
    return path
    
def plot_training_curve(path):
    """ Plots the training curve for a model run, given the csv files
    containing the train/validation error/loss.

    Args:
        path: The base path of the csv files produced during training
    """
    import matplotlib.pyplot as plt
    train_err = np.loadtxt("{}_train_err.csv".format(path))
    val_err = np.loadtxt("{}_val_err.csv".format(path))
    train_loss = np.loadtxt("{}_train_loss.csv".format(path))
    val_loss = np.loadtxt("{}_val_loss.csv".format(path))
    plt.title("Train vs Validation Error")
    n = len(train_err) # number of epochs
    plt.plot(range(1,n+1), train_err, label="Train")
    plt.plot(range(1,n+1), val_err, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend(loc='best')
    plt.show()
    plt.title("Train vs Validation Loss")
    plt.plot(range(1,n+1), train_loss, label="Train")
    plt.plot(range(1,n+1), val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()

def get_accuracy_transferNet(model, ResNet, data, batch_size=32):
    

    correct = 0
    total = 0
    for imgs, labels in torch.utils.data.DataLoader(data, batch_size=batch_size):
        
        
        #############################################
        #To Enable GPU Usage
        if use_cuda and torch.cuda.is_available():
          imgs = imgs.cuda()
          labels = labels.cuda()
        #############################################
        
        
        ResNet_features = ResNet(imgs)
        outputs = model(ResNet_features)
        
        #select index with maximum prediction score
        pred = outputs.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += imgs.shape[0]
    return correct / total

def train_primary(model, ResNet, train_data, val_data, batch_size=32, num_epochs=30, learning_rate=0.001):

    """ Training loop. You should update this."""
    torch.manual_seed(42)
    #create dataloaders from image folders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(list(ResNet.parameters()) + list(model.parameters()), lr=learning_rate, momentum=0.9)


    iters, train_acc, train_loss, val_acc, val_loss = [], [], [], [], []

    start_time = time.time()
    n = 0 
    for epoch in range(num_epochs):
        for imgs, labels in iter(train_loader):

            #############################################
            # To Enable GPU Usage
            if use_cuda and torch.cuda.is_available():
              imgs = imgs.cuda()
              labels = labels.cuda()
            #############################################

            ResNet_features = ResNet(imgs)

            outputs = model(ResNet_features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        #took outside forloop because doing it for each iteration is taking way too long. Put back inside 
        #to do for each iteration. Right now its doing for each epoch
        train_acc.append(get_accuracy_transferNet(model, ResNet, train_data, batch_size= batch_size)) # compute training accuracy 
        train_loss.append(float(loss)/ batch_size)
        

        for imgs, labels in iter(val_loader):

            #############################################
            # To Enable GPU Usage
            if use_cuda and torch.cuda.is_available():
              imgs = imgs.cuda()
              labels = labels.cuda()
            #############################################

        #compute validation loss
        
            ResNet_features_val = ResNet(imgs)
            outputs_val = model(ResNet_features_val)
            val_loss_value = criterion(outputs_val, labels)

        #took outside forloop because doing it for each iteration is taking way too long. Put back inside 
        #to do for each iteration. Right now its doing for each epoch
        val_acc.append(get_accuracy_transferNet(model, ResNet, val_data, batch_size=batch_size))
        val_loss.append(float(val_loss_value)/ batch_size) 


        iters.append(n)
        n += 1
      
        print(("Epoch {}: Train acc: {} |"+"Validation acc: {}").format(
                epoch + 1,
                train_acc[-1],
                val_acc[-1]))
        # Save the current model (checkpoint) to a file at each epoch
        model_path = get_model_name(model.name, batch_size, learning_rate, epoch)
        torch.save(model.state_dict(), model_path)
    print('Finished Training')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Total time elapsed: {:.2f} seconds".format(elapsed_time))

    # Plotting
    plt.title("Training Curve (Loss)")
    plt.plot(iters, train_loss, label="Train")
    plt.plot(iters, val_loss, label="Validation")
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
#endregion

#region PRIMARY MODEL ARCHITECTURE
class PrimaryCNN(nn.Module):
  def __init__(self):
    super(PrimaryCNN, self).__init__()
    self.name = 'PrimaryCNN'

    #transpose convolutional layer
    self.transconv = nn.ConvTranspose2d(2048, 2048, kernel_size=7, stride=1, padding=0, output_padding=0)
    # #stage 1
    self.conv1_0 = nn.Conv2d(2048, 5, 3, 1, 1)
    self.conv1_1 = nn.Conv2d(5, 5, 3, 1, 1)
    # #maxpool
    self.pool = nn.MaxPool2d(2, 2)
    # #stage 2
    self.conv2_0 = nn.Conv2d(5, 10, 3, 1, 1)
    self.conv2_1 = nn.Conv2d(10, 10, 3, 1, 1)
    # #stage 3 
    self.conv3_0 = nn.Conv2d(10, 20, 3, 1, 1)
    self.conv3_1 = nn.Conv2d(20, 20, 3, 1, 1)
    # #stage 4 
    self.conv4_0 = nn.Conv2d(20, 40, 3, 1, 1)
    self.conv4_1 = nn.Conv2d(40, 40, 3, 1, 1)

    #same size padding is applied throughout so input images are still 7x7
    self.fc1 = nn.Linear(3 * 3 * 40, 70)
    self.fc2 = nn.Linear(70, 5)

  def forward(self, x):
    
    x = F.relu(self.transconv(x))
    # #stage 1 with pool on last layer
    x = F.relu(self.conv1_0(x))   
    x = F.relu(self.conv1_1(x))  
    x = F.relu(self.conv1_1(x))
    x = F.relu(self.conv1_1(x))
    
    # #stage 2 with pool on last layer
    x = F.relu(self.conv2_0(x))
    x = F.relu(self.conv2_1(x)) 
    x = F.relu(self.conv2_1(x))
    x = F.relu(self.conv2_1(x))
  
    # #stage 3 with pool on last layer
    x = F.relu(self.conv3_0(x))
    x = F.relu(self.conv3_1(x))
    x = F.relu(self.conv3_1(x))
    x = F.relu(self.conv3_1(x))
    
    # #stage 4 with pool on last layer
    x = F.relu(self.conv4_0(x))
    x = F.relu(self.conv4_1(x))
    x = F.relu(self.conv4_1(x)) 
    x = self.pool(F.relu(self.conv4_1(x)))

    #Fully connected layer (head)
    x = x.view(-1,  3 * 3 * 40)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    x = x.squeeze(1)
    return x
#endregion

#region IMPLEMENTATION 

# Load Dataset
transform = trn.Compose([
    trn.Resize(224),
    trn.CenterCrop(224),
    trn.ToTensor(),
    trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

img_data_filtered = torchvision.datasets.ImageFolder('/content/drive/MyDrive/APS360 Project/na', transform= transform) # PATH WHERE FILTERED DATASET IS LOCATED

train_size = int(0.6 * len(img_data_filtered)+1)
test_size = int((len(img_data_filtered) - train_size) / 2)
val_size = int(test_size)

train_data_filtered, val_data_filtered, test_data_filtered = torch.utils.data.random_split(img_data_filtered,  [train_size, val_size, test_size])

# Import Pre-Trained ResNet50 Model (Transfer Learning)
fine_tuned_ResNet = models.resnet50(pretrained=True)

# remove fully connected layer
fine_tuned_ResNet = torch.nn.Sequential(*list(fine_tuned_ResNet.children())[:-1])
fine_tuned_ResNet.train()

#unfreeze all weights in 4th stage (set autograd to true so the gradients are calculated during back prop)
for name, param in fine_tuned_ResNet.named_parameters():
    if name.startswith('layer4'):
        param.requires_grad = True 
    else:
        param.requires_grad = False

# Import Primary Model
primary_Model = PrimaryCNN()

# Train
if use_cuda and torch.cuda.is_available():
  primary_Model.cuda()
  fine_tuned_ResNet.cuda()
  print('CUDA is available! Training on GPU ...')
else:
  print('CUDA is not available. Training on CPU ...')

train_primary(primary_Model, fine_tuned_ResNet, train_data_filtered, val_data_filtered, batch_size=64, num_epochs=15, learning_rate=0.01)

#endregion
