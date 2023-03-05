# Author Dr. M. Alwarawrah
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import math, os, time, scipy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
# PyTorch Library
import torch

# Import Class Linear
from torch.nn import Linear

# Library for this section
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torch import sigmoid
from matplotlib.colors import ListedColormap
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib as plm

from sklearn.metrics import (r2_score,roc_auc_score,hinge_loss,confusion_matrix,classification_report,mean_squared_error,jaccard_score,log_loss)

plm.rcParams.update({'figure.max_open_warning': 0})

torch.manual_seed(4)

#print if the code is using GPU/CUDA or CPU
if torch.cuda.is_available() == True:
    print('This device is using CUDA')
    device = torch.device("cuda:0")
    os.environ['CUDA_VISIBLE_DEVICES'] ='0'
else:    
    print('This device is using CPU')
    device = torch.device("cpu")


# start recording time
t_initial = time.time()

#class model for convolutional neural network
class CNN(nn.Module):
    
    # Contructor
    def __init__(self, out_1, out_2,number_of_classes):
        super(CNN, self).__init__()
        #1st convlutional neural network with 1, out_1, kernel_size = 5 and padding =2
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=out_1, kernel_size=5, padding=2)
        #Batch with out_1
        self.conv1_bn = nn.BatchNorm2d(out_1)
        #maxpooling with kernel_size =2
        self.maxpool1=nn.MaxPool2d(kernel_size=2)

        #2nd convlutional neural network with out_1, out_2, kernel_size = 5, stride=1 and padding =2        
        self.cnn2 = nn.Conv2d(in_channels=out_1, out_channels=out_2, kernel_size=5, stride=1, padding=2)
        #Batch with out_2
        self.conv2_bn = nn.BatchNorm2d(out_2)
        #maxpooling with kernel_size =2
        self.maxpool2=nn.MaxPool2d(kernel_size=2)
        #linear neural network with dimension input of out_2*4*4 and nymber_of_classes
        self.fc1 = nn.Linear(out_2 * 4 * 4, number_of_classes)
        #Batch with number_of_classes
        self.bn_fc1 = nn.BatchNorm1d(number_of_classes)
    
    # Prediction
    def forward(self, x):
        #convolution part
        #1st CNN
        x = self.cnn1(x)
        x=self.conv1_bn(x)
        x = torch.relu(x)
        x = self.maxpool1(x)
        #2nd CNN
        x = self.cnn2(x)
        x=self.conv2_bn(x)
        x = torch.relu(x)
        x = self.maxpool2(x)
        #flatten
        x = x.view(x.size(0), -1)
        #linear NN
        x = self.fc1(x)
        x=self.bn_fc1(x)
        return x

#accuracy per category:
def acc_per_category(Y_test,yhat, number_of_classes):
    #accuract for each label
    c = (yhat == Y_test).squeeze() 
    class_correct = [0 for i in range(number_of_classes)]
    total_correct = [0 for i in range(number_of_classes)]    
    for j in range(0, len(Y_test)):
        label = Y_test[j]
        class_correct[label] += c[j].item()
        total_correct[label] += 1
    
    acc_per_categ = []    
    for j in range(0,number_of_classes):    
        acc_per_categ.append((class_correct[j]/total_correct[j])*100)
    return acc_per_categ

#training model
def train_model(model,train_loader,validation_loader,optimizer,n_epochs,N_test, N_train, X_test, Y_test, category, num_classes, output_file):
    #define list
    train_loss=[]
    train_acc=[]
    val_loss=[]
    val_acc=[]
    for epoch in range(n_epochs):
        #set initial values for loss and accuracy sum to zero
        loss_sum = 0
        correct_sum=0
        #training part
        for x_train_batch, y_train_batch in train_loader:
            #load the data to device
            x_train_batch,y_train_batch = x_train_batch.to(device), y_train_batch.to(device)
            #train model
            model.train()
            #get predicition
            z = model(x_train_batch)
            #calculate loss
            loss = criterion(z, y_train_batch)
            #find the label based on max probability
            _, yhat = torch.max(z.data, 1)
            # sum the correct labels
            correct_sum += (yhat == y_train_batch).sum().item()
            #Sets the gradients of all optimized torch.Tensor s to zero        
            optimizer.zero_grad()
            # computes dloss/dx for every parameter x        
            loss.backward()
            #Performs a single optimization step (parameter update)        
            optimizer.step()
            #sum loss
            loss_sum += loss.data.item()
        #append loss
        train_loss.append(loss_sum)
        #calculate accuracy
        accuracy = 100*(correct_sum / N_train)
        #append accuracy    
        train_acc.append(accuracy)
        #print the training infor to screen and file
        print('Training, Epoch: {}, Loss: {:.2f}, Accuracy: {:.2f}, Classified: {}, Misclassifier: {}'.format(epoch,loss_sum,accuracy, correct_sum, N_train-correct_sum))    
        print('Training, Epoch: {}, Loss: {:.2f}, Accuracy: {:.2f}, Classified: {}, Misclassifier: {}'.format(epoch,loss_sum,accuracy, correct_sum, N_train-correct_sum), file=output_file)

        #set initial values for loss and accuracy sum to zero
        loss_sum = 0
        correct_sum=0
        #perform a prediction on the validation  data  
        for x_test_batch, y_test_batch in validation_loader:
            #load the data to device
            x_test_batch, y_test_batch = x_test_batch.to(device), y_test_batch.to(device)
            #test model
            model.eval()
            #get predicition
            z = model(x_test_batch)
            #calculate loss           
            loss = criterion(z, y_test_batch)
            #find the label based on max probability
            _, yhat = torch.max(z.data, 1)
            # sum the correct labels
            correct_sum += (yhat == y_test_batch).sum().item()
            #sum loss
            loss_sum += loss.data.item()
        #append loss
        val_loss.append(loss_sum)
        #calculate accuracy
        accuracy = 100*(correct_sum / N_test)
        #append accuracy    
        val_acc.append(accuracy)
        #print the validation infor to screen and file
        print('Validation, Epoch: {}, Loss: {:.2f}, Accuracy: {:.2f}, Classified: {}, Misclassified: {}'.format(epoch,loss_sum,accuracy, correct_sum, N_test-correct_sum))    
        print('Validation, Epoch: {}, Loss: {:.2f}, Accuracy: {:.2f}, Classified: {}, Misclassified: {}'.format(epoch,loss_sum,accuracy, correct_sum, N_test-correct_sum), file=output_file)
    
    #empty cuda cache
    #torch.cuda.empty_cache()
    
    #validation confusion matrix and plotting outputs
    X_test, Y_test = X_test.to(device), Y_test.to(device)        
    model.eval()
    z = model(X_test)
    _, yhat = torch.max(z.data, 1)
    #print accuracy per category
    acc_per_categ = acc_per_category(Y_test,yhat, number_of_classes)
    for i in range(0, number_of_classes):
        print('Epoch: {}, category: {}, accuracy: {:.2f}'.format(epoch, category[i], acc_per_categ[i]))      
        print('Epoch: {}, category: {}, accuracy: {:.2f}'.format(epoch, category[i], acc_per_categ[i]), file=output_file)      
    #confusion matrix
    conf_mat(Y_test.cpu(), yhat.cpu(), category, 'validation')
    #plotting outputs
    plot_outpput(yhat, X_test.cpu(), Y_test, 'validation')

    return train_loss, train_acc, val_loss, val_acc

#plot Loss and Accuracy vs epoch
def plot_loss_accuracy(train_loss, val_loss, train_acc, val_acc):
    plt.clf()
    fig,ax = plt.subplots()
    ax.plot(train_loss, color='k', label = 'Training Loss')
    ax.plot(val_loss, color='r', label = 'Validation Loss')
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Loss', fontsize=16)
    ax2 = ax.twinx()
    ax2.plot(train_acc, color='b', label = 'Training Accuracy')
    ax2.plot(val_acc, color='g', label = 'Validation Accuracy')
    ax2.set_ylabel('Accuracy', fontsize=16)
    fig.legend(loc ="center")
    fig.tight_layout()
    plt.savefig('loss_accuracy_epoch.png')

#plot outputs
def plot_outpput(y_actual, X_test, Y_test, name):
    plt.clf()
    N = 10#int(np.sqrt(N_classified))+1
    fig=plt.figure(figsize=(N, N))
    rows = N
    cols = N
    counter = 1
    for i in range(0,len(Y_test)):
        if y_actual[i] == Y_test[i]:
            plt.subplot(rows, cols, counter)
            plt.imshow(np.reshape(X_test[i], (16,16)))
            plt.title('{}_{}'.format(Y_test[i], y_actual[i]))
            counter += 1
            if counter == N*N:# N_classified:
                break
    fig.tight_layout()        
    plt.savefig('test_classified_%s.png'%(name))

    plt.clf()
    N = 10#int(np.sqrt(N_misclassified))+1
    fig=plt.figure(figsize=(N, N))
    rows = N
    cols = N
    counter = 1
    for i in range(0,len(Y_test)):
        if y_actual[i] != Y_test[i]:
            plt.subplot(rows, cols, counter)
            plt.imshow(np.reshape(X_test[i], (16,16)))
            plt.title('{}_{}'.format(Y_test[i], y_actual[i]))
            counter += 1
            if counter == N*N:#N_misclassified:
                break
    fig.tight_layout()        
    plt.savefig('test_miclassified_%s.png'%(name))

#plot confusion matrix
def conf_mat(y_test, yhat, category, name):
    #calculate confusion matrix
    CM = confusion_matrix(y_test, yhat, labels=[0,1,2,3,4,5,6,7,8,9])

    #plot confusion matrix
    plt.clf()
    fig, ax = plt.subplots()
    ax.matshow(CM, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(CM.shape[0]):
        for j in range(CM.shape[1]):
            ax.text(x=j, y=i,s=CM[i, j], va='center', ha='center', size='medium')
    plt.xticks(np.arange(0, 10, 1), category, fontsize = 8)
    plt.yticks(np.arange(0, 10, 1), category, fontsize = 8)
    plt.xlabel('Predictions', fontsize=16)
    plt.ylabel('Actuals', fontsize=16)
    plt.title('Confusion Matrix', fontsize=16)
    plt.savefig('confusion_matrix_%s.png'%(name))


output_file = open('output.txt','w')
print('Torch version: {}'.format(torch.__version__), file=output_file)

#image size
IMAGE_SIZE = 16
#create transformer to resize and transform to tensor
composed = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])

# Make the training set 
train_dataset = dsets.MNIST(root='./data', train=True, download=True, transform=composed)
# Make the validating set
validation_dataset = dsets.MNIST(root='./data', train=False, download=True, transform=composed)
#train and validation loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=500)

#Number of samples
N_test = len(validation_dataset)
N_train = len(train_dataset)

#create train and test data sets by setting batch_size to N
train_loader2 = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=N_train)
X_train, Y_train = next(iter(train_loader2))
validation_loader2 = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=N_test)
X_test, Y_test = next(iter(validation_loader2))

out_1=16 
out_2=32
number_of_classes=10
category = ['0','1','2','3','4','5','6','7','8','9']

#you can also use Sequential
'''
model = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=out_1, kernel_size=5, padding=2),
                      nn.BatchNorm2d(out_1), nn.MaxPool2d(kernel_size=2),
                      nn.Conv2d(in_channels=out_1, out_channels=out_2, kernel_size=5, stride=1, padding=2),
                      nn.BatchNorm2d(out_2), nn.MaxPool2d(kernel_size=2),
                      nn.Flatten(1, -1),
                      nn.Linear(out_2 * 4 * 4, number_of_classes), nn.BatchNorm1d(number_of_classes)
                      )
'''
# Create the model object using CNN class batch
model=CNN(out_1, out_2, number_of_classes)
#load model to device
model = model.to(device)
#define criterion to calculate loss
criterion = nn.CrossEntropyLoss()
learning_rate = 0.1
#define optimizer
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
epochs = 10
#train model and retrieve losses and accuracy for training and validation
train_loss, train_acc, val_loss, val_acc = train_model(model,train_loader,validation_loader,optimizer,epochs, N_test, N_train, X_test, Y_test, category, number_of_classes, output_file)

#plot train and validation loss and accuracy
plot_loss_accuracy(train_loss, val_loss, train_acc, val_acc)

output_file.close()

#End recording time
t_final = time.time()

t_elapsed = t_final - t_initial
hour = int(t_elapsed/(60.0*60.0))
minute = int(t_elapsed%(60.0*60.0)/(60.0))
second = t_elapsed%(60.0)
print("%d h: %d min: %f s"%(hour,minute,second))
