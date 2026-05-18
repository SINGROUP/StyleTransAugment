import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# Define the CNN model
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.BatchNorm2d(32), 
            nn.MaxPool2d(2))
        
        self.conv_layer_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1), 
            nn.ReLU(), 
            nn.BatchNorm2d(64), 
            nn.MaxPool2d(2))

        # Assuming the input image size is 192x192, after two max pooling layers, the feature map size would be 48x48
        # And the number of feature maps would be 64. So the total number of features going into the linear layer would be 48*48*64
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=48*48*64, out_features=1), 
            nn.Sigmoid())

    def forward(self, x):
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        x = self.classifier(x)
        return x

# Add a method to obtain the image path
class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
    # https://gist.github.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d
    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def plot_confusion_matrix(confusion_mat, class_names, file_name='confusion.pdf'):
    accuracy = (confusion_mat[0, 0] + confusion_mat[1, 1]) / np.sum(confusion_mat)
    plt.figure(figsize=(5, 5))
    plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Accuracy: {:.2%}'.format(accuracy))
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, ['PPAFM', 'AFM'], rotation=0)
    plt.yticks(tick_marks, ['PPAFM', 'AFM'], rotation=90)
    
    thresh = confusion_mat.max() / 2.
    for i in range(confusion_mat.shape[0]):
        for j in range(confusion_mat.shape[1]):
            plt.text(j, i, format(confusion_mat[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if confusion_mat[i, j] > thresh else "black")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()