#!/usr/bin/env python
import torch
from torchvision import transforms
from torch.utils.data import DataLoader 
from sinml import ImageFolderWithPaths, ImageClassifier
import csv
import os


# # Get the folder names that serve as the dictinary keys for data loading
# evaluateFolderNameList = []
# for DATA_DIR in ["Water-bilayer"]:
#     for CYCLEGANDATA in ["PPAFM2Exp_CoAll"]:
#         for LAMBDA1 in [0, 10, 20, 30, 40, 50, 60]:
#             for LAMBDA2 in [0, 0.1, 1, 10]:
#                 for EPOCH in ["latest"]:
#                     fakeFolderName = "fakeB_{}CG{}L{}L{}E{}".format(DATA_DIR, CYCLEGANDATA, LAMBDA1, LAMBDA2, EPOCH)
#                     evaluateFolderNameList.append(fakeFolderName)
# evaluateFolderNameList.append("realA")
# evaluateFolderNameList.append("realB")

#################
# Preparations 
#################
inputPath='../../data/preEvaluate/'
weightPath='../../processed_data/pre_evaluation/learned_model/'
outputPath='../../processed_data/fidelityTest/'

# Define the device
device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print('Using {} device.'.format(device))

# Load the test data 
transform = transforms.Compose([transforms.Grayscale(), transforms.Resize((192, 192)), transforms.ToTensor()])
test_data = ImageFolderWithPaths(inputPath, transform=transform)

class_names = test_data.classes
print('Class_name', class_names)
class_dict = test_data.class_to_idx
print(class_dict)
print('Testing size: {}'.format(len(test_data)))
test_dataloader = DataLoader(test_data, batch_size=8, shuffle=False)

# Initialize the trained model
model = ImageClassifier().to(device)
model.load_state_dict(torch.load(f'{weightPath}/best_memory.pth', map_location=device))
model.eval()

# Prepare a list to store all results
all_results = []

# Iterate through the test dataset
with torch.no_grad():
    for inputs, labels, paths in test_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        for path, label, output in zip(paths, labels, outputs):
            filename = os.path.basename(path)
            output_value = output.item()
            class_name = [key for key, value in class_dict.items() if value == label.item()][0]
            all_results.append([filename, output_value, class_name])

# Save all results to a single CSV file
with open(f'{outputPath}/fidelity_results.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Filename', 'Output Value', 'Class'])
    writer.writerows(all_results)
