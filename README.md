# Develop a Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional neural network (CNN) classification model for the given dataset.

## THEORY
The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), each of size 28×28 pixels. The task is to classify these images into their respective digit categories. CNNs are particularly well-suited for image classification tasks as they can automatically learn spatial hierarchies of features through convolutional layers, pooling layers, and fully connected layers.

## Neural Network Model
<img width="1204" height="419" alt="Screenshot 2026-08-03 092847" src="https://github.com/user-attachments/assets/b46d060d-6049-436f-b500-b23cf2efff45" />


## DESIGN STEPS
### STEP 1: 
Load and Preprocess Data.

### STEP 2: 
Get the shape of the first image in the training dataset.


### STEP 3: 

Get the shape of the first image in the test dataset.

### STEP 4: 
Train and test the Model.


### STEP 5: 
Predict on a Single Image.


### STEP 6: 

Display the image.



## PROGRAM


### Name : Titus Ratna Kumar Karivella 
### Register number : 212224230292

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Lambda, Normalize
from sklearn.metrics import confusion_matrix, classification_report

transform = transforms.Compose([ToTensor(),Normalize((0.5,),(0.5,))])
train_data = datasets.MNIST(root='data',train=True,download=True,transform=transform)
test_data = datasets.MNIST(root='data',train=False,download=True,transform=transform)

image, label = train_data[0] 
print(image.shape) 
print(len(train_data))
image, label = test_data[0] 
print(image.shape) 
print(len(test_data))

train_dataloader = DataLoader(train_data,batch_size=64,shuffle=True)
test_dataloader = DataLoader(test_data,batch_size=64,shuffle=False)

class MNISTClassifier(nn.Module):
  def __init__(self):
    super(MNISTClassifier, self).__init__()
    self.layer1 = nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1)
    self.layer2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
    self.layer3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
    self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
    self.fc1 = nn.Linear(128*3*3,128)
    self.fc2 = nn.Linear(128,64)
    self.fc3 = nn.Linear(64,10)
  def forward(self, x):
    x = self.pool(self.layer1(x))
    x = self.pool(self.layer2(x))
    x = self.pool(self.layer3(x))
    x = x.view(x.size(0),-1)
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = self.fc3(x)
    return x

model = MNISTClassifier()
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

def train_model(dataloader,model,loss,optimizer,epochs=3):
  train_loss = []
  model.train()
  for epoch in range(epochs):
    running_loss = 0.0
    for batch,(X,y) in enumerate(dataloader):
      y_pred = model(X)
      loss_value = loss(y_pred,y)
      optimizer.zero_grad()
      loss_value.backward()
      optimizer.step()
      running_loss += loss_value.item()
    train_loss.append(running_loss/len(dataloader))
    print(f"Epoch: {epoch+1} | Training Loss: {train_loss[-1]}")

train_model(train_dataloader,model,loss,optimizer)

def test_model(dataloader, model, loss):
  test_loss = 0.0
  correct_predictions = 0
  all_preds = []
  all_labels = []

  model.eval()
  with torch.inference_mode():
    for X, y in dataloader:
      y_pred = model(X)
      loss_value = loss(y_pred, y)
      test_loss += loss_value.item()

      _, predicted = torch.max(y_pred.data, 1)
      correct_predictions += (predicted == y).sum().item()

      all_preds.extend(predicted.cpu().numpy())
      all_labels.extend(y.cpu().numpy())

  avg_test_loss = test_loss / len(dataloader)
  accuracy = 100 * correct_predictions / len(dataloader.dataset)

  print(f"\nTest Results:")
  print(f"Average Loss: {avg_test_loss:.4f}")
  print(f"Accuracy: {accuracy:.2f}%")

  conf_matrix = confusion_matrix(all_labels, all_preds)
  class_report = classification_report(all_labels, all_preds, target_names=[str(i) for i in range(10)])

  print("\nConfusion Matrix:")
  sn.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
  plt.xlabel('Predicted')
  plt.ylabel('True')
  plt.show()
  print("\nClassification Report:")
  print(class_report)

test_model(test_dataloader,model,loss)

def predict_image(dataloader, model, ind):
  model.eval()
  image, label = dataloader[ind] 
  with torch.no_grad():
    y_pred = model(image.unsqueeze(0))
    _, pred = torch.max(y_pred,1)
  class_names = [str(i) for i in range(10)]
  plt.imshow(image.squeeze(), cmap="gray")
  plt.title(f'Actual: {class_names[label]}\nPredicted: {class_names[pred.item()]}')
  plt.axis("off")
  plt.show()
  print(f'Actual: {class_names[label]}, Predicted: {class_names[pred.item()]}')

predict_image(test_data, model, 47)
    

```

## OUTPUT

### Training Loss per Epoch

<img width="432" height="244" alt="Screenshot 2026-08-03 094043" src="https://github.com/user-attachments/assets/1c959927-56a0-4295-bf2e-b82eefe96d83" />


### Confusion Matrix

<img width="559" height="431" alt="Screenshot 2026-08-03 093941" src="https://github.com/user-attachments/assets/a1cf7642-e66e-4ebe-a57e-5a429cc5eb95" />


### Classification Report

<img width="457" height="282" alt="Screenshot 2026-08-03 094008" src="https://github.com/user-attachments/assets/04c8013c-81f9-4a3b-9e8f-9973d29e2e7a" />


### New Sample Data Prediction
<img width="459" height="430" alt="Screenshot 2026-08-03 094121" src="https://github.com/user-attachments/assets/faaa5a93-08c5-4571-addb-a6cd62cbed5b" />

