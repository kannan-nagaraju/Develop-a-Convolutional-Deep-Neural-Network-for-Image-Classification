# Develop a Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

##   PROBLEM STATEMENT AND DATASET
The objective of this experiment is to build a Convolutional Neural Network (CNN) model that can classify grayscale fashion images into different clothing categories. The model is trained to identify clothing items such as T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, and Ankle boot.

### Dataset
The **Fashion-MNIST** dataset is used for this experiment. It is a benchmark dataset consisting of grayscale images of fashion products.

**Dataset Details:**
- Training Images: 60,000
- Testing Images: 10,000
- Image Size: 28 × 28 pixels
- Number of Classes: 10
- Input Shape: `(1, 28, 28)`

### Class Labels
1. T-shirt/top  
2. Trouser  
3. Pullover  
4. Dress  
5. Coat  
6. Sandal  
7. Shirt  
8. Sneaker  
9. Bag  
10. Ankle boot
    
## Neural Network Model

<img width="998" height="698" alt="image" src="https://github.com/user-attachments/assets/040871e4-5171-4f91-bf22-97657c8c1d37" />

## DESIGN STEPS
### STEP 1: 

Load the dataset from the tensorflow library.

### STEP 2: 

Preprocess the dataset.


### STEP 3: 

Include the training loss, validation loss vs iteration plot.

### STEP 4: 

Test the model for your handwritten scanned images.

### STEP 5: 
Predict and display the class of a single test image.








## PROGRAM

### Name: KANNAN N

### Register Number: 212223230097



```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
## Step 1: Load and Preprocess Data
# Define transformations for images
transform = transforms.Compose([
transforms.ToTensor(), # Convert images to tensors
transforms.Normalize((0.5,), (0.5,)) # Normalize images
])
# Load Fashion-MNIST dataset
train_dataset = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.FashionMNIST(root="./data", train=False, transform=transform, download=True)

# Get the shape of the first image in the training dataset
image, label = train_dataset[0]
print(image.shape)
print(len(train_dataset))

# Get the shape of the first image in the test dataset
image, label = test_dataset[0]
print(image.shape)
print(len(test_dataset))

# Create DataLoader for batch processing
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class CNNClassifier(nn.Module):
def __init__(self):
super(CNNClassifier, self).__init__()
self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
self.fc1 = nn.Linear(128 * 3 * 3, 128)
self.fc2 = nn.Linear(128, 64)
self.fc3 = nn.Linear(64, 10)

def forward(self, x):
# write your code here
x = self.pool(torch.relu(self.conv1(x)))
x = self.pool(torch.relu(self.conv2(x)))
x = self.pool(torch.relu(self.conv3(x)))
x = x.view(x.size(0),-1)
x = torch.relu(self.fc1(x))
x = torch.relu(self.fc2(x))
x = self.fc3(x)
return x

from torchsummary import summary

# Initialize model
model = CNNClassifier()

# Move model to GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda")
    model.to(device)

# Print model summary
print('Name: KAMALESH R')
print('Register Number: 212223230094')
summary(model, input_size=(1, 28, 28))

# Initialize model, loss function, and optimizer
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

## Step 3: Train the Model
def train_model(model, train_loader, num_epochs=3):
  print('Name: KANNAN N')
  print('Register Number: 212223230097')
  for epoch in range(num_epochs):
      model.train()
      running_loss = 0.0
      for images, labels in train_loader:
          optimizer.zero_grad()
          outputs = model(images)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
          running_loss += loss.item()
      print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

# Train the model
train_model(model, train_loader)

## Step 4: Test the Model
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    print('Name: KANNAN N')
    print('Register Number: 212223230097')
    print(f'Test Accuracy: {accuracy:.4f}')

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    print('Name: KANNAN N')
    print('Register Number: 212223230097')
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Print classification report
    print('Name: KANNAN N')
    print('Register Number: 212223230097')
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))

# Evaluate the model
test_model(model, test_loader)

## Step 5: Predict on a Single Image
import matplotlib.pyplot as plt
def predict_image(model, image_index, dataset):
    model.eval()
    image, label = dataset[image_index]
    with torch.no_grad():
        output = model(image.unsqueeze(0))  # Add batch dimension
        _, predicted = torch.max(output, 1)
    class_names = dataset.classes

    # Display the image
    print('Name: KANNAN N')
    print('Register Number: 212223230097')
    plt.imshow(image.squeeze(), cmap="gray")
    plt.title(f'Actual: {class_names[label]}\nPredicted: {class_names[predicted.item()]}')
    plt.axis("off")
    plt.show()
    print(f'Actual: {class_names[label]}, Predicted: {class_names[predicted.item()]}')

# Example Prediction
predict_image(model, image_index=70, dataset=test_dataset)
```

### OUTPUT

## Training Loss per Epoch


<img width="385" height="117" alt="image" src="https://github.com/user-attachments/assets/40008821-5a4c-4047-ae27-79d90d54740b" />



## Confusion Matrix



<img width="761" height="703" alt="image" src="https://github.com/user-attachments/assets/86920eba-9c8a-464a-a819-149a731a9614" />


## Classification Report


<img width="483" height="350" alt="image" src="https://github.com/user-attachments/assets/c49b4fa9-74b2-4d1e-b33f-a42d2ae1d512" />


### New Sample Data Prediction


<img width="426" height="499" alt="image" src="https://github.com/user-attachments/assets/21b9de80-b46f-4eac-bf39-2afc9740cd24" />


## RESULT
Thus, the program to develop a convolutional deep neural network for Image classification and to verify the response for new images has been completed successfully.
