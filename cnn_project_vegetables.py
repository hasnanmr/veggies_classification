import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 14 * 14, 80)
        self.dropout1 = nn.Dropout(p=0.2) 
        self.fc2 = nn.Linear(80, 80)
        self.fc3 = nn.Linear(80, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load the model
model = Net()
state_dict = torch.load('cnn64_dropout1_2.pth')
model.load_state_dict(state_dict)
model.eval()

# Define the same transforms that were used during the model training
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize to the input size of the model
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

classes = ('broccoli', 'cabbage', 'cauliflower')

def predict(image):
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    max_value, predicted_class = torch.max(probabilities, 0)
    return classes[predicted_class.item()], max_value.item() * 100

st.title('Vegetable Classification')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image')
    label, confidence = predict(image)
    st.write(f'Predicted label: {label}, confidence: {confidence:.2f}%')