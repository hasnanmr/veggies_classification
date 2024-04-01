import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

vgg16 = models.vgg16(pretrained=True)

# Freeze the convolutional base to prevent updating weights during training
for param in vgg16.features.parameters():
    param.requires_grad = False

num_features = vgg16.classifier[6].in_features
num_classes = 3  
vgg16.classifier[6] = torch.nn.Linear(num_features, num_classes)

# Load the model
model = vgg16
state_dict = torch.load('vgg16_transfer_learning.pth')
model.load_state_dict(state_dict)
model.eval()

# Define the same transforms that were used during the model training
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),  
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))        
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
st.write('you can upload your image of veggies below')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image')
    label, confidence = predict(image)
    st.write(f'Predicted label: {label}, confidence: {confidence:.2f}%')