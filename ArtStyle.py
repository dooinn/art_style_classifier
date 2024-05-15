import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.nn as nn
import timm
import pandas as pd


class SimpleArtClassifer(nn.Module):
    def __init__(self, num_classes=12):
        super(SimpleArtClassifer, self).__init__()
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        enet_out_size = 1280
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(enet_out_size, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        output = self.classifier(x)
        return output

@st.cache_data
def load_model():
    model = SimpleArtClassifer(num_classes=12)
    model.load_state_dict(torch.load('artifacts/model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Load art style data
@st.cache_data
def load_style_data():
    return pd.read_csv('data/art_style.csv')

style_data = load_style_data()

def predict(image):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    image = Image.open(image).convert('RGB')
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_idx = np.argmax(probabilities.numpy().flatten())
        return predicted_idx



uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.sidebar.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("The Predicted result is...")
    predicted_idx = predict(uploaded_file)
    class_names = ['Symbolism', 'Surrealism', 'Supermatism', 'Romanticism', 'Renaissance', 'Primitivism', 'Post-Impressionism', 'Pop Art', 'Impressionism', 'Expressionism', 'Cubism', 'Baroque']
    predicted_class = class_names[predicted_idx]
    st.title(f"{predicted_class}")

    style_info = style_data[style_data['style'] == predicted_class].iloc[0]
    st.write(f"{style_info['description']}")
    st.write(f"More Info: [Wikipedia]({style_info['wiki_url']})")
else:
    st.title('ArtStyle Predictor')
    st.subheader('Upload an image of a painting into the image uploader in the sidebar to discover its artistic style!')
    st.write("ArtStyle Predictor is your personal art historian. If you're unsure about the era or the artistic style of a painting, the ArtStyle Predictor will identify it for you, providing detailed explanations about the art style!")
    st.image('assets/landing.png')