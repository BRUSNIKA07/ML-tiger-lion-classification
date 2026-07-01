"""
Streamlit web application for Tiger vs Lion classification.
Users can upload an image and get instant classification results.
"""
import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
import numpy as np
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Tiger vs Lion Classifier",
    page_icon="🐯",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        max-width: 700px;
        margin: 0 auto;
    }
    .title {
        text-align: center;
        color: #FF6B35;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 30px;
    }
    .result-container {
        text-align: center;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .tiger {
        background-color: #fff3e0;
        border: 3px solid #FF6B35;
    }
    .lion {
        background-color: #fce4ec;
        border: 3px solid #E91E63;
    }
    .confidence {
        font-size: 1.2em;
        margin-top: 10px;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">🐯 Tiger vs Lion Classifier 🦁</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload an image to identify whether it\'s a tiger or a lion</div>', unsafe_allow_html=True)

@st.cache_resource
def load_model(model_path="./artifacts/best_model.pt"):
    """Load the trained model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model architecture
    model = resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    
    # Load weights if they exist
    if Path(model_path).exists():
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        st.success("✅ Model loaded successfully!")
    else:
        st.warning(f"⚠️ Model weights not found at {model_path}. Using untrained model.")
    
    model.to(device)
    model.eval()
    return model, device

def preprocess_image(image):
    """Preprocess the uploaded image."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image).unsqueeze(0)

def predict(model, image_tensor, device):
    """Make prediction on the image."""
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()
    
    return prediction, confidence

# Load model
model, device = load_model()

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png", "gif", "bmp"],
    help="Upload a JPG, PNG or GIF image"
)

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, use_column_width=True, caption="Uploaded Image")
    
    # Process and predict
    with st.spinner("🔍 Analyzing image..."):
        image_tensor = preprocess_image(image)
        prediction, confidence = predict(model, image_tensor, device)
    
    # Display results
    class_names = {0: "Tiger", 1: "Lion"}
    predicted_class = class_names[prediction]
    emoji = "🐯" if predicted_class == "Tiger" else "🦁"
    
    result_class = "tiger" if predicted_class == "Tiger" else "lion"
    
    st.markdown(f"""
    <div class="result-container {result_class}">
        <div style="font-size: 3em; margin-bottom: 10px;">{emoji}</div>
        <div style="font-size: 2.5em; font-weight: bold; color: {'#FF6B35' if predicted_class == 'Tiger' else '#E91E63'};">
            {predicted_class}
        </div>
        <div class="confidence">
            Confidence: <strong>{confidence * 100:.2f}%</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Display probability bar
    col1, col2 = st.columns(2)
    with col1:
        tiger_prob = 1 - confidence if predicted_class == "Lion" else confidence
        st.metric("🐯 Tiger", f"{tiger_prob * 100:.2f}%")
    with col2:
        lion_prob = confidence if predicted_class == "Lion" else 1 - confidence
        st.metric("🦁 Lion", f"{lion_prob * 100:.2f}%")

else:
    # Show placeholder
    st.info("👆 Upload an image to get started!")
    
    with st.expander("ℹ️ About this app"):
        st.write("""
        This application uses a **ResNet18** neural network trained on tiger and lion images
        to classify whether an uploaded image contains a tiger or a lion.
        
        **Features:**
        - Fast and accurate classification
        - Shows confidence percentage
        - Works with common image formats (JPG, PNG, GIF, BMP)
        - GPU support for faster predictions
        
        **How to use:**
        1. Upload an image containing a tiger or lion
        2. The AI will analyze it and show the result
        3. Check the confidence score to see how sure the model is
        """)

# Footer
st.markdown("""
---
<div style="text-align: center; color: #999; font-size: 0.9em;">
    <p>Tiger vs Lion Classification • Powered by PyTorch & Streamlit</p>
</div>
""", unsafe_allow_html=True)
