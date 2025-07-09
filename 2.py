import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
import io
import base64

# Set page config
st.set_page_config(
    page_title="Satellite Image Classifier",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 12px;
        border-radius: 5px;
        font-size: 16px;
        margin-top: 10px;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .prediction-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border-left: 5px solid #4CAF50;
    }
    .confidence-bar {
        background-color: #e1e5e9;
        border-radius: 5px;
        overflow: hidden;
        margin: 5px 0;
    }
    .confidence-fill {
        height: 20px;
        background-color: #4CAF50;
        text-align: center;
        line-height: 20px;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Cache the model loading
@st.cache_resource
def load_classification_model():
    try:
        model = load_model('Modelenv.v1.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Define class names
class_names = ['Cloudy', 'Desert', 'Green_Area', 'Water']
class_emojis = {'Cloudy': '☁️', 'Desert': '🏜️', 'Green_Area': '🌿', 'Water': '💧'}

# Function to preprocess image
def preprocess_image(image, target_size=(255, 255)):
    """Preprocess image for prediction"""
    image = image.resize(target_size)
    image = img_to_array(image)
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)
    return image

# Function to predict image class
def predict_image_class(model, image):
    """Predict the class of the image"""
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_index]
    confidence = predictions[0][predicted_class_index]
    
    return predicted_class, confidence, predictions[0]

# Function to create confidence chart
def create_confidence_chart(predictions, class_names):
    """Create a confidence chart using plotly"""
    fig = px.bar(
        x=class_names,
        y=predictions,
        title='Prediction Confidence for Each Class',
        labels={'x': 'Land Cover Type', 'y': 'Confidence Score'},
        color=predictions,
        color_continuous_scale='viridis'
    )
    fig.update_layout(
        height=400,
        showlegend=False,
        xaxis_title="Land Cover Type",
        yaxis_title="Confidence Score"
    )
    return fig

# Main app
def main():
    # Header
    st.title("🛰️ Satellite Image Land Cover Classifier")
    st.markdown("""
    This application classifies satellite images into different land cover types using a trained CNN model.
    Upload a satellite image to get real-time predictions!
    """)
    
    # Sidebar
    st.sidebar.header("🎛️ Controls")
    
    # Model status
    with st.sidebar:
        st.subheader("Model Status")
        model = load_classification_model()
        if model:
            st.success("✅ Model loaded successfully!")
        else:
            st.error("❌ Model not loaded. Please check if 'Modelenv.v1.h5' exists.")
            return
    
    # File uploader
    st.sidebar.subheader("Upload Image")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a satellite image...",
        type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
        help="Upload a satellite image for classification"
    )
    
    # Main content
    if uploaded_file is not None:
        # Load and display image
        image = Image.open(uploaded_file)
        
        # Create columns for layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📸 Original Image")
            st.image(image, use_column_width=True)
            
            # Image info
            st.subheader("🔍 Image Information")
            st.write(f"**Filename:** {uploaded_file.name}")
            st.write(f"**Format:** {image.format}")
            st.write(f"**Size:** {image.size}")
            st.write(f"**Mode:** {image.mode}")
        
        with col2:
            # Make prediction
            if st.button("🔮 Classify Image"):
                with st.spinner("Analyzing image..."):
                    try:
                        predicted_class, confidence, all_predictions = predict_image_class(model, image)
                        
                        # Display prediction
                        st.subheader("🎯 Prediction Results")
                        
                        # Main prediction box
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h3>Predicted Class: {class_emojis[predicted_class]} {predicted_class}</h3>
                            <p>Confidence: {confidence:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Confidence score visualization
                        st.subheader("📊 Confidence Scores")
                        
                        # Create confidence bars
                        for i, class_name in enumerate(class_names):
                            confidence_score = all_predictions[i]
                            st.write(f"**{class_emojis[class_name]} {class_name}**: {confidence_score:.2%}")
                            progress_bar = st.progress(float(confidence_score))
                        
                        # Detailed confidence chart
                        st.subheader("📈 Detailed Confidence Chart")
                        fig = create_confidence_chart(all_predictions, class_names)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Additional insights
                        st.subheader("🔍 Analysis Insights")
                        sorted_predictions = sorted(zip(class_names, all_predictions), 
                                                  key=lambda x: x[1], reverse=True)
                        
                        st.write("**Top 3 predictions:**")
                        for i, (class_name, score) in enumerate(sorted_predictions[:3]):
                            st.write(f"{i+1}. {class_emojis[class_name]} {class_name}: {score:.2%}")
                        
                        # Interpretation
                        if confidence > 0.7:
                            st.success("🎯 High confidence prediction! The model is quite certain about this classification.")
                        elif confidence > 0.5:
                            st.warning("⚠️ Moderate confidence. The model has some uncertainty.")
                        else:
                            st.error("❌ Low confidence. The model is uncertain about this classification.")
                        
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")
    
    else:
        # Landing page when no image is uploaded
        st.info("👆 Please upload a satellite image using the sidebar to get started.")
        
        # Add some sample information
        st.subheader("🎯 Supported Classes")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            ### ☁️ Cloudy
            Identifies areas with heavy cloud cover in satellite imagery.
            """)
        
        with col2:
            st.markdown("""
            ### 🏜️ Desert
            Detects desert and arid land regions with sparse vegetation.
            """)
        
        with col3:
            st.markdown("""
            ### 🌿 Green Area
            Identifies areas with vegetation, forests, and agricultural land.
            """)
        
        with col4:
            st.markdown("""
            ### 💧 Water
            Detects water bodies including rivers, lakes, and oceans.
            """)
    
    # Add some additional features in the sidebar
    with st.sidebar:
        st.subheader("📚 Model Information")
        st.write("""
        - **Model Type**: Convolutional Neural Network (CNN)
        - **Input Size**: 255x255 pixels
        - **Classes**: 4 (Cloudy, Desert, Green Area, Water)
        - **Framework**: TensorFlow/Keras
        """)
        
        st.subheader("🎯 How to Use")
        st.write("""
        1. Upload a satellite image (PNG, JPG, JPEG, TIFF, BMP)
        2. Click 'Classify Image'
        3. View prediction results and confidence scores
        4. Analyze the detailed confidence chart
        """)
        
        st.subheader("💡 Tips")
        st.write("""
        - Use clear, high-quality satellite images for best results
        - Images are automatically resized to 255x255 pixels
        - The model works best with images similar to its training data
        """)

if __name__ == "__main__":
    main()