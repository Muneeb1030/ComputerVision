import streamlit as st
import numpy as np
import cv2
import pickle
from tensorflow.keras.applications import VGG16

with open('svm_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)

with open('pca_object.pkl', 'rb') as f:
    pca = pickle.load(f)
    
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

def preprocess_image(image):
    image = cv2.resize(image, (32, 32))
    image = image.astype('float32') / 255.0
    return image

def extract_cnn_features(image):
    features = base_model.predict(np.expand_dims(image, axis=0))
    features = features.reshape((features.shape[0], -1))
    return features

st.title('CIFAR-10 Image Classification')
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    preprocessed_image = preprocess_image(image)
    cnn_features = extract_cnn_features(preprocessed_image)
    pca_features = pca.transform(cnn_features)
    prediction = svm_model.predict(pca_features)
    
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    st.write(f'Predicted Class: {classes[prediction[0]]}')
