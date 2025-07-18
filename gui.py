import streamlit as st
import torch
import numpy as np

from config import Config
from model import EmotionClassifier
from data_loader import TextPreprocessor, load_glove_embeddings

def main():
    # Initialize Streamlit app
    st.title("Emotion Classification with LSTM + Attention")
    st.write("Enter text below to predict its emotion.")

    # Load config
    config = Config()
    
    # Load pretrained GloVe embeddings
    st.write("Loading GloVe embeddings (this might take a moment)...")
    glove_dict = load_glove_embeddings(config.GLOVE_PATH, config.EMBEDDING_DIM)
    
    # Load the trained model
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    model = EmotionClassifier(config).to(device)
    
    checkpoint = torch.load(config.MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set model to evaluation mode
    
    # Create a text preprocessor
    preprocessor = TextPreprocessor()

    # Input text from user
    user_input = st.text_area("Enter text here", height=100)

    # Predict button
    if st.button("Predict Emotion"):
        if not user_input.strip():
            st.warning("Please enter some text before predicting.")
            return

        # Preprocess the user input
        tokens = preprocessor.preprocess(user_input)
        
        # Convert tokens to GloVe embeddings
        embeddings = []
        for token in tokens[:config.MAX_SEQUENCE_LENGTH]:
            if token in glove_dict:
                embedding = glove_dict[token]
            else:
                embedding = np.zeros(config.EMBEDDING_DIM, dtype='float32')
            embeddings.append(embedding)

        # Pad if needed
        if len(embeddings) < config.MAX_SEQUENCE_LENGTH:
            pad_size = config.MAX_SEQUENCE_LENGTH - len(embeddings)
            embeddings.extend([np.zeros(config.EMBEDDING_DIM, dtype='float32')] * pad_size)

        # Convert to a batch of size 1
        embeddings_tensor = torch.FloatTensor(embeddings).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            output = model(embeddings_tensor)
            pred_label = torch.argmax(output, dim=1).item()

        # Map predicted label to emotion
        emotion = config.EMOTIONS[pred_label]

        st.success(f"**Predicted Emotion:** {emotion}")

if __name__ == "__main__":
    main()
