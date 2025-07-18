from dataclasses import dataclass

@dataclass
class Config:
    MAX_SEQUENCE_LENGTH = 64
    BATCH_SIZE = 64
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    EMBEDDING_DIM = 300  # Match the dimension of your GloVe file (e.g., glove.6B.300d.txt)
    HIDDEN_DIM = 512
    NUM_CLASSES = 6
    NUM_LAYERS = 3
    DROPOUT = 0.5
    
    LEARNING_RATE = 3e-4
    NUM_EPOCHS = 40
    EARLY_STOPPING_PATIENCE = 5
    GRADIENT_CLIP = 1.0
    
    # Path to your GloVe embeddings. Example: "glove.6B.300d.txt"
    GLOVE_PATH = "glove.6B/glove.6B.300d.txt"
    
    MODEL_SAVE_PATH = "emotion_model.pt"
    DEVICE = "cuda"  # or "cpu"
    
    EMOTIONS = {
        0: "sadness",
        1: "joy",
        2: "love",
        3: "anger",
        4: "fear",
        5: "surprise"
    }
