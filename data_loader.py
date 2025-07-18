import spacy
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from typing import List, Tuple, Dict
from config import Config

class TextPreprocessor:
    def __init__(self):
        # Ensure you have 'en_core_web_sm' installed: python -m spacy download en_core_web_sm
        self.nlp = spacy.load('en_core_web_sm')
        
    def preprocess(self, text: str) -> List[str]:
        """
        Basic preprocessing: lowercasing, lemmatization,
        filtering out non-alpha tokens except certain punctuation.
        """
        doc = self.nlp(text.lower())
        tokens = []
        for token in doc:
            if token.is_alpha or token.text in ["!", "?", "..."]:
                tokens.append(token.lemma_)
        return tokens

def load_glove_embeddings(glove_path: str, embedding_dim: int) -> Dict[str, np.ndarray]:
    """
    Loads GloVe embeddings from a .txt file into a dictionary.
    """
    glove_dict = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            if len(vector) == embedding_dim:
                glove_dict[word] = vector
    return glove_dict

class EmotionDataset(Dataset):
    def __init__(
        self, 
        texts: List[str], 
        labels: List[int], 
        glove_dict: Dict[str, np.ndarray], 
        config: Config
    ):
        self.texts = texts
        self.labels = labels
        self.glove_dict = glove_dict
        self.config = config
        self.preprocessor = TextPreprocessor()
        
        # For unknown tokens, we'll use a zero vector.
        self.unk_vector = np.zeros(self.config.EMBEDDING_DIM, dtype='float32')
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        tokens = self.preprocessor.preprocess(text)
        
        embeddings = []
        for token in tokens[:self.config.MAX_SEQUENCE_LENGTH]:
            if token in self.glove_dict:
                embedding = self.glove_dict[token]
            else:
                embedding = self.unk_vector
            embeddings.append(embedding)
        
        # Pad to MAX_SEQUENCE_LENGTH if needed
        if len(embeddings) < self.config.MAX_SEQUENCE_LENGTH:
            pad_size = self.config.MAX_SEQUENCE_LENGTH - len(embeddings)
            padding = [np.zeros(self.config.EMBEDDING_DIM, dtype='float32')] * pad_size
            embeddings.extend(padding)
        
        embeddings_tensor = torch.FloatTensor(embeddings)
        
        return {
            'embeddings': embeddings_tensor,
            'label': torch.LongTensor([label])[0]
        }

def load_and_split_data(config: Config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Loads the 'dair-ai/emotion' dataset, splits it, 
    and returns DataLoaders for train/val/test.
    """
    dataset = load_dataset("dair-ai/emotion")
    
    # Extract texts and labels
    texts = dataset['train']['text']
    labels = dataset['train']['label']
    
    # Train/validation/test split
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, 
        train_size=config.TRAIN_RATIO, 
        stratify=labels
    )
    
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels,
        train_size=0.5,
        stratify=temp_labels
    )
    
    # Load pretrained GloVe embeddings
    print(f"Loading GloVe embeddings from: {config.GLOVE_PATH}")
    glove_dict = load_glove_embeddings(config.GLOVE_PATH, config.EMBEDDING_DIM)
    
    # Create PyTorch datasets
    train_dataset = EmotionDataset(train_texts, train_labels, glove_dict, config)
    val_dataset = EmotionDataset(val_texts, val_labels, glove_dict, config)
    test_dataset = EmotionDataset(test_texts, test_labels, glove_dict, config)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE)
    
    return train_loader, val_loader, test_loader
