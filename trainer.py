import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from config import Config
from model import EmotionClassifier

class Trainer:
    def __init__(self, model: nn.Module, config: Config):
        self.model = model
        self.config = config
        self.device = torch.device(config.DEVICE)
        self.model.to(self.device)
        
        # Simple uniform class weights (could be improved if dataset is imbalanced)
        class_weights = torch.ones(config.NUM_CLASSES, device=self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=0.01
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=2,
            verbose=True
        )
        
        self.best_val_f1 = 0
        self.patience_counter = 0
        
        self.plots_dir = "training_plots"
        os.makedirs(self.plots_dir, exist_ok=True)
        
    def plot_training_history(self, history: List[Dict]):
        train_loss = [epoch['train']['loss'] for epoch in history]
        val_loss = [epoch['val']['loss'] for epoch in history]
        train_acc = [epoch['train']['accuracy'] for epoch in history]
        val_acc = [epoch['val']['accuracy'] for epoch in history]
        epochs = range(1, len(history) + 1)
        
        plt.figure(figsize=(15, 6))
        
        # Plot Loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_loss, 'b-', label='Training Loss')
        plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_acc, 'b-', label='Training Accuracy')
        plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(self.plots_dir, f'training_history_{timestamp}.png')
        
        plt.savefig(plot_path)
        plt.close()
        
        print(f"\nTraining plots saved to: {plot_path}")

    def train_epoch(self, train_loader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch in tqdm(train_loader, desc="Training"):
            embeddings = batch['embeddings'].to(self.device)
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(embeddings)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRADIENT_CLIP)
            self.optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = total_loss / len(train_loader)
        epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        return {
            'loss': epoch_loss,
            'f1': epoch_f1,
            'accuracy': epoch_acc
        }
    
    def evaluate(self, val_loader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                embeddings = batch['embeddings'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(embeddings)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        val_loss = total_loss / len(val_loader)
        val_f1 = f1_score(all_labels, all_preds, average='weighted')
        val_acc = accuracy_score(all_labels, all_preds)
        
        return {
            'loss': val_loss,
            'f1': val_f1,
            'accuracy': val_acc
        }
    
    def train(self, train_loader, val_loader) -> List[Dict[str, float]]:
        history = []
        
        for epoch in range(self.config.NUM_EPOCHS):
            print(f"\nEpoch {epoch+1}/{self.config.NUM_EPOCHS}")
            
            train_metrics = self.train_epoch(train_loader)
            print(f"Training - Loss: {train_metrics['loss']:.4f}, F1: {train_metrics['f1']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            
            val_metrics = self.evaluate(val_loader)
            print(f"Validation - Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['f1']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
            
            self.scheduler.step(val_metrics['f1'])
            
            history.append({
                'train': train_metrics,
                'val': val_metrics
            })
            
            # Check for best validation F1
            if val_metrics['f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1']
                self.patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_val_f1': self.best_val_f1,
                }, self.config.MODEL_SAVE_PATH)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                    print("Early stopping triggered!")
                    break
        
        self.plot_training_history(history)
        return history

    def get_predictions(self, data_loader):
        """
        Return all predictions and true labels for the given DataLoader.
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                embeddings = batch['embeddings'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(embeddings)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        return all_preds, all_labels

    def plot_confusion_matrix(self, y_true, y_pred, emotion_labels, title="Confusion Matrix"):
        """
        Plot and show a confusion matrix given true/pred labels and class names.
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues', 
            xticklabels=emotion_labels, 
            yticklabels=emotion_labels
        )
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        plt.show()
        plt.close()
