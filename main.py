import torch
import logging
from datetime import datetime
from config import Config
from data_loader import load_and_split_data
from model import EmotionClassifier
from trainer import Trainer

def setup_logging():
    # Create logs directory if it doesn't exist
    import os
    os.makedirs("logs", exist_ok=True)
    
    # Setup logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )

def main():
    try:
        # Setup logging
        setup_logging()
        logger = logging.getLogger(__name__)
        
        # Log start time
        logger.info(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Initialize config
        config = Config()
        
        # Load and prepare data
        logger.info("Loading and preparing data...")
        train_loader, val_loader, test_loader = load_and_split_data(config)
        
        # Initialize model
        logger.info("Initializing model...")
        model = EmotionClassifier(config)
        
        # Initialize trainer
        trainer = Trainer(model, config)
        
        # Train model
        logger.info("Starting training...")
        history = trainer.train(train_loader, val_loader)
        
        # Evaluate on test set
        logger.info("\nEvaluating on test set...")
        
        # Load best model checkpoint
        checkpoint = torch.load(config.MODEL_SAVE_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        test_metrics = trainer.evaluate(test_loader)
        logger.info(f"Test Results - Loss: {test_metrics['loss']:.4f}, "
                   f"F1: {test_metrics['f1']:.4f}, "
                   f"Accuracy: {test_metrics['accuracy']:.4f}")
        
        # ---------------------------
        # CONFUSION MATRIX ON TRAIN DATA
        # ---------------------------
        logger.info("Generating confusion matrix on TRAIN data...")
        train_preds, train_labels = trainer.get_predictions(train_loader)
        trainer.plot_confusion_matrix(
            y_true=train_labels,
            y_pred=train_preds,
            emotion_labels=list(config.EMOTIONS.values()),
            title="Train Confusion Matrix"
        )
        
        # ---------------------------
        # CONFUSION MATRIX ON TEST DATA
        # ---------------------------
        logger.info("Generating confusion matrix on TEST data...")
        test_preds, test_labels = trainer.get_predictions(test_loader)
        trainer.plot_confusion_matrix(
            y_true=test_labels,
            y_pred=test_preds,
            emotion_labels=list(config.EMOTIONS.values()),
            title="Test Confusion Matrix"
        )
        
        # Save final results
        results = {
            'test_metrics': test_metrics,
            'training_history': history,
            'best_val_f1': checkpoint['best_val_f1'],
            'final_epoch': checkpoint['epoch']
        }
        torch.save(results, 'model_results.pt')
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
