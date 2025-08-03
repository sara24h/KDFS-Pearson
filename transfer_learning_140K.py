from data.dataset import Dataset_selector  # Replace with actual import
from model.pruned_model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal  # Replace with actual import
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import os



class StudentModelTester:
    """
    Simple tester for evaluating generalization of pruned student model
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        masks: List[torch.Tensor],
        device: str = 'cuda',
        num_classes: int = 1
    ):
        self.checkpoint_path = checkpoint_path
        self.masks = masks
        self.device = device
        self.num_classes = num_classes
        
        # Load the student model
        self.model = self._load_student_model()
        
    def _load_student_model(self) -> nn.Module:
        """Load the pruned student model from checkpoint"""
        model = ResNet_50_pruned_hardfakevsreal(self.masks)
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model.to(self.device)
        model.eval()  # Set to evaluation mode
        
        print(f"✅ Student model loaded from: {self.checkpoint_path}")
        
        # Print model stats
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"📊 Total parameters: {total_params:,}")
        print(f"📊 Trainable parameters: {trainable_params:,}")
        
        return model
    
    def test_single_dataset(self, dataloader, dataset_name: str = "Test") -> Dict:
        """Test model on a single dataset"""
        print(f"\n🧪 Testing on {dataset_name} dataset...")
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        total_loss = 0.0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output, features = self.model(data)
                
                # Calculate loss
                if self.num_classes == 1:
                    loss = F.binary_cross_entropy_with_logits(output.squeeze(), target)
                    probabilities = torch.sigmoid(output.squeeze())
                    predictions = (probabilities > 0.5).float()
                else:
                    loss = F.cross_entropy(output, target.long())
                    probabilities = F.softmax(output, dim=1)
                    predictions = output.argmax(dim=1).float()
                
                total_loss += loss.item()
                
                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                if batch_idx % 100 == 0:
                    print(f"  Processed {batch_idx * len(data)}/{len(dataloader.dataset)} samples")
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average='binary' if self.num_classes == 1 else 'macro'
        )
        
        results = {
            'dataset': dataset_name,
            'accuracy': accuracy * 100,
            'precision': precision * 100,
            'recall': recall * 100,
            'f1_score': f1 * 100,
            'avg_loss': total_loss / len(dataloader),
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': all_probabilities
        }
        
        # Calculate AUC if binary classification
        if self.num_classes == 1:
            try:
                auc = roc_auc_score(all_targets, all_probabilities)
                results['auc'] = auc * 100
            except:
                results['auc'] = 0.0
        
        # Print results
        print(f"📈 {dataset_name} Results:")
        print(f"   Accuracy: {results['accuracy']:.2f}%")
        print(f"   Precision: {results['precision']:.2f}%")
        print(f"   Recall: {results['recall']:.2f}%")
        print(f"   F1-Score: {results['f1_score']:.2f}%")
        if 'auc' in results:
            print(f"   AUC: {results['auc']:.2f}%")
        print(f"   Avg Loss: {results['avg_loss']:.4f}")
        
        return results
    
    def test_cross_dataset_generalization(self, datasets_dict: Dict[str, any]) -> Dict:
        """
        Test generalization across multiple datasets
        datasets_dict: {'dataset_name': dataloader}
        """
        print("🔄 Testing Cross-Dataset Generalization...")
        
        all_results = {}
        
        for dataset_name, dataloader in datasets_dict.items():
            results = self.test_single_dataset(dataloader, dataset_name)
            all_results[dataset_name] = results
        
        # Calculate generalization metrics
        accuracies = [results['accuracy'] for results in all_results.values()]
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        
        print(f"\n📊 Cross-Dataset Generalization Summary:")
        print(f"   Mean Accuracy: {mean_acc:.2f}%")
        print(f"   Std Accuracy: {std_acc:.2f}%")
        print(f"   Generalization Gap: {std_acc:.2f}%")
        
        # Rank datasets by difficulty
        sorted_results = sorted(all_results.items(), key=lambda x: x[1]['accuracy'])
        print(f"\n🏆 Dataset Ranking (by accuracy):")
        for i, (name, results) in enumerate(sorted_results):
            print(f"   {i+1}. {name}: {results['accuracy']:.2f}%")
        
        return all_results
    
    def plot_confusion_matrix(self, results: Dict, save_path: str = None):
        """Plot confusion matrix"""
        cm = confusion_matrix(results['targets'], results['predictions'])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
        plt.title(f'Confusion Matrix - {results["dataset"]}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_model_confidence(self, results: Dict):
        """Analyze model confidence distribution"""
        probabilities = np.array(results['probabilities'])
        targets = np.array(results['targets'])
        
        plt.figure(figsize=(12, 4))
        
        # Plot 1: Overall confidence distribution
        plt.subplot(1, 3, 1)
        plt.hist(probabilities, bins=50, alpha=0.7, color='blue')
        plt.title('Confidence Distribution')
        plt.xlabel('Probability')
        plt.ylabel('Count')
        
        # Plot 2: Confidence by class
        plt.subplot(1, 3, 2)
        real_probs = probabilities[targets == 1]
        fake_probs = probabilities[targets == 0]
        
        plt.hist(fake_probs, bins=30, alpha=0.7, label='Fake', color='red')
        plt.hist(real_probs, bins=30, alpha=0.7, label='Real', color='green')
        plt.title('Confidence by True Class')
        plt.xlabel('Probability')
        plt.ylabel('Count')
        plt.legend()
        
        # Plot 3: Calibration
        plt.subplot(1, 3, 3)
        bin_boundaries = np.linspace(0, 1, 11)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        accuracies = []
        confidences = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probabilities > bin_lower) & (probabilities <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = targets[in_bin].mean()
                avg_confidence_in_bin = probabilities[in_bin].mean()
                accuracies.append(accuracy_in_bin)
                confidences.append(avg_confidence_in_bin)
        
        plt.plot([0, 1], [0, 1], '--', color='gray', label='Perfect Calibration')
        plt.plot(confidences, accuracies, 'o-', color='blue', label='Model')
        plt.title('Calibration Plot')
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Print confidence statistics
        print(f"\n📈 Confidence Analysis for {results['dataset']}:")
        print(f"   Mean Confidence: {np.mean(probabilities):.3f}")
        print(f"   Confidence Std: {np.std(probabilities):.3f}")
        print(f"   Min Confidence: {np.min(probabilities):.3f}")
        print(f"   Max Confidence: {np.max(probabilities):.3f}")


def main():
    """
    Example usage - Simple testing
    """
    
    # Configuration
    CHECKPOINT_PATH = "/kaggle/input/kdfs-4-mordad-140k-new-pearson-final-part1/results/run_resnet50_imagenet_prune1/student_model/ResNet_50_sparse_last.pt"
    
    # You need to provide the masks used during pruning
    # Example: Load masks if you have them saved
    # masks = torch.load("path_to_masks.pt")
    masks = []  # Replace with your actual masks
    
    # Initialize tester
    tester = StudentModelTester(
        checkpoint_path=CHECKPOINT_PATH,
        masks=masks,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Setup test datasets
    print("📁 Loading test datasets...")
    
    # Test on 140k dataset
    dataset_140k = Dataset_selector(
        dataset_mode='140k',
        realfake140k_train_csv='/path/to/train.csv',  # Update paths
        realfake140k_valid_csv='/path/to/valid.csv',
        realfake140k_test_csv='/path/to/test.csv',
        realfake140k_root_dir='/path/to/dataset',
        train_batch_size=64,
        eval_batch_size=64,
        ddp=False
    )
    
    # Test on other datasets for generalization
    # dataset_190k = Dataset_selector(dataset_mode='190k', ...)
    # dataset_200k = Dataset_selector(dataset_mode='200k', ...)
    
    # Single dataset test
    results_140k = tester.test_single_dataset(
        dataset_140k.loader_test, 
        "Real-Fake 140k"
    )
    
    # Plot results
    tester.plot_confusion_matrix(results_140k)
    tester.analyze_model_confidence(results_140k)
    

    
    print("\n🎉 Testing completed!")
    print(f"Final Test Accuracy: {results_140k['accuracy']:.2f}%")


if __name__ == "__main__":
    main()
