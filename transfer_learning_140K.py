import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import os
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Import your existing classes (assuming they're in the same directory)
from data.dataset import Dataset_selector, FaceDataset
from model.pruned_model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal

class GeneralizationEvaluator:
    def __init__(self, model_checkpoint_path, device='cuda'):
        self.device = device
        self.model_checkpoint_path = model_checkpoint_path
        self.model = None
        self.results = {}
        
    def load_student_model(self, masks):
        """Load the pruned student model"""
        try:
            # Initialize the pruned model architecture
            self.model = ResNet_50_pruned_hardfakevsreal(masks)
            
            # Load the checkpoint
            checkpoint = torch.load(self.model_checkpoint_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.to(self.device)
            self.model.eval()
            
            print(f"Student model loaded successfully from {self.model_checkpoint_path}")
            
            # Print model statistics
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            
        except Exception as e:
            print(f"Error loading student model: {e}")
            raise e
    
    def evaluate_on_dataset(self, data_loader, dataset_name):
        """Evaluate model on a specific dataset"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        total_loss = 0.0
        criterion = nn.BCEWithLogitsLoss()
        
        print(f"\nEvaluating on {dataset_name}...")
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(tqdm(data_loader, desc=f"Evaluating {dataset_name}")):
                images = images.to(self.device)
                labels = labels.to(self.device).float()
                
                # Forward pass
                outputs, _ = self.model(images)
                outputs = outputs.squeeze()
                
                # Calculate loss
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                # Get predictions and probabilities
                probabilities = torch.sigmoid(outputs)
                predictions = (probabilities > 0.5).float()
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='binary')
        
        try:
            auc = roc_auc_score(all_labels, all_probabilities)
        except:
            auc = 0.0
        
        avg_loss = total_loss / len(data_loader)
        
        # Store results
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'loss': avg_loss,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities
        }
        
        self.results[dataset_name] = results
        
        print(f"{dataset_name} Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  AUC: {auc:.4f}")
        print(f"  Loss: {avg_loss:.4f}")
        
        return results
    
    def comprehensive_evaluation(self, dataset_configs):
        """
        Perform comprehensive evaluation across multiple datasets
        
        dataset_configs: list of dictionaries with dataset configuration
        Example:
        dataset_configs = [
            {
                'name': '140k',
                'mode': '140k',
                'train_csv': '/path/to/train.csv',
                'valid_csv': '/path/to/valid.csv',
                'test_csv': '/path/to/test.csv',
                'root_dir': '/path/to/root'
            }
        ]
        """
        evaluation_results = {}
        
        for config in dataset_configs:
            try:
                print(f"\n{'='*50}")
                print(f"Processing {config['name']} dataset")
                print(f"{'='*50}")
                
                # Create dataset
                dataset = Dataset_selector(
                    dataset_mode=config['mode'],
                    **{k: v for k, v in config.items() 
                       if k not in ['name', 'mode']},
                    train_batch_size=32,
                    eval_batch_size=32,
                    num_workers=4,
                    ddp=False
                )
                
                # Evaluate on test set
                test_results = self.evaluate_on_dataset(
                    dataset.loader_test, 
                    f"{config['name']}_test"
                )
                
                # Evaluate on validation set
                val_results = self.evaluate_on_dataset(
                    dataset.loader_val, 
                    f"{config['name']}_val"
                )
                
                evaluation_results[config['name']] = {
                    'test': test_results,
                    'validation': val_results
                }
                
            except Exception as e:
                print(f"Error processing {config['name']}: {e}")
                continue
        
        return evaluation_results
    
    def plot_confusion_matrices(self, save_path=None):
        """Plot confusion matrices for all evaluated datasets"""
        n_datasets = len(self.results)
        if n_datasets == 0:
            print("No results to plot")
            return
        
        # Calculate grid dimensions
        cols = min(3, n_datasets)
        rows = (n_datasets + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_datasets == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, (dataset_name, results) in enumerate(self.results.items()):
            cm = confusion_matrix(results['labels'], results['predictions'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Fake', 'Real'], 
                       yticklabels=['Fake', 'Real'],
                       ax=axes[idx])
            axes[idx].set_title(f'{dataset_name}\nAccuracy: {results["accuracy"]:.3f}')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        # Hide unused subplots
        for idx in range(n_datasets, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrices saved to {save_path}")
        
        plt.show()
    
    def plot_performance_comparison(self, save_path=None):
        """Plot performance comparison across datasets"""
        if not self.results:
            print("No results to plot")
            return
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
        dataset_names = list(self.results.keys())
        
        # Prepare data for plotting
        metric_data = {metric: [] for metric in metrics}
        
        for dataset_name in dataset_names:
            for metric in metrics:
                metric_data[metric].append(self.results[dataset_name][metric])
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(dataset_names))
        width = 0.15
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, metric in enumerate(metrics):
            ax.bar(x + i * width, metric_data[metric], 
                   width, label=metric.replace('_', ' ').title(), 
                   color=colors[i], alpha=0.8)
        
        ax.set_xlabel('Datasets')
        ax.set_ylabel('Score')
        ax.set_title('Student Model Performance Across Different Datasets')
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(dataset_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
        
        # Add value labels on bars
        for i, metric in enumerate(metrics):
            for j, value in enumerate(metric_data[metric]):
                ax.text(j + i * width, value + 0.01, f'{value:.3f}', 
                       ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Performance comparison saved to {save_path}")
        
        plt.show()
    
    def generate_report(self, save_path=None):
        """Generate a comprehensive evaluation report"""
        if not self.results:
            print("No results to generate report")
            return
        
        report = []
        report.append("="*60)
        report.append("STUDENT MODEL GENERALIZATION EVALUATION REPORT")
        report.append("="*60)
        report.append(f"Model Checkpoint: {self.model_checkpoint_path}")
        report.append(f"Device: {self.device}")
        report.append("")
        
        # Summary table
        report.append("PERFORMANCE SUMMARY")
        report.append("-" * 40)
        report.append(f"{'Dataset':<20} {'Accuracy':<10} {'F1-Score':<10} {'AUC':<10}")
        report.append("-" * 40)
        
        for dataset_name, results in self.results.items():
            report.append(f"{dataset_name:<20} {results['accuracy']:<10.4f} "
                         f"{results['f1_score']:<10.4f} {results['auc']:<10.4f}")
        
        report.append("")
        
        # Detailed results
        report.append("DETAILED RESULTS")
        report.append("-" * 40)
        
        for dataset_name, results in self.results.items():
            report.append(f"\n{dataset_name.upper()}:")
            report.append(f"  Accuracy:  {results['accuracy']:.4f}")
            report.append(f"  Precision: {results['precision']:.4f}")
            report.append(f"  Recall:    {results['recall']:.4f}")
            report.append(f"  F1-Score:  {results['f1_score']:.4f}")
            report.append(f"  AUC:       {results['auc']:.4f}")
            report.append(f"  Loss:      {results['loss']:.4f}")
        
        # Generalization analysis
        report.append("")
        report.append("GENERALIZATION ANALYSIS")
        report.append("-" * 40)
        
        accuracies = [results['accuracy'] for results in self.results.values()]
        f1_scores = [results['f1_score'] for results in self.results.values()]
        
        report.append(f"Mean Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        report.append(f"Mean F1-Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
        report.append(f"Best Dataset:  {max(self.results.keys(), key=lambda k: self.results[k]['accuracy'])}")
        report.append(f"Worst Dataset: {min(self.results.keys(), key=lambda k: self.results[k]['accuracy'])}")
        
        report_text = "\n".join(report)
        print(report_text)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"\nReport saved to {save_path}")
        
        return report_text


# Usage example
def main():
    # Initialize evaluator
    checkpoint_path = "/kaggle/input/kdfs-4-mordad-140k-new-pearson-final-part1/results/run_resnet50_imagenet_prune1/student_model/ResNet_50_sparse_last.pt"
    evaluator = GeneralizationEvaluator(checkpoint_path, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # You need to provide the masks used during pruning
    # This should be the same masks used when training the student model
    # Example: masks = torch.load('path_to_masks.pt')
    masks = None  # Replace with actual masks
    
    # Load the student model
    evaluator.load_student_model(masks)
    
    # Define dataset configurations for evaluation
    dataset_configs = [
        {
            'name': '140k_dataset',
            'mode': '140k',
            'realfake140k_train_csv': '/path/to/140k/train.csv',
            'realfake140k_valid_csv': '/path/to/140k/valid.csv',
            'realfake140k_test_csv': '/path/to/140k/test.csv',
            'realfake140k_root_dir': '/path/to/140k/root'
        },
        {
            'name': 'rvf10k_dataset',
            'mode': 'rvf10k',
            'rvf10k_train_csv': '/path/to/rvf10k/train.csv',
            'rvf10k_valid_csv': '/path/to/rvf10k/valid.csv',
            'rvf10k_root_dir': '/path/to/rvf10k/root'
        },
        # Add more datasets as needed
    ]
    
    # Perform comprehensive evaluation
    results = evaluator.comprehensive_evaluation(dataset_configs)
    
    # Generate visualizations
    evaluator.plot_confusion_matrices(save_path='confusion_matrices.png')
    evaluator.plot_performance_comparison(save_path='performance_comparison.png')
    
    # Generate report
    evaluator.generate_report(save_path='generalization_report.txt')
    
    return evaluator, results


if __name__ == "__main__":
    evaluator, results = main()
