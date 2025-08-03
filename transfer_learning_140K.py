import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Import your classes (make sure the imports match your file structure)
from data.dataset import Dataset_selector, FaceDataset
from model.pruned_model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal

def inspect_checkpoint_structure(checkpoint_path):
    """Inspect checkpoint structure to understand what's inside"""
    print("="*60)
    print("CHECKPOINT INSPECTION")
    print("="*60)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"Checkpoint type: {type(checkpoint)}")
    print("\nMain keys in checkpoint:")
    
    if isinstance(checkpoint, dict):
        for key, value in checkpoint.items():
            print(f"  {key}: {type(value)}")
            if isinstance(value, dict):
                print(f"    -> Sub-keys: {list(value.keys())[:5]}...")
            elif hasattr(value, 'shape'):
                print(f"    -> Shape: {value.shape}")
            elif isinstance(value, (list, tuple)):
                print(f"    -> Length: {len(value)}")
    
    return checkpoint

def extract_masks_from_checkpoint(checkpoint_path):
    """
    Extract masks from checkpoint or generate them based on model structure
    """
    print("\n" + "="*60)
    print("MASK EXTRACTION")
    print("="*60)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Method 1: Direct masks in checkpoint
    if 'masks' in checkpoint:
        masks = checkpoint['masks']
        print(f"✓ Found 'masks' in checkpoint with {len(masks)} masks")
        return masks
    
    # Method 2: Look for mask-related keys
    mask_keys = [k for k in checkpoint.keys() if 'mask' in k.lower()]
    if mask_keys:
        print(f"✓ Found mask-related keys: {mask_keys}")
        return checkpoint[mask_keys[0]]
    
    # Method 3: Extract from model state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print("✓ Using model_state_dict to generate masks")
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print("✓ Using state_dict to generate masks")
    else:
        state_dict = checkpoint
        print("✓ Using checkpoint as state_dict to generate masks")
    
    # Generate masks from state dict
    masks = generate_masks_from_state_dict(state_dict)
    return masks

def generate_masks_from_state_dict(state_dict):
    """
    Generate masks based on the actual model weights
    This works by examining the conv layer weights in the state dict
    """
    masks = []
    
    # ResNet50 structure: 4 layers with [3, 4, 6, 3] blocks
    # Each Bottleneck block has 3 conv layers
    layer_configs = [
        ('layer1', 3),  # 3 blocks
        ('layer2', 4),  # 4 blocks  
        ('layer3', 6),  # 6 blocks
        ('layer4', 3),  # 3 blocks
    ]
    
    print("Generating masks from state dict...")
    
    for layer_name, num_blocks in layer_configs:
        for block_idx in range(num_blocks):
            for conv_idx in [1, 2, 3]:  # conv1, conv2, conv3
                weight_key = f"{layer_name}.{block_idx}.conv{conv_idx}.weight"
                
                if weight_key in state_dict:
                    weight_tensor = state_dict[weight_key]
                    num_filters = weight_tensor.shape[0]  # Output channels
                    
                    # Create mask with all filters enabled (no actual pruning info)
                    mask = torch.ones(num_filters, dtype=torch.bool)
                    masks.append(mask)
                    
                    print(f"  {weight_key}: {num_filters} filters -> mask created")
                else:
                    print(f"  Warning: {weight_key} not found in state_dict")
    
    print(f"Generated {len(masks)} masks total")
    return masks

def create_default_resnet50_masks():
    """
    Create default masks for standard ResNet50 architecture
    Use this if mask extraction fails
    """
    print("Creating default ResNet50 masks (no pruning assumed)...")
    
    masks = []
    
    # Standard ResNet50 Bottleneck filter sizes
    block_configs = [
        # Layer 1: 3 blocks
        [(64, 64, 256)] * 3,
        # Layer 2: 4 blocks  
        [(128, 128, 512)] * 4,
        # Layer 3: 6 blocks
        [(256, 256, 1024)] * 6,
        # Layer 4: 3 blocks
        [(512, 512, 2048)] * 3,
    ]
    
    for layer_blocks in block_configs:
        for conv1_filters, conv2_filters, conv3_filters in layer_blocks:
            masks.extend([
                torch.ones(conv1_filters, dtype=torch.bool),
                torch.ones(conv2_filters, dtype=torch.bool), 
                torch.ones(conv3_filters, dtype=torch.bool),
            ])
    
    print(f"Created {len(masks)} default masks")
    return masks

def load_and_test_student_model(checkpoint_path, test_dataloader, device='cuda'):
    """
    Load student model and test on dataset with comprehensive error handling
    """
    print("="*60)
    print("LOADING AND TESTING STUDENT MODEL")
    print("="*60)
    
    # Step 1: Inspect checkpoint
    try:
        checkpoint_info = inspect_checkpoint_structure(checkpoint_path)
    except Exception as e:
        print(f"Error inspecting checkpoint: {e}")
        return None
    
    # Step 2: Extract masks
    try:
        masks = extract_masks_from_checkpoint(checkpoint_path)
        print(f"✓ Successfully obtained {len(masks)} masks")
    except Exception as e:
        print(f"Error extracting masks: {e}")
        print("Falling back to default masks...")
        masks = create_default_resnet50_masks()
    
    # Step 3: Create model
    try:
        print("\nCreating ResNet50 pruned model...")
        model = ResNet_50_pruned_hardfakevsreal(masks)
        print("✓ Model created successfully")
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Total parameters: {total_params:,}")
        
    except Exception as e:
        print(f"✗ Error creating model: {e}")
        return None
    
    # Step 4: Load weights
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Try different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✓ Loaded weights from 'model_state_dict'")
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            print("✓ Loaded weights from 'state_dict'")
        else:
            model.load_state_dict(checkpoint)
            print("✓ Loaded weights directly from checkpoint")
            
    except Exception as e:
        print(f"✗ Error loading weights: {e}")
        return None
    
    # Step 5: Prepare model for evaluation
    model.to(device)
    model.eval()
    print(f"✓ Model moved to {device} and set to eval mode")
    
    # Step 6: Evaluate on dataset
    print(f"\nStarting evaluation on dataset...")
    return evaluate_model(model, test_dataloader, device)

def evaluate_model(model, dataloader, device):
    """Evaluate model on dataloader"""
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    total_loss = 0.0
    
    criterion = nn.BCEWithLogitsLoss()
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Evaluating")):
            try:
                images = images.to(device)
                labels = labels.to(device).float()
                
                # Forward pass
                outputs, _ = model(images)
                outputs = outputs.squeeze()
                
                # Ensure outputs and labels have same shape
                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
                if labels.dim() == 0:
                    labels = labels.unsqueeze(0)
                
                # Calculate loss
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                # Get predictions and probabilities
                probabilities = torch.sigmoid(outputs)
                predictions = (probabilities > 0.5).float()
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
    
    # Calculate metrics
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    if len(all_predictions) == 0:
        print("No valid predictions made!")
        return None
    
    # Calculate performance metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='binary', zero_division=0
    )
    
    try:
        auc = roc_auc_score(all_labels, all_probabilities)
    except:
        auc = 0.0
    
    avg_loss = total_loss / len(dataloader)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'loss': avg_loss,
        'confusion_matrix': cm,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'true_positives': tp,
        'total_samples': len(all_labels)
    }
    
    # Print results
    print("\n" + "="*40)
    print("EVALUATION RESULTS")
    print("="*40)
    print(f"Total Samples: {len(all_labels)}")
    print(f"Accuracy:      {accuracy:.4f}")
    print(f"Precision:     {precision:.4f}")
    print(f"Recall:        {recall:.4f}")
    print(f"F1-Score:      {f1:.4f}")
    print(f"AUC:           {auc:.4f}")
    print(f"Loss:          {avg_loss:.4f}")
    print(f"Confusion Matrix:")
    print(f"  TN: {tn:4d}  FP: {fp:4d}")
    print(f"  FN: {fn:4d}  TP: {tp:4d}")
    
    return results

def test_generalization_on_140k():
    """Quick test on 140k dataset"""
    print("QUICK GENERALIZATION TEST ON 140K DATASET")
    print("="*60)
    
    try:
        # Create 140k dataset
        dataset = Dataset_selector(
            dataset_mode='140k',
            realfake140k_train_csv='/kaggle/input/140k-real-and-fake-faces/train.csv',
            realfake140k_valid_csv='/kaggle/input/140k-real-and-fake-faces/valid.csv', 
            realfake140k_test_csv='/kaggle/input/140k-real-and-fake-faces/test.csv',
            realfake140k_root_dir='/kaggle/input/140k-real-and-fake-faces',
            train_batch_size=16,  # Smaller batch size for safety
            eval_batch_size=16,
            num_workers=2,
            ddp=False,
        )
        
        checkpoint_path = '/kaggle/input/kdfs-4-mordad-140k-new-pearson-final-part1/results/run_resnet50_imagenet_prune1/student_model/ResNet_50_sparse_last.pt'
        
        # Test on validation set
        print("\nTesting on VALIDATION set:")
        val_results = load_and_test_student_model(checkpoint_path, dataset.loader_val)
        
        # Test on test set  
        print("\nTesting on TEST set:")
        test_results = load_and_test_student_model(checkpoint_path, dataset.loader_test)
        
        return val_results, test_results
        
    except Exception as e:
        print(f"Error in generalization test: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def plot_results(val_results, test_results, save_path=None):
    """Plot comparison between validation and test results"""
    if not val_results or not test_results:
        print("Cannot plot - missing results")
        return
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
    val_values = [val_results[m] for m in metrics]
    test_values = [test_results[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.bar(x - width/2, val_values, width, label='Validation', alpha=0.8)
    ax.bar(x + width/2, test_values, width, label='Test', alpha=0.8)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('Student Model Performance: Validation vs Test')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    # Add value labels
    for i, (val, test) in enumerate(zip(val_values, test_values)):
        ax.text(i - width/2, val + 0.01, f'{val:.3f}', ha='center', va='bottom')
        ax.text(i + width/2, test + 0.01, f'{test:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()

# Main execution
if __name__ == "__main__":
    print("STUDENT MODEL GENERALIZATION TESTING")
    print("="*60)
    
    # Run the test
    val_results, test_results = test_generalization_on_140k()
    
    # Plot results if successful
    if val_results and test_results:
        plot_results(val_results, test_results, 'generalization_results.png')
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Validation Accuracy: {val_results['accuracy']:.4f}")
        print(f"Test Accuracy:       {test_results['accuracy']:.4f}")
        print(f"Validation F1:       {val_results['f1_score']:.4f}")
        print(f"Test F1:             {test_results['f1_score']:.4f}")
        print("✓ Generalization test completed successfully!")
    else:
        print("✗ Generalization test failed!")
