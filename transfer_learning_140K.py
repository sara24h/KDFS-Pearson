import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Import your classes
# from data.dataset import Dataset_selector
# from model.pruned_model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal

def analyze_student_checkpoint(checkpoint_path):
    """Analyze the specific checkpoint structure"""
    print("="*60)
    print("ANALYZING STUDENT CHECKPOINT")
    print("="*60)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    print(f"Best precision: {checkpoint.get('best_prec1', 'N/A')}")
    print(f"Start epoch: {checkpoint.get('start_epoch', 'N/A')}")
    
    # Analyze student model structure
    if 'student' in checkpoint:
        student_state = checkpoint['student']
        print(f"\nStudent model has {len(student_state)} parameters")
        
        # Look for conv layers to understand pruning
        conv_layers = {}
        for key, value in student_state.items():
            if 'conv' in key and 'weight' in key:
                conv_layers[key] = value.shape
        
        print(f"\nConv layers found: {len(conv_layers)}")
        for key, shape in list(conv_layers.items())[:10]:  # Show first 10
            print(f"  {key}: {shape}")
        
        return student_state, conv_layers
    else:
        print("No 'student' key found!")
        return None, None

def extract_masks_from_conv_layers(conv_layers):
    """Extract masks based on actual conv layer sizes in the pruned model"""
    print("\n" + "="*60)
    print("EXTRACTING MASKS FROM CONV LAYERS")
    print("="*60)
    
    masks = []
    
    # ResNet50 layer structure
    layer_patterns = [
        ('layer1', 3),  # 3 blocks
        ('layer2', 4),  # 4 blocks
        ('layer3', 6),  # 6 blocks
        ('layer4', 3),  # 3 blocks
    ]
    
    for layer_name, num_blocks in layer_patterns:
        for block_idx in range(num_blocks):
            for conv_idx in [1, 2, 3]:  # conv1, conv2, conv3
                key = f"{layer_name}.{block_idx}.conv{conv_idx}.weight"
                
                if key in conv_layers:
                    out_channels = conv_layers[key][0]
                    mask = torch.ones(out_channels, dtype=torch.bool)
                    masks.append(mask)
                    print(f"  {key}: {out_channels} filters -> mask created")
                else:
                    print(f"  Warning: {key} not found")
    
    print(f"\nTotal masks created: {len(masks)}")
    return masks

def create_and_load_student_model(checkpoint_path, device='cuda'):
    """Create and load the student model with proper error handling"""
    print("="*60)
    print("CREATING AND LOADING STUDENT MODEL")
    print("="*60)
    
    # Step 1: Analyze checkpoint
    student_state, conv_layers = analyze_student_checkpoint(checkpoint_path)
    if student_state is None:
        print("Failed to extract student state")
        return None
    
    # Step 2: Extract masks
    masks = extract_masks_from_conv_layers(conv_layers)
    if not masks:
        print("Failed to extract masks, using defaults")
        masks = create_default_resnet50_masks()
    
    # Step 3: Create model
    try:
        print(f"\nCreating ResNet50 with {len(masks)} masks...")
        model = ResNet_50_pruned_hardfakevsreal(masks)
        print("✓ Model created successfully")
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Total parameters: {total_params:,}")
        
    except Exception as e:
        print(f"✗ Error creating model: {e}")
        print("Trying with default masks...")
        try:
            default_masks = create_default_resnet50_masks()
            model = ResNet_50_pruned_hardfakevsreal(default_masks)
            print("✓ Model created with default masks")
        except Exception as e2:
            print(f"✗ Still failed: {e2}")
            return None
    
    # Step 4: Load weights
    try:
        model.load_state_dict(student_state)
        print("✓ Student weights loaded successfully")
    except Exception as e:
        print(f"✗ Error loading weights: {e}")
        return None
    
    # Step 5: Prepare for inference
    model.to(device)
    model.eval()
    print(f"✓ Model ready on {device}")
    
    return model

def create_default_resnet50_masks():
    """Create default masks for ResNet50"""
    print("Creating default ResNet50 masks...")
    
    masks = []
    
    # Standard ResNet50 Bottleneck structure
    # Layer 1: 3 blocks, each with [64, 64, 256] filters  
    for _ in range(3):
        masks.extend([
            torch.ones(64, dtype=torch.bool),
            torch.ones(64, dtype=torch.bool),
            torch.ones(256, dtype=torch.bool)
        ])
    
    # Layer 2: 4 blocks, each with [128, 128, 512] filters
    for _ in range(4):
        masks.extend([
            torch.ones(128, dtype=torch.bool),
            torch.ones(128, dtype=torch.bool),
            torch.ones(512, dtype=torch.bool)
        ])
    
    # Layer 3: 6 blocks, each with [256, 256, 1024] filters
    for _ in range(6):
        masks.extend([
            torch.ones(256, dtype=torch.bool),
            torch.ones(256, dtype=torch.bool),
            torch.ones(1024, dtype=torch.bool)
        ])
    
    # Layer 4: 3 blocks, each with [512, 512, 2048] filters
    for _ in range(3):
        masks.extend([
            torch.ones(512, dtype=torch.bool),
            torch.ones(512, dtype=torch.bool),
            torch.ones(2048, dtype=torch.bool)
        ])
    
    print(f"Created {len(masks)} default masks")
    return masks

def evaluate_student_model(model, dataloader, device='cuda'):
    """Evaluate the student model"""
    print("="*50)
    print("EVALUATING STUDENT MODEL")
    print("="*50)
    
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
                
                # Handle single sample batches
                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
                if labels.dim() == 0:
                    labels = labels.unsqueeze(0)
                
                # Calculate loss
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                # Get predictions
                probabilities = torch.sigmoid(outputs)
                predictions = (probabilities > 0.5).float()
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
    
    if len(all_predictions) == 0:
        print("No predictions made!")
        return None
    
    # Calculate metrics
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
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
        'total_samples': len(all_labels),
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
    }
    
    # Print results
    print(f"\nEVALUATION RESULTS:")
    print(f"Total Samples: {len(all_labels)}")
    print(f"Accuracy:      {accuracy:.4f}")
    print(f"Precision:     {precision:.4f}")
    print(f"Recall:        {recall:.4f}")
    print(f"F1-Score:      {f1:.4f}")
    print(f"AUC:           {auc:.4f}")
    print(f"Loss:          {avg_loss:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"              Fake  Real")
    print(f"Actual Fake   {tn:4d}  {fp:4d}")
    print(f"       Real   {fn:4d}  {tp:4d}")
    
    return results

def test_student_generalization():
    """Main function to test student model generalization"""
    print("STUDENT MODEL GENERALIZATION TEST")
    print("="*60)
    
    checkpoint_path = '/kaggle/input/kdfs-4-mordad-140k-new-pearson-final-part1/results/run_resnet50_imagenet_prune1/student_model/ResNet_50_sparse_last.pt'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Step 1: Create and load model
    model = create_and_load_student_model(checkpoint_path, device)
    if model is None:
        print("Failed to load student model!")
        return None, None
    
    # Step 2: Create dataset
    try:
        print("\nCreating 140k dataset...")
        dataset = Dataset_selector(
            dataset_mode='140k',
            realfake140k_train_csv='/kaggle/input/140k-real-and-fake-faces/train.csv',
            realfake140k_valid_csv='/kaggle/input/140k-real-and-fake-faces/valid.csv',
            realfake140k_test_csv='/kaggle/input/140k-real-and-fake-faces/test.csv',
            realfake140k_root_dir='/kaggle/input/140k-real-and-fake-faces',
            train_batch_size=16,
            eval_batch_size=16,
            num_workers=2,
            ddp=False,
        )
        print("✓ Dataset created successfully")
        
    except Exception as e:
        print(f"Error creating dataset: {e}")
        return None, None
    
    # Step 3: Evaluate on validation set
    print("\n" + "="*60)
    print("TESTING ON VALIDATION SET")
    print("="*60)
    val_results = evaluate_student_model(model, dataset.loader_val, device)
    
    # Step 4: Evaluate on test set
    print("\n" + "="*60)
    print("TESTING ON TEST SET")
    print("="*60)
    test_results = evaluate_student_model(model, dataset.loader_test, device)
    
    return val_results, test_results

def plot_comparison(val_results, test_results):
    """Plot validation vs test results"""
    if not val_results or not test_results:
        return
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
    val_values = [val_results[m] for m in metrics]
    test_values = [test_results[m] for m in metrics]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar plot
    x = np.arange(len(metrics))
    width = 0.35
    
    ax1.bar(x - width/2, val_values, width, label='Validation', alpha=0.8, color='skyblue')
    ax1.bar(x + width/2, test_values, width, label='Test', alpha=0.8, color='lightcoral')
    
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Score')
    ax1.set_title('Student Model: Validation vs Test Performance')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.1)
    
    # Add value labels
    for i, (val, test) in enumerate(zip(val_values, test_values)):
        ax1.text(i - width/2, val + 0.01, f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        ax1.text(i + width/2, test + 0.01, f'{test:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Confusion matrices
    cm_val = val_results['confusion_matrix']
    cm_test = test_results['confusion_matrix']
    
    # Validation confusion matrix
    ax2.text(0.25, 0.8, 'Validation', transform=ax2.transAxes, ha='center', fontsize=12, weight='bold')
    ax2.text(0.25, 0.7, f"Acc: {val_results['accuracy']:.3f}", transform=ax2.transAxes, ha='center', fontsize=10)
    ax2.text(0.25, 0.6, f"F1: {val_results['f1_score']:.3f}", transform=ax2.transAxes, ha='center', fontsize=10)
    
    # Test confusion matrix  
    ax2.text(0.75, 0.8, 'Test', transform=ax2.transAxes, ha='center', fontsize=12, weight='bold')
    ax2.text(0.75, 0.7, f"Acc: {test_results['accuracy']:.3f}", transform=ax2.transAxes, ha='center', fontsize=10)
    ax2.text(0.75, 0.6, f"F1: {test_results['f1_score']:.3f}", transform=ax2.transAxes, ha='center', fontsize=10)
    
    # Simple confusion matrix display
    ax2.text(0.25, 0.4, f"TN: {val_results['tn']}", transform=ax2.transAxes, ha='center', fontsize=9)
    ax2.text(0.25, 0.3, f"FP: {val_results['fp']}", transform=ax2.transAxes, ha='center', fontsize=9)
    ax2.text(0.25, 0.2, f"FN: {val_results['fn']}", transform=ax2.transAxes, ha='center', fontsize=9)
    ax2.text(0.25, 0.1, f"TP: {val_results['tp']}", transform=ax2.transAxes, ha='center', fontsize=9)
    
    ax2.text(0.75, 0.4, f"TN: {test_results['tn']}", transform=ax2.transAxes, ha='center', fontsize=9)
    ax2.text(0.75, 0.3, f"FP: {test_results['fp']}", transform=ax2.transAxes, ha='center', fontsize=9)
    ax2.text(0.75, 0.2, f"FN: {test_results['fn']}", transform=ax2.transAxes, ha='center', fontsize=9)
    ax2.text(0.75, 0.1, f"TP: {test_results['tp']}", transform=ax2.transAxes, ha='center', fontsize=9)
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_title('Confusion Matrix Summary')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('student_generalization_results.png', dpi=300, bbox_inches='tight')
    plt.show()

# Main execution
if __name__ == "__main__":
    # Run the generalization test
    val_results, test_results = test_student_generalization()
    
    if val_results and test_results:
        print("\n" + "="*60)
        print("FINAL SUMMARY")
        print("="*60)
        print(f"Validation Accuracy: {val_results['accuracy']:.4f}")
        print(f"Test Accuracy:       {test_results['accuracy']:.4f}")
        print(f"Validation F1:       {val_results['f1_score']:.4f}")
        print(f"Test F1:             {test_results['f1_score']:.4f}")
        print(f"Generalization Gap:  {abs(val_results['accuracy'] - test_results['accuracy']):.4f}")
        
        # Plot results
        plot_comparison(val_results, test_results)
        
        print("\n✓ Student model generalization test completed!")
    else:
        print("\n✗ Generalization test failed!")
