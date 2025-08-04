import torch
import torch.nn as nn
from your_pruned_model import ResNet_50_pruned_hardfakevsreal  # Import your pruned model
from data.dataset import FaceDataset, Dataset_selector

def load_pruned_model_weights(model_path, masks, device='cuda'):
    """
    Load weights onto a pruned ResNet model
    
    Args:
        model_path (str): Path to the saved model weights
        masks (list): List of pruning masks for the model
        device (str): Device to load the model on
    
    Returns:
        model: Loaded pruned ResNet model
    """
    
    # Create the pruned model instance
    model = ResNet_50_pruned_hardfakevsreal(masks=masks)
    
    try:
        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                # Assume the checkpoint itself is the state_dict
                state_dict = checkpoint
        else:
            # If checkpoint is directly the model
            state_dict = checkpoint.state_dict() if hasattr(checkpoint, 'state_dict') else checkpoint
        
        # Remove 'module.' prefix if present (from DataParallel/DistributedDataParallel)
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                new_key = key[7:]  # Remove 'module.' prefix
            else:
                new_key = key
            new_state_dict[new_key] = value
        
        # Load weights with strict=False to handle missing/extra keys
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        
        # Print information about loading
        if missing_keys:
            print(f"Missing keys in state_dict: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys in state_dict: {unexpected_keys}")
            
        print("Model weights loaded successfully!")
        
        # Move model to device
        model = model.to(device)
        
        # Set to evaluation mode
        model.eval()
        
        return model
        
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return None

def load_masks_from_checkpoint(model_path):
    """
    Try to extract masks from the checkpoint if they're saved there
    
    Args:
        model_path (str): Path to the saved model
    
    Returns:
        masks (list): List of masks if found, None otherwise
    """
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if isinstance(checkpoint, dict) and 'masks' in checkpoint:
            return checkpoint['masks']
        
        print("No masks found in checkpoint. You'll need to provide them manually.")
        return None
        
    except Exception as e:
        print(f"Error loading masks from checkpoint: {e}")
        return None

# Example usage with dataset loading
def main():
    # Path to your saved model
    model_path = '/kaggle/input/kdfs-4-mordad-140k-new-pearson-final-part1/results/run_resnet50_imagenet_prune1/student_model/ResNet_50_sparse_last.pt'
    
    # First, try to load masks from checkpoint
    masks = load_masks_from_checkpoint(model_path)
    
    if masks is None:
        print("Masks not found in checkpoint. You need to provide the masks used during pruning.")
        print("Please load the masks from your pruning process or provide them manually.")
        
        # If you have the masks saved separately, load them here
        # Example:
        # masks = torch.load('path_to_your_masks.pt')
        # or create dummy masks for testing (NOT recommended for actual use)
        # This is just an example - you need the actual masks used during pruning
        print("Creating dummy masks for demonstration - REPLACE WITH ACTUAL MASKS!")
        
        # Create dummy masks for ResNet-50 with Bottleneck blocks
        # ResNet-50 structure: [3, 4, 6, 3] blocks, each block has 3 conv layers
        # Total: (3+4+6+3) * 3 = 48 conv layers in residual blocks
        print("Creating dummy masks for ResNet-50 - REPLACE WITH ACTUAL MASKS!")
        
        masks = []
        num_blocks = [3, 4, 6, 3]  # ResNet-50 structure
        base_channels = [64, 128, 256, 512]  # Base channels for each stage
        
        for stage_idx, (num_block, base_ch) in enumerate(zip(num_blocks, base_channels)):
            for block_idx in range(num_block):
                # Each Bottleneck block has 3 conv layers
                # Conv1: 1x1, reduces channels
                # Conv2: 3x3, processes features  
                # Conv3: 1x1, expands channels
                
                # Layer 1: 1x1 conv (channel reduction)
                mask1_size = base_ch
                mask1 = torch.ones(mask1_size, dtype=torch.bool)
                if stage_idx > 0 or block_idx > 0:  # Don't prune first few layers too much
                    num_to_prune = int(0.1 * mask1_size)  # Prune 10%
                    mask1[-num_to_prune:] = False
                masks.append(mask1)
                
                # Layer 2: 3x3 conv (same channels)
                mask2_size = base_ch
                mask2 = torch.ones(mask2_size, dtype=torch.bool)
                if stage_idx > 0 or block_idx > 0:
                    num_to_prune = int(0.15 * mask2_size)  # Prune 15%
                    mask2[-num_to_prune:] = False
                masks.append(mask2)
                
                # Layer 3: 1x1 conv (channel expansion, output should match residual)
                mask3_size = base_ch * 4  # Bottleneck expansion factor is 4
                mask3 = torch.ones(mask3_size, dtype=torch.bool)
                # Don't prune the final layer of each block too much to maintain residual connection
                if stage_idx > 1:  # Only prune later stages lightly
                    num_to_prune = int(0.05 * mask3_size)  # Prune only 5%
                    mask3[-num_to_prune:] = False
                masks.append(mask3)
        
        print(f"Created {len(masks)} dummy masks for ResNet-50")
        print("Mask sizes:", [mask.sum().item() for mask in masks[:9]], "...")  # Show first 9 mask sizes
    
    # Load the model with weights
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_pruned_model_weights(model_path, masks, device)
    
    if model is not None:
        print("Model loaded successfully!")
        print(f"Model is on device: {next(model.parameters()).device}")
        
        # Load dataset for testing (using your Dataset_selector)
        dataset = Dataset_selector(
            dataset_mode='140k',
            realfake140k_train_csv='/kaggle/input/140k-real-and-fake-faces/train.csv',
            realfake140k_valid_csv='/kaggle/input/140k-real-and-fake-faces/valid.csv',
            realfake140k_test_csv='/kaggle/input/140k-real-and-fake-faces/test.csv',
            realfake140k_root_dir='/kaggle/input/140k-real-and-fake-faces',
            train_batch_size=32,
            eval_batch_size=32,
            num_workers=4,
            pin_memory=True,
            ddp=False,
        )
        
        # Test the model with real data
        model.eval()
        with torch.no_grad():
            # Get a batch from test loader
            test_batch = next(iter(dataset.loader_test))
            images, labels = test_batch
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs, features = model(images)
            predictions = torch.sigmoid(outputs)
            
            print(f"Batch size: {images.shape[0]}")
            print(f"Input shape: {images.shape}")
            print(f"Output shape: {outputs.shape}")
            print(f"Predictions (after sigmoid): {predictions.flatten()[:5]}...")  # Show first 5 predictions
            print(f"True labels: {labels.flatten()[:5]}...")  # Show first 5 labels
            print(f"Number of feature maps: {len(features)}")
            for i, feat in enumerate(features):
                print(f"Feature map {i+1} shape: {feat.shape}")
    
    return model

if __name__ == "__main__":
    loaded_model = main()
