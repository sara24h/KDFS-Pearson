import os
import torch
import torch.nn as nn
import argparse
from model.pruned_model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal

def parse_args():
    parser = argparse.ArgumentParser(description="Load and save pruned ResNet50 model (architecture + weights)")
    parser.add_argument('--checkpoint_path', type=str, default='/kaggle/input/kdfs-4-mordad-140k-new-pearson-final-part1/results/run_resnet50_imagenet_prune1/student_model/ResNet_50_sparse_last.pt',
                        help="Path to the pruned model checkpoint")
    parser.add_argument('--save_path', type=str, default='/kaggle/working/results/full_model.pth',
                        help="Path to save the full model (architecture + weights)")
    return parser.parse_args()

def save_full_model(model, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model, save_path)
    print(f"Full model (architecture + weights) saved to {save_path}")

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # بارگذاری چک‌پوینت مدل
    checkpoint_path = args.checkpoint_path
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file {checkpoint_path} not found!")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

    if 'student' in checkpoint:
        state_dict = checkpoint['student']
    else:
        raise KeyError("'student' key not found in checkpoint")

    # استخراج ماسک‌های هرس
    masks = []
    mask_dict = {}
    for key, value in state_dict.items():
        if 'mask_weight' in key:
            mask_binary = (torch.argmax(value, dim=1).squeeze(1).squeeze(1) != 0).float()
            masks.append(mask_binary)
            mask_dict[key] = mask_binary
            preserved_filters = mask_binary.sum().int().item()
            print(f"Layer {key}: {preserved_filters} filters preserved")
    print(f"Number of masks extracted: {len(masks)}")
    if len(masks) != 48:
        print(f"Warning: Expected 48 masks for ResNet50, got {len(masks)}")

    # تعریف مدل هرس‌شده
    model = ResNet_50_pruned_hardfakevsreal(masks=masks)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)  # برای طبقه‌بندی باینری

    # آماده‌سازی state_dict هرس‌شده
    pruned_state_dict = {}
    for key, value in state_dict.items():
        if 'mask_weight' in key or 'feat' in key:
            continue
        if 'weight' in key and 'conv' in key:
            mask_key = key.replace('.weight', '.mask_weight')
            if mask_key in state_dict:
                mask_binary = mask_dict[mask_key].float()
                out_channels = mask_binary.sum().int().item()
                pruned_weight = value[mask_binary.bool()]
                if 'conv2' in key:
                    prev_mask_key = key.replace('conv2', 'conv1').replace('.weight', '.mask_weight')
                    if prev_mask_key in state_dict:
                        prev_mask_binary = mask_dict[prev_mask_key].float()
                        if prev_mask_binary.shape[0] != value.shape[1]:
                            print(f"Warning: Mismatch in input channels for {key}: expected {value.shape[1]}, got {prev_mask_binary.shape[0]}")
                            pruned_weight = pruned_weight[:, :prev_mask_binary.shape[0]][:, prev_mask_binary.bool()]
                        else:
                            pruned_weight = pruned_weight[:, prev_mask_binary.bool()]
                elif 'conv3' in key:
                    prev_mask_key = key.replace('conv3', 'conv2').replace('.weight', '.mask_weight')
                    if prev_mask_key in state_dict:
                        prev_mask_binary = mask_dict[prev_mask_key].float()
                        if prev_mask_binary.shape[0] != value.shape[1]:
                            print(f"Warning: Mismatch in input channels for {key}: expected {value.shape[1]}, got {prev_mask_binary.shape[0]}")
                            pruned_weight = pruned_weight[:, :prev_mask_binary.shape[0]][:, prev_mask_binary.bool()]
                        else:
                            pruned_weight = pruned_weight[:, prev_mask_binary.bool()]
                pruned_state_dict[key] = pruned_weight
            else:
                pruned_state_dict[key] = value
        elif 'bn' in key and any(s in key for s in ['weight', 'bias', 'running_mean', 'running_var']):
            conv_key = key.replace('.bn1.', '.conv1.').replace('.bn2.', '.conv2.').replace('.bn3.', '.conv3.')
            conv_key = conv_key.replace('.weight', '.mask_weight').replace('.bias', '.mask_weight').replace('.running_mean', '.mask_weight').replace('.running_var', '.mask_weight')
            if conv_key in state_dict:
                mask_binary = mask_dict[conv_key].float()
                pruned_param = value[mask_binary.bool()]
                pruned_state_dict[key] = pruned_param
            else:
                pruned_state_dict[key] = value
        else:
            pruned_state_dict[key] = value

    # بارگذاری state_dict هرس‌شده
    missing, unexpected = model.load_state_dict(pruned_state_dict, strict=False)
    print(f"Missing keys: {missing}")
    print(f"Unexpected keys: {unexpected}")
    model = model.to(device)

    # ذخیره کل مدل
    save_full_model(model, args.save_path)

if __name__ == "__main__":
    main()
