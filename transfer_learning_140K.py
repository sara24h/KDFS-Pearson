import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from thop import profile
import argparse
from model.pruned_model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal
from model.student.ResNet_sparse import ResNet_50_sparse_rvf10k

def get_data_loaders(data_dir, batch_size, img_size=256):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
    val_dataset = datasets.ImageFolder(root=f"{data_dir}/valid", transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader

def evaluate(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = running_loss / len(data_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def train(model, train_loader, val_loader, criterion, optimizer, device, epochs, output_dir):
    model.train()
    best_acc = 0.0
    best_model_path = f"{output_dir}/best_model.pt"
    
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model with accuracy {val_acc:.2f}% at {best_model_path}")
    
    return best_acc

def apply_pruning_to_state_dict(state_dict, masks, layer_names):
    """
    فیلتر کردن وزن‌های state_dict بر اساس ماسک‌های هرس
    """
    pruned_state_dict = {}
    mask_idx = 0
    
    for name, param in state_dict.items():
        if name in layer_names:
            # پیدا کردن ماسک مربوطه
            mask = masks[mask_idx]
            # اگر ماسک تک‌بعدی است، مستقیماً از آن استفاده می‌کنیم
            if mask.dim() == 1:
                indices = mask.nonzero(as_tuple=True)[0]
            else:
                # اگر ماسک چند‌بعدی است (مثل [C, 1] یا [C, 1, 1])
                indices = torch.argmax(mask, dim=0).nonzero(as_tuple=True)[0]
            
            # برای وزن‌های conv (شکل: [out_channels, in_channels, k, k])
            if "conv" in name and "weight" in name:
                pruned_param = param[indices]
                # برای لایه‌های conv2 و conv3، ورودی‌ها نیز باید فیلتر شوند
                if "conv2" in name or "conv3" in name:
                    prev_mask = masks[mask_idx-1]
                    prev_indices = prev_mask.nonzero(as_tuple=True)[0] if prev_mask.dim() == 1 else torch.argmax(prev_mask, dim=0).nonzero(as_tuple=True)[0]
                    pruned_param = pruned_param[:, prev_indices]
                pruned_state_dict[name] = pruned_param
                mask_idx += 1
            
            # برای bn (weight, bias, running_mean, running_var)
            elif "bn" in name:
                pruned_state_dict[name] = param[indices]
            
        else:
            # وزن‌هایی که نیازی به هرس ندارند (مثل fc.weight)
            pruned_state_dict[name] = param
    
    return pruned_state_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the student checkpoint')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to rvf10k dataset')
    parser.add_argument('--dataset_mode', type=str, default='rvf10k', help='Dataset mode')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate for most layers')
    parser.add_argument('--lr_layer4', type=float, default=0.0001, help='Learning rate for layer4')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--output_dir', type=str, default='/kaggle/working', help='Output directory for saving models')
    args = parser.parse_args()

    # تنظیم دستگاه
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # بارگذاری مدل دانش‌آموز برای استخراج ماسک‌ها
    student = ResNet_50_sparse_rvf10k().to(device)
    ckpt_student = torch.load(args.checkpoint_path, map_location="cpu")
    student.load_state_dict(ckpt_student["student"])
    mask_weights = [m.mask_weight for m in student.mask_modules]
    masks = [mask_weight if mask_weight.dim() == 1 else torch.argmax(mask_weight, dim=0).squeeze() for mask_weight in mask_weights]

    # بررسی تعداد کانال‌های حفظ‌شده
    layer_names = [f"layer{i}.{j}.conv{k}.weight" for i in range(1, 5) for j in range([3, 4, 6, 3][i-1]) for k in range(1, 4)]
    for i, mask in enumerate(masks):
        print(f"Mask {i} ({layer_names[i]}): {mask.sum().item()}/{mask.numel()} channels kept")

    # بارگذاری مدل هرس‌شده
    model = ResNet_50_pruned_hardfakevsreal(masks=masks).to(device)
    
    # فیلتر کردن state_dict بر اساس ماسک‌ها
    pruned_state_dict = apply_pruning_to_state_dict(ckpt_student["student"], masks, layer_names)
    
    # بارگذاری state_dict هرس‌شده
    model_dict = model.state_dict()
    model_dict.update(pruned_state_dict)
    model.load_state_dict(model_dict)

    # محاسبه فلاپس و پارامترها
    input = torch.rand([1, 3, 256, 256]).to(device)
    flops, params = profile(model, inputs=(input,), verbose=True)
    print(f"FLOPs: {flops / (10**9):.2f} GMac")
    print(f"Parameters: {params / (10**6):.2f} M")

    # بارگذاری داده‌ها
    train_loader, val_loader = get_data_loaders(args.data_dir, args.batch_size, img_size=256)

    # تعریف معیار و بهینه‌ساز
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {'params': [p for n, p in model.named_parameters() if 'layer4' not in n], 'lr': args.lr},
        {'params': [p for n, p in model.named_parameters() if 'layer4' in n], 'lr': args.lr_layer4}
    ], weight_decay=args.weight_decay)

    # آموزش و ارزیابی
    best_acc = train(model, train_loader, val_loader, criterion, optimizer, device, args.epochs, args.output_dir)
    
    # ارزیابی نهایی روی مجموعه تست
    final_loss, final_acc = evaluate(model, val_loader, criterion, device)
    print(f"Final Test Loss: {final_loss:.4f}, Final Test Accuracy: {final_acc:.2f}%")

    # ذخیره نتایج
    with open(f"{args.output_dir}/results.txt", "w") as f:
        f.write(f"FLOPs: {flops / (10**9):.2f} GMac\n")
        f.write(f"Parameters: {params / (10**6):.2f} M\n")
        f.write(f"Final Test Accuracy: {final_acc:.2f}%\n")

if __name__ == "__main__":
    main()
