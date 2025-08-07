import torch
from data.dataset import Dataset_selector
from tqdm import tqdm

# تابع تست مدل
def test_model(model, test_loader, device, dataset_name):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"Testing {dataset_name}", ncols=100):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()
            outputs, _ = model(images)
            outputs = outputs.squeeze()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = 100.0 * correct / total
    print(f"Accuracy on {dataset_name} test dataset: {accuracy:.2f}%")
    return accuracy

# تنظیمات عمومی
saved_model_path = "/kaggle/input/pruned_resnet50_140k/pytorch/default/1/pruned_model (3).pt"  # مسیر مدل ذخیره‌شده
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eval_batch_size = 64
num_workers = 8
pin_memory = True

# لود مدل
model = torch.load(saved_model_path, map_location=device)
model.to(device)
model.eval()
print("Model loaded successfully.")

# لیست دیتاست‌ها و تنظیمات مربوطه
datasets = [
    {
        "mode": "hardfake",
        "params": {
            "hardfake_csv_file": "/path/to/hardfakevsrealfaces/data.csv",
            "hardfake_root_dir": "/path/to/hardfakevsrealfaces",
        }
    },
    {
        "mode": "rvf10k",
        "params": {
            "rvf10k_train_csv": "/path/to/rvf10k/train.csv",
            "rvf10k_valid_csv": "/path/to/rvf10k/valid.csv",
            "rvf10k_root_dir": "/path/to/rvf10k",
        }
    },
    {
        "mode": "140k",
        "params": {
            "realfake140k_train_csv": "/path/to/140k-real-and-fake-faces/train.csv",
            "realfake140k_valid_csv": "/path/to/140k-real-and-fake-faces/valid.csv",
            "realfake140k_test_csv": "/path/to/140k-real-and-fake-faces/test.csv",
            "realfake140k_root_dir": "/path/to/140k-real-and-fake-faces",
        }
    },
    {
        "mode": "190k",
        "params": {
            "realfake190k_root_dir": "/path/to/deepfake-and-real-images/Dataset",
        }
    },
    {
        "mode": "200k",
        "params": {
            "realfake200k_train_csv": "/path/to/200k-real-and-fake-faces/train_labels.csv",
            "realfake200k_val_csv": "/path/to/200k-real-and-fake-faces/val_labels.csv",
            "realfake200k_test_csv": "/path/to/200k-real-and-fake-faces/test_labels.csv",
            "realfake200k_root_dir": "/path/to/200k-real-and-fake-faces",
        }
    },
    {
        "mode": "330k",
        "params": {
            "realfake330k_root_dir": "/path/to/deepfake-dataset",
        }
    }
]

# تست مدل روی تمام دیتاست‌ها
results = {}
for dataset in datasets:
    try:
        dataset_mode = dataset["mode"]
        params = dataset["params"]
        params.update({
            "dataset_mode": dataset_mode,
            "eval_batch_size": eval_batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "ddp": False
        })

        # ایجاد Dataset_selector
        print(f"\nLoading test dataset for {dataset_mode}...")
        dataset_selector = Dataset_selector(**params)
        test_loader = dataset_selector.loader_test
        print(f"Test loader batches for {dataset_mode}: {len(test_loader)}")

        # تست مدل
        accuracy = test_model(model, test_loader, device, dataset_mode)
        results[dataset_mode] = accuracy
    except Exception as e:
        print(f"Error processing {dataset_mode}: {str(e)}")
        results[dataset_mode] = None

# چاپ نتایج نهایی
print("\nSummary of results:")
for dataset_mode, accuracy in results.items():
    if accuracy is not None:
        print(f"{dataset_mode}: {accuracy:.2f}%")
    else:
        print(f"{dataset_mode}: Failed to process")
