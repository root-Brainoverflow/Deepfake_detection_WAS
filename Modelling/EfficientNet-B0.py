import os
import time
import random
import torch
import torch.multiprocessing as mp
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import EfficientNet_B0_Weights
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# ───────────────────────────────────────────────────
# Mac M2 DataLoader 워커 이슈 방지
# ───────────────────────────────────────────────────
mp.set_start_method('fork', force=True)
os.environ.pop("MALLOC_STACK_LOGGING", None)
os.environ.pop("MallocStackLogging", None)

# ───────────────────────────────────────────────────
# 1) Dataset 정의 (폴더 스캔 + transform)
# ───────────────────────────────────────────────────
class DeepfakeDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.samples = []
        root = Path(root_dir)
        for cls in ["real", "fake"]:
            label = 0 if cls == "real" else 1
            for gender_dir in (root/cls).iterdir():
                if not gender_dir.is_dir(): continue
                for img_path in gender_dir.glob("*.jpg"):
                    self.samples.append((img_path, label))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)

# ───────────────────────────────────────────────────
# 2) Training & Validation 루프
# ───────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=f"Epoch {epoch} ▶ Train", unit="batch", leave=False)
    for imgs, labels in pbar:
        imgs   = imgs.to(device)
        labels = labels.unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds       = (outputs > 0.5).long()
        correct    += (preds == labels.long()).sum().item()
        total      += imgs.size(0)

        pbar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{correct/total:.4f}")

def validate(model, loader, criterion, device, epoch):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=f"Epoch {epoch} ▶ Valid", unit="batch", leave=False)
    with torch.no_grad():
        for imgs, labels in pbar:
            imgs   = imgs.to(device)
            labels = labels.unsqueeze(1).to(device)

            outputs = model(imgs)
            loss    = criterion(outputs, labels)

            total_loss += loss.item() * imgs.size(0)
            preds       = (outputs > 0.5).long()
            correct    += (preds == labels.long()).sum().item()
            total      += imgs.size(0)

            pbar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{correct/total:.4f}")

    return correct / total

# ───────────────────────────────────────────────────
# 3) Main
# ───────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("mps")
    print("Using device:", device)

    # 3-1) Transforms
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(128),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize(144),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])

    # 3-2) Dataset
    full_train = DeepfakeDataset("frames/train",    transform=train_tf)
    full_val   = DeepfakeDataset("frames/validate", transform=val_tf)

    # 3-3) DataLoader (전체 데이터)
    train_loader = DataLoader(
        full_train,
        batch_size=64,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    val_loader = DataLoader(
        full_val,
        batch_size=64,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    # 3-4) 모델 준비 (weights=API 사용)
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = models.efficientnet_b0(weights=weights)
    in_feats = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2, inplace=True),
        nn.Linear(in_feats, 1),
        nn.Sigmoid()
    )
    model = model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 3-5) 학습 루프
    best_acc = 0.0
    epochs   = 5
    for ep in range(1, epochs+1):
        print(f"\n=== Epoch {ep}/{epochs} ===")
        t0 = time.time()
        train_one_epoch(model, train_loader, criterion, optimizer, device, ep)
        print(f"→ train time: {time.time()-t0:.1f}s")

        v0  = time.time()
        acc = validate(model, val_loader, criterion, device, ep)
        print(f"→ valid time: {time.time()-v0:.1f}s, Acc: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_efficientnetb0_deepfake.pt")
            print(f">>> New best model saved (Acc: {best_acc:.4f})")

    print(f"\nTraining complete (Best Acc: {best_acc:.4f})")