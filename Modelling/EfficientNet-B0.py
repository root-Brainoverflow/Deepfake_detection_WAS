import time, random, os
from pathlib import Path
from PIL import Image

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms, models
from tqdm import tqdm

# ─── Dataset ────────────────────────────────────────────────────────────────
class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        for cls in ["real","fake"]:
            label = 0 if cls=="real" else 1
            for gender_dir in (Path(root_dir)/cls).iterdir():
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

# ─── Train / Validate ───────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = correct = total = 0
    pbar = tqdm(loader, desc=f"[{epoch}] Train", unit="batch")
    for imgs, labels in pbar:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.unsqueeze(1).to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds      = (outputs>0.5).long()
        correct   += (preds==labels.long()).sum().item()
        total     += imgs.size(0)
        pbar.set_postfix(acc=correct/total, loss=total_loss/total)

def validate(model, loader, criterion, device, epoch):
    model.eval()
    total_loss = correct = total = 0
    pbar = tqdm(loader, desc=f"[{epoch}]  Val ", unit="batch")
    with torch.no_grad():
        for imgs, labels in pbar:
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.unsqueeze(1).to(device, non_blocking=True)
            outputs = model(imgs)
            loss    = criterion(outputs, labels)

            total_loss += loss.item() * imgs.size(0)
            preds      = (outputs>0.5).long()
            correct   += (preds==labels.long()).sum().item()
            total     += imgs.size(0)
            pbar.set_postfix(acc=correct/total, loss=total_loss/total)
    return correct/total

# ─── Main ───────────────────────────────────────────────────────────────────
if __name__=="__main__":
    # 1) Device (무조건 MPS)
    device = torch.device("mps")
    print("Using device:", device)

    # 2) Transforms
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(128),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(144),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    # 3) Dataset + DataLoader
    train_ds = DeepfakeDataset("frames/train", transform=train_tf)
    val_ds   = DeepfakeDataset("frames/validate", transform=val_tf)

    train_loader = DataLoader(
        train_ds, batch_size=64, shuffle=True,
        num_workers=4, pin_memory=False,
        persistent_workers=True, prefetch_factor=2
    )
    # validation은 20%만 샘플링
    idx = list(range(len(val_ds)))
    random.shuffle(idx)
    sampler = SubsetRandomSampler(idx[:int(0.2*len(idx))])
    val_loader = DataLoader(
        val_ds, batch_size=64, sampler=sampler,
        num_workers=4, pin_memory=False,
        persistent_workers=True, prefetch_factor=2
    )

    # 4) Model, Loss, Optimizer
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_feats = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2, inplace=True),
        nn.Linear(in_feats,1),
        nn.Sigmoid()
    )
    # ── **모델 전체를** MPS로 옮깁니다 ────────────────────────────────
    model = model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 5) Training Loop
    best_acc = 0.0
    epochs   = 15
    for ep in range(1, epochs+1):
        print(f"\n=== Epoch {ep}/{epochs} ===")
        t0 = time.time()
        train_one_epoch(model, train_loader, criterion, optimizer, device, ep)
        print(f"→ train: {time.time()-t0:.1f}s", end="  ")

        v0 = time.time()
        acc = validate(model, val_loader, criterion, device, ep)
        print(f"→ val:   {time.time()-v0:.1f}s, Acc: {acc:.4f}")

        if acc>best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_efficientnetb0.pt")
            print(f">>> New best model: {best_acc:.4f}")

    print(f"\n== Training complete (Best Acc: {best_acc:.4f}) ==")