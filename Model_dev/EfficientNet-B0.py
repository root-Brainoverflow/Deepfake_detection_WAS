import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import torch.multiprocessing as mp
import random
import os
os.environ.pop("MALLOC_STACK_LOGGING", None)
os.environ.pop("MallocStackLogging", None)

# Mac M2 DataLoader 워커 이슈 방지
mp.set_start_method('fork', force=True)

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
# 2) 학습/검증 함수
# ───────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
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

    print(f"[Epoch {epoch}] Train Loss: {total_loss/total:.4f}, Acc: {correct/total:.4f}")

def validate(model, loader, criterion, device, epoch):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(device)
            labels = labels.unsqueeze(1).to(device)

            outputs = model(imgs)
            loss    = criterion(outputs, labels)

            total_loss += loss.item() * imgs.size(0)
            preds       = (outputs > 0.5).long()
            correct    += (preds == labels.long()).sum().item()
            total      += imgs.size(0)

    acc = correct/total
    print(f"[Epoch {epoch}] Val   Loss: {total_loss/total:.4f}, Acc: {acc:.4f}")
    return acc

# ───────────────────────────────────────────────────
# 3) 메인 스크립트
# ───────────────────────────────────────────────────
if __name__ == "__main__":
    # 3-0) 디바이스 선택
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    # 3-1) Transform 정의 (해상도 128로 축소)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(128),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(144),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    # 3-2) Dataset + 샘플링: 전체의 20%만 사용
    full_train = DeepfakeDataset("frames/train", transform=train_transform)
    full_val   = DeepfakeDataset("frames/validate", transform=val_transform)

    train_idx = list(range(len(full_train)))
    val_idx   = list(range(len(full_val)))
    random.shuffle(train_idx)
    random.shuffle(val_idx)

    train_sampler = SubsetRandomSampler(train_idx[:int(0.2*len(train_idx))])
    val_sampler   = SubsetRandomSampler(val_idx[:int(0.2*len(val_idx))])

    train_loader = DataLoader(full_train, batch_size=32,
                              sampler=train_sampler, num_workers=4)
    val_loader   = DataLoader(full_val,   batch_size=32,
                              sampler=val_sampler,   num_workers=4)

    # 3-3) 모델 준비: EfficientNet-B0 (pretrained, classifier 수정)
    model = models.efficientnet_b0(pretrained=True)
    in_feats = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2, inplace=True),
        nn.Linear(in_feats, 1),
        nn.Sigmoid()
    )
    model = model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 3-4) 학습 루프: 에폭 5
    best_acc   = 0.0
    num_epochs = 5
    for epoch in range(1, num_epochs+1):
        print(f"\n=== Epoch {epoch} / {num_epochs} ===")
        train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_acc = validate(model, val_loader, criterion, device, epoch)

        # best 모델 저장
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_efficientnetb0_deepfake.pt")
            print(f">>> New best model saved (Acc: {best_acc:.4f})")

    print(f"\n== Training complete (Best Acc: {best_acc:.4f}) ==")