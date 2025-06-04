import time
import random
import os
from pathlib import Path
from PIL import Image

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ─── Dataset 정의 ────────────────────────────────────────────────────────────────
class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        for cls in ["real", "fake"]:
            fake_label = 0 if cls == "real" else 1
            cls_path = Path(root_dir) / cls
            for gender_dir in cls_path.iterdir():
                if not gender_dir.is_dir():
                    continue
                # gender_dir.name 은 '남성' 또는 '여성'
                gender_label = 0 if "남성" in gender_dir.name else 1
                for img_path in gender_dir.glob("*.jpg"):
                    self.samples.append((img_path, fake_label, gender_label))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, fake_label, gender_label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return (
            img,
            torch.tensor(fake_label, dtype=torch.float32),
            torch.tensor(gender_label, dtype=torch.float32),
        )

# ─── 멀티태스크 모델 정의 ─────────────────────────────────────────────────────────
class MultiTaskNet(nn.Module):
    def __init__(self, backbone_name="efficientnet"):
        super().__init__()
        if backbone_name == "efficientnet":
            self.backbone = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
            )
            in_feats = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            self.backbone = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V2
            )
            in_feats = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        # fake vs real head
        self.head_fake = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_feats, 1),
            nn.Sigmoid()
        )
        # gender head
        self.head_gender = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_feats, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.backbone(x)
        out_fake   = self.head_fake(feat)
        out_gender = self.head_gender(feat)
        return out_fake, out_gender

# ─── Train 함수 ────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device, epoch, name):
    model.train()
    running_loss = 0.0
    correct_fake = correct_gender = total = 0
    pbar = tqdm(
        loader,
        desc=f"[{name}][{epoch}] Train",
        unit="batch",
        ncols=120
    )
    for i, (imgs, y_fake, y_gender) in enumerate(pbar, 1):
        imgs      = imgs.to(device, non_blocking=True)
        y_fake    = y_fake.to(device, non_blocking=True).unsqueeze(1)
        y_gender  = y_gender.to(device, non_blocking=True).unsqueeze(1)

        optimizer.zero_grad()
        pred_fake, pred_gender = model(imgs)
        loss_fake   = criterion(pred_fake,   y_fake)
        loss_gender = criterion(pred_gender, y_gender)
        loss = loss_fake + loss_gender
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds_f = (pred_fake > 0.5).long()
        preds_g = (pred_gender > 0.5).long()
        correct_fake   += (preds_f == y_fake.long()).sum().item()
        correct_gender += (preds_g == y_gender.long()).sum().item()
        total += imgs.size(0)

        avg_loss = running_loss / i
        acc_f    = correct_fake   / total
        acc_g    = correct_gender / total
        pbar.set_postfix(
            loss=f"{avg_loss:.4f}",
            fake_acc=f"{acc_f:.4f}",
            gen_acc=f"{acc_g:.4f}"
        )

# ─── Validation 함수 ─────────────────────────────────────────────────────────
def validate(model, loader, criterion, device, epoch, name):
    model.eval()
    running_loss = 0.0
    correct_f = correct_g = total = 0
    yf_true, yf_pred = [], []
    yg_true, yg_pred = [], []
    pbar = tqdm(
        loader,
        desc=f"[{name}][{epoch}]  Val ",
        unit="batch",
        ncols=120
    )
    with torch.no_grad():
        for i, (imgs, y_fake, y_gender) in enumerate(pbar, 1):
            imgs      = imgs.to(device, non_blocking=True)
            y_fake    = y_fake.to(device, non_blocking=True).unsqueeze(1)
            y_gender  = y_gender.to(device, non_blocking=True).unsqueeze(1)

            pf, pg = model(imgs)
            loss = criterion(pf, y_fake) + criterion(pg, y_gender)
            running_loss += loss.item()

            pr_f = (pf > 0.5).long()
            pr_g = (pg > 0.5).long()
            correct_f += (pr_f == y_fake.long()).sum().item()
            correct_g += (pr_g == y_gender.long()).sum().item()
            total += imgs.size(0)

            yf_true.extend(y_fake.cpu().squeeze(1).long().tolist())
            yf_pred.extend(pr_f.cpu().squeeze(1).tolist())
            yg_true.extend(y_gender.cpu().squeeze(1).long().tolist())
            yg_pred.extend(pr_g.cpu().squeeze(1).tolist())

            avg_loss = running_loss / i
            acc_f    = correct_f / total
            acc_g    = correct_g / total
            pbar.set_postfix(
                loss=f"{avg_loss:.4f}",
                fake_acc=f"{acc_f:.4f}",
                gen_acc=f"{acc_g:.4f}"
            )

    # Deepfake 태스크 지표
    f_acc  = accuracy_score(yf_true, yf_pred)
    f_prec = precision_score(yf_true, yf_pred, zero_division=0)
    f_rec  = recall_score(yf_true, yf_pred, zero_division=0)
    f_f1   = f1_score(yf_true, yf_pred, zero_division=0)
    # Gender 태스크 지표
    g_acc  = accuracy_score(yg_true, yg_pred)
    g_prec = precision_score(yg_true, yg_pred, zero_division=0)
    g_rec  = recall_score(yg_true, yg_pred, zero_division=0)
    g_f1   = f1_score(yg_true, yg_pred, zero_division=0)

    print(
        f"→ Epoch {epoch} End — "
        f"[Fake]  Acc:{f_acc:.4f} Prec:{f_prec:.4f} Rec:{f_rec:.4f} F1:{f_f1:.4f}  |  "
        f"[Gender] Acc:{g_acc:.4f} Prec:{g_prec:.4f} Rec:{g_rec:.4f} F1:{g_f1:.4f}"
    )
    return f_acc

# ─── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1) Device (MPS 전용)
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

    # 3) Dataset & DataLoader
    train_ds = DeepfakeDataset("frames/train",   transform=train_tf)
    val_ds   = DeepfakeDataset("frames/validate",transform=val_tf)
    train_loader = DataLoader(
        train_ds, batch_size=64, shuffle=True,
        num_workers=4, pin_memory=False,
        persistent_workers=True, prefetch_factor=2
    )
    val_loader   = DataLoader(
        val_ds, batch_size=64, shuffle=False,
        num_workers=4, pin_memory=False,
        persistent_workers=True, prefetch_factor=2
    )

    # 4) 모델 정의
    model_defs = {
        "EfficientNet": MultiTaskNet("efficientnet"),
        "ResNet50":     MultiTaskNet("resnet"),
    }
    criterion = nn.BCELoss()
    trained_models = {}

    # 5) 각 모델별 파인튜닝
    for name, model in model_defs.items():
        print(f"\n=== Training {name} ===")
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        best_acc = 0.0
        epochs = 10

        for ep in range(1, epochs+1):
            print(f"\n--- [{name}] Epoch {ep}/{epochs} ---")
            train_one_epoch(model, train_loader, criterion, optimizer, device, ep, name)
            f_acc = validate(model, val_loader, criterion, device, ep, name)
            if f_acc > best_acc:
                best_acc = f_acc
                torch.save(model.state_dict(), f"best_{name}.pt")
                print(f">>> New best for {name}: {best_acc:.4f}")

        trained_models[name] = model

    # 6) 앙상블 평가 (Fake 태스크만)
    print("\n=== Ensemble Fake Validation ===")
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, yf, yg in tqdm(val_loader, desc="[Ensemble Val]"):
            imgs = imgs.to(device)
            probs = []
            for m in trained_models.values():
                pf, _ = m(imgs)
                probs.append(pf.cpu().squeeze(1))
            avg_prob = torch.stack(probs, dim=0).mean(dim=0)
            y_true.extend(yf.tolist())
            y_pred.extend((avg_prob > 0.5).int().tolist())

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    print(f"Ensemble → Acc:{acc:.4f} Prec:{prec:.4f} Rec:{rec:.4f} F1:{f1:.4f}")