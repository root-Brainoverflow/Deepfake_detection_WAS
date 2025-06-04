import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import f1_score
from tqdm import tqdm

from torchvision.datasets import ImageFolder

class CleanImageFolder(ImageFolder):
    def find_classes(self, directory):
        # 기존 클래스 찾기
        classes, class_to_idx = super().find_classes(directory)
        # 숨김 폴더 등 제거
        classes = [c for c in classes if not c.startswith(".")]
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

def train():
    print("====== [0] PyTorch & 환경 체크 ======")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 사용 디바이스: {device}")

    BATCH_SIZE = 32
    NUM_EPOCHS = 10

    train_dir = "./frames/train"
    val_dir = "./frames/validate"

    print(f"train_dir: {train_dir} / val_dir: {val_dir}")
    print("train 폴더 fake 하위:", os.listdir(os.path.join(train_dir, "fake")) if os.path.exists(os.path.join(train_dir, "fake")) else "폴더 없음")
    print("train 폴더 real 하위:", os.listdir(os.path.join(train_dir, "real")) if os.path.exists(os.path.join(train_dir, "real")) else "폴더 없음")

    print("====== [1] 데이터 전처리/로드 ======")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    for root, dirs, files in os.walk(train_dir):
        print(root, "→", len([f for f in files if f.endswith(('.jpg', '.png'))]), "images")

    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)

    print(f"✔️ 전체 학습 데이터 수: {len(train_dataset)}개")
    print(f"✔️ 전체 검증 데이터 수: {len(val_dataset)}개")
    print(f"✔️ 클래스 목록 (폴더명 기준): {train_dataset.classes}")

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("❌ 데이터가 없습니다. 폴더 경로/이미지 파일 확인!")
        return

    # ---- 전체 train 데이터에서 20%만 랜덤으로 샘플링 ----
    use_len = int(len(train_dataset) * 0.2)
    ignore_len = len(train_dataset) - use_len
    subset_train, _ = random_split(train_dataset, [use_len, ignore_len])
    print(f"✔️ 사용할 학습 데이터 수(20%): {len(subset_train)}개")

    train_loader = DataLoader(subset_train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    print("====== [2] 클래스 불균형 가중치 계산 ======")
    # class_counts는 전체 데이터 기준! (20% 데이터로 재계산 원하면 아래 주석 참고)
    class_counts = torch.tensor([train_dataset.targets.count(i) for i in range(len(train_dataset.classes))], dtype=torch.float)
    print("클래스별 데이터 수:", dict(zip(train_dataset.classes, class_counts.tolist())))
    weights = 1. / class_counts
    print("가중치(클래스 불균형 반영):", dict(zip(train_dataset.classes, weights.tolist())))
    weights = weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    print("====== [3] 모델 및 Optimizer 준비 ======")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))  # 클래스 개수에 맞춤
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print(model)



    best_val_acc = 0.0
    for epoch in range(NUM_EPOCHS):
        print(f"\n\n=== [Epoch {epoch+1}/{NUM_EPOCHS}] 학습 시작 ===")
        model.train()
        total_loss, correct, total = 0, 0, 0
        all_preds, all_labels = [], []

        train_bar = tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}", ncols=80)
        for batch_idx, (imgs, labels) in enumerate(train_bar):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_loss = loss.item() * imgs.size(0)
            total_loss += batch_loss
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            if (batch_idx+1) % 10 == 0 or (batch_idx+1) == len(train_loader):
                train_bar.set_postfix({
                    'Loss': f"{total_loss / total:.4f}",
                    'Acc': f"{correct / total:.4f}"
                })

        train_acc = correct / total
        train_f1 = f1_score(all_labels, all_preds, average='weighted')
        print(f"✅ [Epoch {epoch+1}] Train Loss: {total_loss/total:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")

        # ---- Validation ----
        print(f"\n🔍 [Epoch {epoch+1}] 검증 단계")
        model.eval()
        val_correct, val_total = 0, 0
        val_preds, val_labels = [], []

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"[Val] Epoch {epoch+1}", ncols=80)
            for batch_idx, (imgs, labels) in enumerate(val_bar):
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += imgs.size(0)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                if (batch_idx+1) % 5 == 0 or (batch_idx+1) == len(val_loader):
                    val_bar.set_postfix({'Acc': f"{val_correct / val_total:.4f}"})

        val_acc = val_correct / val_total
        val_f1 = f1_score(val_labels, val_preds, average='weighted')
        print(f"📊 [Epoch {epoch+1}] Val Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"💾 [Epoch {epoch+1}] Best 모델 저장! (Val Acc: {val_acc:.4f})")
        else:
            print(f"📉 [Epoch {epoch+1}] 모델 저장 X (이전 최고 Val Acc: {best_val_acc:.4f})")

    print("\n====== 전체 학습 완료 ======")
    print(f"📌 최고 검증 정확도(Best Val Acc): {best_val_acc:.4f}")

if __name__ == "__main__":
    train() 