import os
import time
import random
import numpy as np
from datetime import timedelta
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from collections import Counter
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import f1_score
from torchvision.models import resnet18, ResNet18_Weights

# ----- 설정 -----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 디바이스: {device}")

BATCH_SIZE = 32
NUM_EPOCHS = 10
NUM_PARTS = 4
USE_PERCENT = 0.2

train_dir = r'C:\Users\sam43\Desktop\frames\train'
val_dir = r'C:\Users\sam43\Desktop\frames\validate'

# 체크포인트 설정
checkpoint_dir = './checkpoints'
checkpoint_filename = 'checkpoint2.pth'
checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
os.makedirs(checkpoint_dir, exist_ok=True)

# 자동 이어하기 여부 판단
NEW_RUN = not os.path.exists(checkpoint_path)
print(f"{'새로 시작합니다.' if NEW_RUN else '이전 학습을 이어갑니다.'}")

# ----- 전처리 -----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ----- 이미지 무결성 검사 -----
def check_image_loading(img_dir):
    for root, _, files in os.walk(img_dir):
        for file in tqdm(files, desc=f"검사 중 - {os.path.basename(root)}"):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img_path = os.path.join(root, file)
                try:
                    img = Image.open(img_path).convert('RGB')
                    img.close()
                except Exception as e:
                    print(f"이미지 로딩 실패: {img_path}, 오류: {e}")

check_done_flag = 'image_check_done.flag'
if not os.path.exists(check_done_flag):
    check_image_loading(train_dir)
    check_image_loading(val_dir)
    with open(check_done_flag, 'w') as f:
        f.write('done')

# ----- 데이터셋 -----
print("데이터셋 로딩 중...")
full_train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

total_len = len(full_train_dataset)
use_len = int(total_len * USE_PERCENT)

shuffle_file = 'shuffled_indices_20percent.npy'
if os.path.exists(shuffle_file):
    selected_indices = np.load(shuffle_file)
else:
    all_indices = list(range(total_len))
    random.Random(42).shuffle(all_indices)
    selected_indices = np.array(all_indices[:use_len])
    np.save(shuffle_file, selected_indices)

train_dataset = Subset(full_train_dataset, selected_indices)

# ----- 클래스 가중치 -----
print("클래스 가중치 계산 중...")
label_list = [full_train_dataset.targets[i] for i in selected_indices]
class_counts = Counter(label_list)
weights = [1.0 / class_counts[i] for i in range(len(full_train_dataset.classes))]
weights = torch.tensor(weights, dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)

# ----- 모델 설정 -----
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ----- 이어서 학습 -----
start_epoch = 0
start_part = 0
best_val_acc = 0.0
best_model_path = 'best_model2.pth'

if not NEW_RUN:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    start_part = checkpoint.get('part', 0) + 1
    if start_part >= NUM_PARTS:
        start_epoch += 1
        start_part = 0
    print(f"이어서 학습: Epoch {start_epoch+1}, Part {start_part+1}")

# ----- 데이터 분할 -----
def get_partial_loader(dataset, part_idx, num_parts=4, batch_size=32):
    total_len = len(dataset)
    part_size = total_len // num_parts
    start_idx = part_idx * part_size
    end_idx = total_len if part_idx == num_parts - 1 else (part_idx + 1) * part_size
    subset = Subset(dataset, list(range(start_idx, end_idx)))
    return DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=0)

# ----- 학습 루프 -----
for epoch in range(start_epoch, NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} 시작")
    epoch_start = time.time()

    part_range = range(start_part, NUM_PARTS) if epoch == start_epoch else range(NUM_PARTS)
    for part in part_range:
        print(f"Part {part+1}/{NUM_PARTS} 학습 중...")
        train_loader = get_partial_loader(train_dataset, part_idx=part, num_parts=NUM_PARTS, batch_size=BATCH_SIZE)

        model.train()
        total_loss, correct, total = 0, 0, 0
        all_preds, all_labels = [], []

        for imgs, labels in tqdm(train_loader, desc=f"[Epoch {epoch+1} Part {part+1}]"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        acc = correct / total
        f1 = f1_score(all_labels, all_preds, average='weighted')
        print(f"[Epoch {epoch+1} Part {part+1}] Loss: {total_loss/total:.4f}, Acc: {acc:.4f}, F1: {f1:.4f}")

        torch.save({
            'epoch': epoch,
            'part': part,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
        print(f"체크포인트 저장: {checkpoint_path}")

    # 검증
    model.eval()
    val_correct, val_total = 0, 0
    val_preds, val_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc=f"[Val] Epoch {epoch+1}"):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)

            val_correct += (preds == labels).sum().item()
            val_total += imgs.size(0)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_acc = val_correct / val_total
    val_f1 = f1_score(val_labels, val_preds, average='weighted')
    print(f"검증 결과: Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        print("Best 모델 저장됨!")
    else:
        print("성능 향상 없음. 모델 저장 생략.")

    print(f"Epoch {epoch+1} 소요 시간: {timedelta(seconds=int(time.time() - epoch_start))}")
    start_part = 0  # 다음 epoch은 처음 part부터 시작

print(f"\n전체 학습 완료! 최고 검증 정확도: {best_val_acc:.4f}")
