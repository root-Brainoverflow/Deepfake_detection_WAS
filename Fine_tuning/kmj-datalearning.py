import os
import re
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import torch.optim as optim
from tqdm import tqdm

print("=== 프로그램이 시작되었습니다 ===")

# 1. 커스텀 데이터셋 클래스
class DeepfakeDataset(Dataset):
    def __init__(self, root_dir):
        self.image_paths = []
        self.labels = []

        for label, category in enumerate(['real', 'fake']):
            category_path = os.path.join(root_dir, category)
            for gender in ['남성', '여성']:
                gender_path = os.path.join(category_path, gender)
                if not os.path.exists(gender_path):
                    continue
                for filename in os.listdir(gender_path):
                    if filename.lower().endswith(('.jpg', '.png')):
                        full_path = os.path.join(gender_path, filename)
                        self.image_paths.append(full_path)
                        self.labels.append(label)

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
        except Exception:
            image = Image.new('RGB', (64, 64), (0, 0, 0))

        image = self.to_tensor(image)
        label = self.labels[idx]
        return image, label

# 2. 데이터셋 및 DataLoader 설정
train_dir = 'archive/frames/train'
val_dir = 'archive/frames/validate'

train_dataset = DeepfakeDataset(train_dir)
val_dataset = DeepfakeDataset(val_dir)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# 3. 모델, 장치, 손실함수, 옵티마이저 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 저장된 마지막 에폭 모델 불러오기
checkpoint_dir = '.'
latest_epoch = 0

for filename in os.listdir(checkpoint_dir):
    match = re.match(r'resnet18_epoch(\d+)\.pth', filename)
    if match:
        epoch_num = int(match.group(1))
        if epoch_num > latest_epoch:
            latest_epoch = epoch_num

if latest_epoch > 0:
    checkpoint_path = f'resnet18_epoch{latest_epoch}.pth'
    model.load_state_dict(torch.load(checkpoint_path))
    print(f"🔄 이전 학습 모델 로드 완료 (epoch {latest_epoch})")
else:
    print("🆕 새로 학습을 시작합니다")

# 최고 정확도 저장용 변수
best_val_acc = 0.0

# 5. 학습 루프
num_epochs = 10
print("===== 학습을 시작합니다 =====")

for epoch in range(latest_epoch, num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", ncols=100)
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        loop.set_postfix(loss=loss.item(), acc=100 * correct / total)

    train_acc = correct / total
    print(f"Epoch {epoch+1} 완료 - Loss: {total_loss:.4f}, Accuracy: {train_acc:.4f}")

    # 모델 저장
    save_path = f'resnet18_epoch{epoch+1}.pth'
    torch.save(model.state_dict(), save_path)
    print(f"💾 에폭 {epoch+1} 모델 저장 완료: {save_path}")

    # 검증
    print("🔍 검증 중...")
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total if val_total > 0 else 0
    print(f"📊 Validation Accuracy: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'resnet18_best.pth')
        print(f"🌟 새로운 최고 정확도 모델 저장됨 (acc: {val_acc:.4f})")

print("✅ 전체 학습 완료")
