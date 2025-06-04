import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import classification_report, confusion_matrix

print("🚀 평가 코드 시작됨")

# 1. 커스텀 데이터셋 정의
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

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        image = self.transform(image)
        label = self.labels[idx]
        return image, label

# 2. 데이터 로드
val_dir = 'archive/frames/validate'
val_dataset = DeepfakeDataset(val_dir)
print(f"🔍 Validation 이미지 개수: {len(val_dataset)}")
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# 3. 모델 불러오기
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('resnet18_best.pth', map_location=device))
model = model.to(device)
model.eval()
print("✅ 모델 로드 완료")

# 4. 평가 루프
correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

        correct += (predicted == labels).sum().item()
        total += labels.size(0)

# 5. 평가 지표 출력
val_acc = correct / total if total > 0 else 0
print(f"\n📊 Accuracy: {val_acc:.4f}")

# 추가 평가 지표
print("\n📋 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=['real', 'fake']))

print("🔢 Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))
