import os
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F

# 클래스 정보 (학습 당시와 반드시 동일!)
num_classes = 2
class_names = ['fake', 'real']

# 1. 세 개 모델 불러오기
def load_model(weight_path):
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    state_dict = torch.load(weight_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model

model_paths = [
    '모델/best_model1.pth',
    '모델/best_model2.pth',
    '모델/best_model3.pth',
]
models_list = [load_model(p) for p in model_paths]

# 2. 전처리 정의 (학습 때와 똑같이!)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# 3. 테스트할 폴더 내 모든 이미지 앙상블 예측
test_folder = '판별사진'  # 파일 현위치 기준 상대주소
results = []

for fname in os.listdir(test_folder):
    if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
        img_path = os.path.join(test_folder, fname)
        img = Image.open(img_path).convert('RGB')
        input_tensor = transform(img).unsqueeze(0)  # 배치 차원

        # 각 모델별 예측 확률 계산
        probs = []
        for model in models_list:
            with torch.no_grad():
                output = model(input_tensor)
                prob = F.softmax(output, dim=1)
                probs.append(prob.cpu().numpy())

        # 앙상블(softmax 확률 평균)
        avg_prob = sum(probs) / len(probs)
        pred_idx = avg_prob.argmax(axis=1)[0]
        pred_label = class_names[pred_idx]
        results.append((fname, pred_label, float(avg_prob[0][0]), float(avg_prob[0][1])))
        print(f"{fname} → 예측: {pred_label} (fake확률: {avg_prob[0][0]:.3f}, real확률: {avg_prob[0][1]:.3f})")
