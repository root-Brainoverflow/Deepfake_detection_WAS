# model_utils.py

# import torch
# from torchvision import transforms
# from torchvision.models import resnet18
# import torch.nn as nn
# from PIL import Image

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5]*3, [0.5]*3)
# ])

# def load_model(model_path='best_model2.pth'):
#     model = resnet18(weights=None)
#     model.fc = nn.Linear(model.fc.in_features, 2)
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.to(device)
#     model.eval()
#     return model

# def predict_image(image_path, model):
#     img = Image.open(image_path).convert('RGB')
#     img_tensor = transform(img).unsqueeze(0).to(device)
#     with torch.no_grad():
#         output = model(img_tensor)
#         pred = torch.argmax(output, dim=1).item()
#     return 'Real' if pred == 0 else 'Fake'
# model_utils.py

import torch
from torchvision import transforms
from torchvision.models import resnet18
import torch.nn as nn
from PIL import Image
import os
import subprocess

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 이미지 전처리 방식 (학습 시와 동일해야 함)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def load_model(model_path='best_ResNet50.pt'):
    # ResNet18 구조를 불러오고 출력층 수정
    model = resnet18(weights=None)  # pretrained=False 와 같음
    model.fc = nn.Linear(model.fc.in_features, 2)  # 클래스 수: 2 (Real, Fake)
    
    # 저장된 state_dict 로드
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_image(image_path, model, temperature=1.0, alpha=6.0):
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_tensor)
        scaled_logits = logits / temperature
        probs = torch.softmax(scaled_logits, dim=1)[0]
        pred_class = torch.argmax(probs).item()
        confidence = probs[pred_class].item()

        # 로짓 차이 기반 보정 신뢰도 계산
        logit_margin = abs(logits[0, 0] - logits[0, 1])
        adjusted_conf = torch.sigmoid(logit_margin / alpha).item()

    # ✅ 클래스 매핑 수정 (0 → Real, 1 → Fake)
    result_label = 'Real' if pred_class == 0 else 'Fake'

    print("\n──────────── 예측 상세 정보 ────────────")
    print(f"🔹 이미지 경로: {image_path}")
    print(f"🔸 Temperature 값: {temperature}")
    print(f"📤 원본 Logits: {logits.cpu().numpy()}")
    print(f"🌡️  Scaled Logits (logits / T): {scaled_logits.cpu().numpy()}")
    print(f"📈 Softmax 확률분포: {probs.cpu().numpy()}")
    print(f"✅ 예측 클래스: {pred_class} → {result_label}")
    print(f"🔒 Softmax Confidence: {confidence:.6f}")
    print(f"⚖️ 조정된 Confidence (보정값): {adjusted_conf:.6f}")
    print("──────────────────────────────────────\n")

    return result_label, adjusted_conf

#영상 업로드 및 이미지 추출
def extract_frames_with_ffmpeg(video_path, output_folder, fps=1):
    
    os.makedirs(output_folder, exist_ok=True)
    
    # 예: frames/frame_%04d.jpg 형태로 저장
    output_pattern = os.path.join(output_folder, "frame_%04d.jpg")
    
    command = [
                './ffmpeg',
                '-i', video_path,
                '-r', '0.5',
                '-s', '224x224',
                '-qscale:v', '5',
                output_pattern
            ]
    
    subprocess.run(command, check=True)


#추출한 이미지 딥페이크 탐지
def predict_frames_in_folder(folder_path, model):
    results = []
    filenames = sorted(os.listdir(folder_path))

    for filename in filenames:
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            label, confidence = predict_image(image_path, model)
            
            # 🔍 디버깅 로그 추가
            print(f"[프레임 예측] {filename}: {label} ({confidence:.4f})")

            results.append({
                "filename": filename,
                "label": label,
                "confidence": confidence
            })

    representative_image = filenames[0] if filenames else None
    return results, representative_image

def summarize_frame_results(frame_results):
    total = len(frame_results)
    if total == 0:
        return {
            "total_frames": 0,
            "fake_count": 0,
            "real_count": 0,
            "fake_ratio": 0.0,
            "real_ratio": 0.0,
            "average_confidence": 0.0,
            "judgement": "No frames to analyze"
        }

    fake_count = sum(1 for r in frame_results if r['label'] == 'Fake')
    real_count = total - fake_count
    avg_confidence = sum(r['confidence'] for r in frame_results) / total

    fake_ratio = fake_count / total
    judgement = "딥페이크일 가능성 높음" if fake_ratio >= 0.5 else "실제 영상일 가능성 높음"

    return {
        "total_frames": total,
        "fake_count": fake_count,
        "real_count": real_count,
        "fake_ratio": fake_ratio,
        "real_ratio": real_count / total,
        "average_confidence": avg_confidence,
        "judgement": judgement
    }

