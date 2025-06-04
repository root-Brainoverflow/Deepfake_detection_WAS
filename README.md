# Deepfake Detection WAS (Flask 기반)

이 프로젝트는 이미지 및 영상에서 딥페이크 여부를 탐지하는 웹 애플리케이션입니다. 사용자는 웹 페이지를 통해 이미지를 업로드하거나 영상을 업로드하고, 각 프레임 단위로 분석된 결과를 확인할 수 있습니다. 딥러닝 기반 ResNet50 모델을 활용하며, Flask를 통해 웹 인터페이스를 제공합니다.

---

## 🔧 주요 기능

- 이미지 기반 딥페이크 탐지
- 영상(.mp4) 업로드 → 프레임 추출 후 프레임별 탐지
- 신뢰도(Confidence) 및 요약 결과 제공
- 조정된 Confidence 보정 적용 (로짓 차이 기반)

---

## 📁 디렉토리 구조

```
project/
├── app.py                  # Flask 웹 서버 실행 파일
├── model_utils.py          # 모델 로딩 및 예측 로직
├── best_ResNet50.pth       # 학습된 ResNet 모델 가중치 파일
├── ffmpeg.exe              # 프레임 추출에 사용할 ffmpeg 실행 파일
├── static/
│   ├── uploads/
│   │   ├── images/         # 업로드된 이미지 저장 경로
│   │   └── videos/         # 업로드된 비디오 저장 경로
│   └── frames/             # 비디오에서 추출된 프레임 저장 경로
└── templates/
    └── index.html          # 메인 웹페이지 템플릿
```

---

## ▶ 실행 방법

1. **의존성 설치**
    ```bash
    pip install flask torchvision pillow
    ```
    
2. **프로젝트 클론**

   아래 명령어로 GitHub에서 클론하세요.

   ```bash
   git clone https://github.com/username/deepfake-detection-was.git
   cd deepfake-detection-was
   ```

   ※ `ffmpeg.exe` 파일은 이미 포함되어 있어 따로 설치할 필요가 없습니다.

3. **서버 실행**

   ```bash
   python app.py
   ```

4. **브라우저 접속**

   ```
   http://127.0.0.1:5000
   ```

---

## 🧠 모델 구조

- ResNet50 (PyTorch torchvision)
- 출력층: 2 클래스 (`Real`, `Fake`)
- 로짓 차이 기반 Confidence 보정 적용:
    ```python
    adjusted_conf = torch.sigmoid(logit_margin / alpha).item()
    ```

---

## 📷 사용 예시

### 이미지 업로드
- JPEG, PNG, JPG 지원
- 탐지 결과 및 신뢰도 출력

### 영상 업로드
- mp4 영상 업로드
- 내부적으로 프레임 단위로 추출 (224x224 해상도, 2FPS)
- 각 프레임 분석 → 전체 평균 결과 요약 제공

---

## 📤 출력 예시

<img width="574" alt="Screenshot 2025-06-04 at 13 12 05" src="https://github.com/user-attachments/assets/de77e3a8-0516-4549-90e2-4a30f9003389" />


```text
──────────── 예측 상세 정보 ────────────
🔹 이미지 경로: static/uploads/images/Real.png
🔸 Temperature 값: 1.0
📤 원본 Logits: [[ 5.158088  -5.2176557]]
📈 Softmax 확률분포: [0.9999688, 0.0000311]
✅ 예측 클래스: 0 → Real
🔒 Softmax Confidence: 0.999969
⚖️ 조정된 Confidence (보정값): 0.999994
──────────────────────────────────────
```

---

## ❗ 주의사항

- 현재 가상환경 없이 실행하도록 구성되어 있습니다.
- 영상 프레임 추출은 `ffmpeg`를 필요로 하며, 경로에 `./ffmpeg` 실행파일이 있어야 합니다.
- 영상 탐지 정확도는 프레임 수 및 퀄리티에 따라 달라질 수 있으며, 평균 신뢰도 및 Fake 비율을 기준으로 판단합니다.

---
