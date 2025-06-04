# from flask import Flask

# app = Flask(__name__)

# @app.route('/')
# def home():
#     return "Flask 서버 실행 성공!"

# if __name__ == '__main__':
#     app.run(debug=True)


# app.py

from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from model_utils import load_model, predict_image, extract_frames_with_ffmpeg, predict_frames_in_folder, summarize_frame_results




import sys
import functools
print = functools.partial(print, flush=True)






IMAGE_UPLOAD_FOLDER = os.path.join('static', 'uploads', 'images')
VIDEO_UPLOAD_FOLDER = os.path.join('static', 'uploads', 'videos')
FRAME_OUTPUT_FOLDER = os.path.join('static', 'frames')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'jfif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = IMAGE_UPLOAD_FOLDER

# 모델 로딩
model = load_model('best_model2.pth')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

#이미지 딥페이크 탐지
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(request.url)

    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        return redirect(request.url)

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    result, confidence = predict_image(filepath, model)
    return render_template('index.html', result=result, confidence=confidence, filename=filename)


#비디오 영상 업로드 파트
@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return redirect(request.url)

    video = request.files['video']
    path = os.path.join(VIDEO_UPLOAD_FOLDER, video.filename)
    video.save(path)

    # 프레임 추출
    extract_frames_with_ffmpeg(path, FRAME_OUTPUT_FOLDER, fps=1)

    # 프레임별 예측 및 요약
    frame_results, representative_image = predict_frames_in_folder(FRAME_OUTPUT_FOLDER, model)
    summary = summarize_frame_results(frame_results)

    return render_template('index.html', frame_results=frame_results, summary=summary, representative_image=representative_image)






if __name__ == '__main__':
    os.makedirs(IMAGE_UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
