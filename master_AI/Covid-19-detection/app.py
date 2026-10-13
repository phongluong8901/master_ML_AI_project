from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import cv2
import numpy as np
import os
import pickle
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

# 1. Khởi tạo ứng dụng web Flask
app = Flask(__name__)  # Tạo đối tượng chính 'app' cho trang web

# 2. Tải mô hình AI và nhãn dữ liệu từ ổ cứng
model = load_model('models/CNN_Covid19_Xray_Version.keras')  # Nạp mô hình mạng nơ-ron CNN đã train xong
le = pickle.load(open("models/Label_encoder.pkl", 'rb'))     # Nạp danh sách tên các nhãn bệnh (dạng list)

# 3. Cấu hình thư mục lưu trữ ảnh người dùng tải lên
UPLOAD_FOLDER = 'uploads'                  # Đặt tên thư mục chứa ảnh upload là 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER  # Gán vào cấu hình của Flask
os.makedirs(UPLOAD_FOLDER, exist_ok=True)    # Tự động tạo thư mục này nếu chưa có trên máy tính

# 4. Định nghĩa hàm xử lý ảnh và dự đoán kết quả bằng AI
def process_image(image_path):
    image = cv2.imread(image_path)           # Đọc file ảnh từ đường dẫn vừa lưu
    if image is None:
        raise ValueError(f"Image not found at path: {image_path}") # Báo lỗi nếu không tìm thấy ảnh
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)   # Chuyển hệ màu OpenCV (BGR) sang hệ màu chuẩn (RGB)
    image_resized = cv2.resize(image_rgb, (150, 150))    # Thu phóng ảnh về kích thước chuẩn (150x150 pixel) mà mô hình yêu cầu
    image_normalized = image_resized / 255.0             # Chuẩn hóa giá trị điểm ảnh về khoảng [0, 1] để AI dễ nhận diện
    image_input = np.expand_dims(image_normalized, axis=0) # Thêm chiều batch cho mảng ảnh (đầu vào của mô hình cần dạng [1, 150, 150, 3])
    
    predictions = model.predict(image_input)               # Đưa ảnh vào mô hình AI để dự đoán (trả về mảng xác suất)
    predicted_index = np.argmax(predictions)             # Tìm vị trí (index) có xác suất cao nhất trong mảng kết quả
    confidence_score = float(predictions[0][predicted_index]) # Lấy độ tự tin (xác suất phần trăm) của dự đoán đó
    
    predicted_label = le[predicted_index]                # Dùng vị trí index tra cứu tên nhãn bệnh tương ứng trong list `le`
    
    return predicted_label, confidence_score             # Trả về tên bệnh dự đoán và độ chính xác

# 5. Các đường dẫn (Routes / Endpoints) của trang web

# Trang chủ (hiển thị giao diện upload file)
@app.route('/')
def home():
    return render_template('index.html')                 # Trả về file giao diện index.html

# Đường dẫn để hiển thị lại ảnh đã tải lên từ thư mục uploads
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Đường dẫn nhận dữ liệu POST khi người dùng bấm nút Upload ảnh
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:                      # Kiểm tra xem có phần file gửi lên không
        return redirect(request.url)                     # Nếu không, load lại trang hiện tại

    file = request.files['file']
    if file.filename == '':                              # Kiểm tra xem người dùng đã chọn file chưa
        return redirect(request.url)                     # Nếu chưa chọn tên file, load lại trang

    if file:
        filename = secure_filename(file.filename)        # Làm sạch tên file để tránh lỗi bảo mật
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename) # Tạo đường dẫn đầy đủ lưu file
        file.save(file_path)                             # Lưu file ảnh vào thư mục uploads
        
        # Gọi hàm xử lý ảnh và dự đoán kết quả
        predicted_label, confidence_score = process_image(file_path)
        
        # Chuyển hướng sang trang hiển thị kết quả (result.html) và truyền dữ liệu sang đó
        return render_template('result.html',
                               image_path=file_path,
                               filename=filename,
                               predicted_label=predicted_label,
                               confidence_score=confidence_score)

# Trang mở camera chụp ảnh trực tiếp
@app.route('/camera')
def camera():
    return render_template('camera.html')                 # Trả về file camera.html

# 6. Khởi chạy ứng dụng Flask
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)              # Chạy web ở chế độ Debug, tắt watchdog reloader để tránh lỗi lặp log