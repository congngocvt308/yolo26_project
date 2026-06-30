# Bước 1: Sử dụng base image Python phiên bản rút gọn (slim) để giảm dung lượng
FROM python:3.12-slim

# Bước 2: Thiết lập thư mục làm việc mặc định bên trong môi trường ảo
WORKDIR /app

# Bước 3: Cài đặt các gói thư viện hệ thống tối thiểu cần cho OpenCV chạy ngầm
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Bước 4: Sao chép file khai báo thư viện vào container trước
COPY requirements.txt .

# Bước 5: Tiến hành cài đặt các thư viện Python và xóa cache ngay lập tức để tiết kiệm bộ nhớ
RUN pip install --no-cache-dir -r requirements.txt

# Bước 6: Sao chép toàn bộ mã nguồn và file mô hình ONNX từ máy tính vào container
COPY . .

# Bước 7: Lệnh mặc định kích hoạt file app.py chạy khi container khởi động
CMD ["python", "app.py"]