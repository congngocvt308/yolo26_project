import os
import cv2
import numpy as np
import onnxruntime as ort

def process_video_onnx(video_path, model_path, output_path, conf_threshold=0.25):
    print("🔄 Bước 1: Đang nạp mô hình YOLO26 dưới dạng ONNX...")
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    
    # Định nghĩa danh sách tên các class (Cần khớp với file data.yaml của bạn)
    # Ví dụ danh sách class phổ biến của bạn:
    class_names = ['bed', 'book', 'bottle', 'bowl', 'chair', 'charger', 'cup', 'door', 'handbag', 'knife', 
                   'laptop', 'mouse', 'phone', 'plug', 'sofa', 'table'] 
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Lỗi: Không thể tìm thấy hoặc mở file video: {video_path}")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"🚀 Bước 2: Bắt đầu quét và VẼ NHÃN lên video thực tế ({width}x{height})...")
    frame_id = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_id += 1
        
        # ─── TIỀN XỬ LÝ ẢNH (PREPROCESSING) ───
        img = cv2.resize(frame, (640, 640))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        
        # ─── CHẠY INFERENCE ONNX ───
        outputs = session.run(None, {input_name: img})
        
        # Đầu ra YOLO26 NMS-free thường có dạng: [1, số_box, 6]
        predictions = np.squeeze(outputs[0]) 
        
        # ─── HÀM HẬU XỬ LÝ & VẼ KHUNG ĐỘC LẬP ───
        for pred in predictions:
            # Tách các giá trị: Tọa độ box (đang ở quy chuẩn ảnh 640x640), độ tự tin, và ID lớp
            x1, y1, x2, y2, score, class_id = pred
            
            # Chỉ lọc lấy những vật thể có độ tin cậy cao hơn ngưỡng conf_threshold
            if score >= conf_threshold:
                # Ép tỷ lệ tọa độ từ ảnh vuông 640x640 ngược về kích thước thật của Video gốc
                x1 = int(x1 * width / 640)
                y1 = int(y1 * height / 640)
                x2 = int(x2 * width / 640)
                y2 = int(y2 * height / 640)
                
                class_id = int(class_id)
                label_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
                caption = f"{label_name} {score:.2f}"
                
                # Tiến hành vẽ Bounding Box màu xanh lá lên Khung hình gốc
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Vẽ nền chữ nhãn vật thể
                cv2.putText(frame, caption, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Ghi khung hình ĐÃ ĐƯỢC VẼ NHÃN vào video đầu ra
        out.write(frame) 
        
    cap.release()
    out.release()
    print(f"🎉 Hoàn thành! Video đã có nhãn lưu tại: {output_path}")

if __name__ == "__main__":
    VIDEO_IN = os.getenv("VIDEO_IN", "video_input.mp4")
    VIDEO_OUT = os.getenv("VIDEO_OUT", "video_output.mp4")
    MODEL_PATH = "models/best.onnx"
    
    process_video_onnx(VIDEO_IN, MODEL_PATH, VIDEO_OUT, conf_threshold=0.25)