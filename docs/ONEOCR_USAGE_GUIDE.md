# Hướng Dẫn Sử Dụng OneOCR Optimized

## Tổng Quan

Module `oneocr_optimized.py` cung cấp engine OCR hiệu năng cao sử dụng OneOCR DLL. Module hỗ trợ xử lý ảnh từ PIL Image và OpenCV, với khả năng thread-safe và quản lý tài nguyên tự động.

## Cài Đặt

### Yêu Cầu Hệ Thống
- Windows OS (sử dụng WinDLL)
- Python 3.7+
- OneOCR DLL và model file

### Thư Viện Cần Thiết
```python
pip install Pillow
pip install opencv-python  # Tùy chọn, nếu sử dụng recognize_cv2
pip install fastapi uvicorn  # Tùy chọn, nếu chạy web service
```

### Cấu Trúc Thư Mục
```
~/.config/oneocr/
├── oneocr.dll
└── oneocr.onemodel
```

## Sử Dụng Cơ Bản

### 1. Khởi Tạo OCR Engine

```python
from core.oneocr_optimized import OcrEngine

# Cách 1: Khởi tạo thông thường
ocr = OcrEngine()

# Sử dụng OCR
result = ocr.recognize_pil(image)

# Dọn dẹp tài nguyên khi hoàn thành
ocr.cleanup()
```

```python
# Cách 2: Sử dụng context manager (Khuyến nghị)
with OcrEngine() as ocr:
    result = ocr.recognize_pil(image)
    # Tự động cleanup khi thoát khỏi context
```

### 2. Nhận Dạng Văn Bản Từ PIL Image

```python
from PIL import Image
from core.oneocr_optimized import OcrEngine

# Đọc ảnh
image = Image.open("path/to/image.png")

# Khởi tạo OCR engine
with OcrEngine() as ocr:
    # Thực hiện OCR
    result = ocr.recognize_pil(image)
    
    # Lấy văn bản nhận dạng được
    text = result["text"]
    print(f"Văn bản: {text}")
    
    # Lấy góc nghiêng của văn bản
    angle = result["text_angle"]
    print(f"Góc nghiêng: {angle}°")
    
    # Lấy thông tin chi tiết từng dòng
    for line in result["lines"]:
        print(f"Dòng: {line['text']}")
        print(f"Tọa độ: {line['bounding_rect']}")
```

### 3. Nhận Dạng Văn Bản Từ OpenCV

```python
import cv2
import numpy as np
from core.oneocr_optimized import OcrEngine

# Đọc ảnh bằng OpenCV
image_buffer = np.fromfile("path/to/image.png", dtype=np.uint8)

# Khởi tạo OCR engine
with OcrEngine() as ocr:
    # Thực hiện OCR
    result = ocr.recognize_cv2(image_buffer)
    
    # Xử lý kết quả
    if "error" in result:
        print(f"Lỗi: {result['error']}")
    else:
        print(f"Văn bản: {result['text']}")
```

## Cấu Trúc Kết Quả

### Định Dạng Kết Quả OCR

```python
{
    "text": "Văn bản đầy đủ\nCác dòng cách nhau bởi \\n",
    "text_angle": 0.5,  # Góc nghiêng (độ), hoặc None
    "lines": [
        {
            "text": "Dòng văn bản 1",
            "bounding_rect": {
                "x1": 10.5, "y1": 20.3,
                "x2": 100.2, "y2": 20.5,
                "x3": 100.0, "y3": 40.8,
                "x4": 10.3, "y4": 40.5
            },
            "words": [
                {
                    "text": "Dòng",
                    "bounding_rect": {...},
                    "confidence": 0.98
                },
                {
                    "text": "văn",
                    "bounding_rect": {...},
                    "confidence": 0.95
                }
            ]
        }
    ]
}
```

### Kết Quả Lỗi

```python
{
    "text": "",
    "text_angle": None,
    "lines": [],
    "error": "Thông báo lỗi"
}
```

## Ví Dụ Nâng Cao

### 1. Xử Lý Batch Nhiều Ảnh

```python
from PIL import Image
from core.oneocr_optimized import OcrEngine
import os

def process_images_batch(image_paths):
    """Xử lý nhiều ảnh cùng lúc"""
    results = []
    
    with OcrEngine() as ocr:
        for path in image_paths:
            try:
                image = Image.open(path)
                result = ocr.recognize_pil(image)
                results.append({
                    "path": path,
                    "result": result
                })
            except Exception as e:
                results.append({
                    "path": path,
                    "error": str(e)
                })
    
    return results

# Sử dụng
image_files = ["image1.png", "image2.png", "image3.png"]
results = process_images_batch(image_files)
```

### 2. Lọc Kết Quả Theo Độ Tin Cậy

```python
def filter_by_confidence(result, min_confidence=0.8):
    """Lọc các từ có độ tin cậy thấp"""
    filtered_lines = []
    
    for line in result["lines"]:
        filtered_words = [
            word for word in line["words"]
            if word["confidence"] and word["confidence"] >= min_confidence
        ]
        
        if filtered_words:
            filtered_text = " ".join(word["text"] for word in filtered_words)
            filtered_lines.append({
                "text": filtered_text,
                "words": filtered_words
            })
    
    return filtered_lines

# Sử dụng
with OcrEngine() as ocr:
    result = ocr.recognize_pil(image)
    high_confidence_lines = filter_by_confidence(result, min_confidence=0.9)
```

### 3. Trích Xuất Tọa Độ Vùng Văn Bản

```python
def extract_text_regions(result):
    """Trích xuất tọa độ các vùng văn bản"""
    regions = []
    
    for line in result["lines"]:
        if line["bounding_rect"]:
            bbox = line["bounding_rect"]
            # Tính toán bounding box chữ nhật
            x_coords = [bbox["x1"], bbox["x2"], bbox["x3"], bbox["x4"]]
            y_coords = [bbox["y1"], bbox["y2"], bbox["y3"], bbox["y4"]]
            
            regions.append({
                "text": line["text"],
                "x_min": min(x_coords),
                "y_min": min(y_coords),
                "x_max": max(x_coords),
                "y_max": max(y_coords)
            })
    
    return regions

# Sử dụng
with OcrEngine() as ocr:
    result = ocr.recognize_pil(image)
    regions = extract_text_regions(result)
    
    for region in regions:
        print(f"Văn bản: {region['text']}")
        print(f"Vị trí: ({region['x_min']}, {region['y_min']}) -> ({region['x_max']}, {region['y_max']})")
```

### 4. Xử Lý Ảnh Screenshot

```python
from PIL import ImageGrab
from core.oneocr_optimized import OcrEngine

def ocr_screenshot(bbox=None):
    """
    Chụp màn hình và thực hiện OCR
    
    Args:
        bbox: Tuple (x1, y1, x2, y2) để chụp vùng cụ thể, None để chụp toàn màn hình
    """
    # Chụp màn hình
    screenshot = ImageGrab.grab(bbox=bbox)
    
    # Thực hiện OCR
    with OcrEngine() as ocr:
        result = ocr.recognize_pil(screenshot)
    
    return result

# Sử dụng
# Chụp toàn màn hình
result = ocr_screenshot()

# Chụp vùng cụ thể (x1=100, y1=100, x2=500, y2=300)
result = ocr_screenshot(bbox=(100, 100, 500, 300))
```

### 5. Tích Hợp Với OpenCV

```python
import cv2
import numpy as np
from core.oneocr_optimized import OcrEngine

def preprocess_and_ocr(image_path):
    """Tiền xử lý ảnh trước khi OCR"""
    # Đọc ảnh
    img = cv2.imread(image_path)
    
    # Chuyển sang grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Áp dụng threshold
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Encode lại thành buffer
    _, buffer = cv2.imencode('.png', binary)
    
    # Thực hiện OCR
    with OcrEngine() as ocr:
        result = ocr.recognize_cv2(buffer)
    
    return result
```

## Chạy Web Service

### Khởi Động Server

```python
from core.oneocr_optimized import serve

# Chạy với cấu hình mặc định
serve()

# Chạy với cấu hình tùy chỉnh
serve(host="127.0.0.1", port=8080, workers=2)
```

### Gọi API Từ Client

```python
import requests

# Đọc ảnh
with open("image.png", "rb") as f:
    image_data = f.read()

# Gửi request
response = requests.post(
    "http://localhost:8001/ocr",
    data=image_data,
    headers={"Content-Type": "application/octet-stream"}
)

# Xử lý kết quả
if response.status_code == 200:
    result = response.json()
    print(result["text"])
else:
    print(f"Lỗi: {response.json()['detail']}")
```

```bash
# Sử dụng curl
curl -X POST http://localhost:8001/ocr \
  --data-binary @image.png \
  -H "Content-Type: application/octet-stream"
```

### Health Check

```python
import requests

response = requests.get("http://localhost:8001/health")
print(response.json())
# Output: {"status": "healthy", "service": "OneOCR"}
```

## Xử Lý Lỗi

### Kiểm Tra Lỗi Trong Kết Quả

```python
with OcrEngine() as ocr:
    result = ocr.recognize_pil(image)
    
    if "error" in result:
        print(f"OCR thất bại: {result['error']}")
    elif not result["text"]:
        print("Không tìm thấy văn bản trong ảnh")
    else:
        print(f"Thành công: {result['text']}")
```

### Xử Lý Exception

```python
from core.oneocr_optimized import OcrEngine

try:
    ocr = OcrEngine()
    result = ocr.recognize_pil(image)
except RuntimeError as e:
    print(f"Lỗi khởi tạo OCR engine: {e}")
except Exception as e:
    print(f"Lỗi không xác định: {e}")
finally:
    if 'ocr' in locals():
        ocr.cleanup()
```

## Giới Hạn và Lưu Ý

### Kích Thước Ảnh
- Kích thước tối thiểu: 50x50 pixels
- Kích thước tối đa: 10000x10000 pixels

### Thread Safety
- `OcrEngine` là thread-safe, có thể sử dụng trong môi trường đa luồng
- Mỗi instance sử dụng lock nội bộ để đảm bảo an toàn

### Quản Lý Tài Nguyên
- Luôn gọi `cleanup()` hoặc sử dụng context manager
- Tránh tạo quá nhiều instance cùng lúc (tốn bộ nhớ)

### Hiệu Năng
- Ảnh lớn hơn 1MB sẽ sử dụng phương pháp xử lý tối ưu
- Khuyến nghị resize ảnh quá lớn trước khi OCR

## Troubleshooting

### Lỗi: "DLL initialization failed"
- Kiểm tra file `oneocr.dll` và `oneocr.onemodel` trong `~/.config/oneocr/`
- Đảm bảo đang chạy trên Windows

### Lỗi: "OCR engine not initialized"
- Engine khởi tạo thất bại, kiểm tra logs
- Thử khởi tạo lại hoặc restart ứng dụng

### Kết quả rỗng
- Ảnh có thể không chứa văn bản
- Thử tiền xử lý ảnh (tăng độ tương phản, threshold)
- Kiểm tra kích thước ảnh có hợp lệ

### Độ chính xác thấp
- Cải thiện chất lượng ảnh đầu vào
- Sử dụng ảnh có độ phân giải cao hơn
- Lọc kết quả theo confidence score

## Tham Khảo API

### Class: OcrEngine

#### Methods

**`__init__()`**
- Khởi tạo OCR engine
- Raises: `RuntimeError` nếu khởi tạo thất bại

**`recognize_pil(image: Image.Image) -> Dict[str, Any]`**
- Nhận dạng văn bản từ PIL Image
- Args: `image` - PIL Image object
- Returns: Dictionary chứa kết quả OCR

**`recognize_cv2(image_buffer) -> Dict[str, Any]`**
- Nhận dạng văn bản từ OpenCV buffer
- Args: `image_buffer` - Numpy array chứa dữ liệu ảnh
- Returns: Dictionary chứa kết quả OCR

**`cleanup()`**
- Giải phóng tài nguyên
- Nên gọi khi không sử dụng engine nữa

### Function: serve()

**`serve(host: str = "0.0.0.0", port: int = 8001, workers: int = 1)`**
- Khởi động web service OCR
- Args:
  - `host`: Địa chỉ host
  - `port`: Cổng listen
  - `workers`: Số worker processes

## Ví Dụ Tích Hợp Thực Tế

### Tích Hợp Vào Game Automation

```python
from PIL import ImageGrab
from core.oneocr_optimized import OcrEngine

class GameTextReader:
    def __init__(self):
        self.ocr = OcrEngine()
    
    def read_game_text(self, region):
        """Đọc văn bản từ vùng game"""
        screenshot = ImageGrab.grab(bbox=region)
        result = self.ocr.recognize_pil(screenshot)
        return result["text"]
    
    def find_text_position(self, region, target_text):
        """Tìm vị trí của văn bản trong game"""
        screenshot = ImageGrab.grab(bbox=region)
        result = self.ocr.recognize_pil(screenshot)
        
        for line in result["lines"]:
            if target_text in line["text"]:
                return line["bounding_rect"]
        
        return None
    
    def cleanup(self):
        self.ocr.cleanup()

# Sử dụng
reader = GameTextReader()
text = reader.read_game_text((100, 100, 500, 200))
print(f"Văn bản trong game: {text}")
reader.cleanup()
```

---

**Phiên bản:** 1.0.0  
**Cập nhật:** 2025-11-21
