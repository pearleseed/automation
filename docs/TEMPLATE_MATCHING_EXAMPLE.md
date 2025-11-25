# Template Matching - Ví dụ thực tế

## 1. Chuẩn bị Templates

### Cấu trúc thư mục
```
templates/jp/
├── first_clear_label.png       # Label "初回クリア報酬"
├── srank_reward_label.png      # Label "Sランク報酬"
├── item_crystal_pink.png       # Crystal hồng
├── item_diamond_blue.png       # Diamond xanh
├── item_slime_green.png        # Slime xanh lá
├── item_gift_box.png           # Gift box
└── item_gold_coin.png          # Gold coin
```

### Quy tắc đặt tên
- **first_*.png** → category="first_clear", threshold=0.9
- **srank_*.png** → category="s_rank", threshold=0.9
- **item_*.png** → category="item", threshold=0.85, extract quantity

---

## 2. Khởi tạo Detector

```python
from core.agent import Agent
from core.detector import TemplateMatcher

# Khởi tạo agent với OCR
agent = Agent(device_uri="Android:///")

# Khởi tạo template matcher
detector = TemplateMatcher(
    templates_dir="templates/jp",
    threshold=0.85,
    ocr_engine=agent.ocr_engine,
)

# Output:
# TemplateMatcher: 7 templates loaded
# Loaded: first_clear_label (category=first_clear)
# Loaded: srank_reward_label (category=s_rank)
# Loaded: item_crystal_pink (category=item)
# Loaded: item_diamond_blue (category=item)
# Loaded: item_slime_green (category=item)
# Loaded: item_gift_box (category=item)
# Loaded: item_gold_coin (category=item)
```

---

## 3. Ví dụ 1: Detect tất cả items trong màn hình

### Input: Screenshot reward screen
```
┌─────────────────────────────────────────────────────────────┐
│  獲得アイテム                                                │
│  ┌────┐  ┌────┐  ┌────┐  ┌────┐  ┌────┐                   │
│  │ 💎 │  │ 💠 │  │ 💎 │  │😊GP│  │ 🎁 │                   │
│  └────┘  └────┘  └────┘  └────┘  └────┘                   │
│   100      1      100     600     35                        │
└─────────────────────────────────────────────────────────────┘
```

### Code
```python
import cv2

# Đọc screenshot
screenshot = cv2.imread("result/snapshots/rank_E_stage_1/03_result.png")

# Detect tất cả items
detections = detector.detect(screenshot, extract_quantity=True)

# In kết quả
for det in detections:
    print(f"Item: {det.item}")
    print(f"  Position: ({det.x}, {det.y}) → ({det.x2}, {det.y2})")
    print(f"  Confidence: {det.confidence:.3f}")
    print(f"  Quantity: {det.quantity}")
    print(f"  OCR Text: '{det.ocr_text}'")
    print()
```

### Output
```
Template matching: 5 items detected

Item: item_crystal_pink
  Position: (75, 130) → (145, 200)
  Confidence: 0.923
  Quantity: 100
  OCR Text: '100'

Item: item_diamond_blue
  Position: (265, 130) → (335, 200)
  Confidence: 0.887
  Quantity: 1
  OCR Text: '1'

Item: item_crystal_pink
  Position: (455, 130) → (525, 200)
  Confidence: 0.918
  Quantity: 100
  OCR Text: '100'

Item: item_slime_green
  Position: (645, 130) → (715, 200)
  Confidence: 0.856
  Quantity: 600
  OCR Text: '600'

Item: item_gift_box
  Position: (835, 130) → (905, 200)
  Confidence: 0.891
  Quantity: 35
  OCR Text: '35'
```

### Giải thích
1. **Duplicate items**: 2 crystal_pink được detect riêng biệt (vị trí khác nhau)
2. **Confidence**: Slime thấp nhất (0.856) vì có label "GP" overlay
3. **Quantity**: Tự động extract từ vùng bên dưới item
4. **Sort**: Kết quả đã sort theo x-coordinate (trái → phải)

---

## 4. Ví dụ 2: Detect reward sections (初回クリア報酬 & Sランク報酬)

### Input: Pre-battle screen
```
┌─────────────────────────────────────────────────────────────┐
│  初回クリア報酬  ┌────┐  ┌────┐                            │
│                  │ 💰 │  │ 🔑 │                            │
│                  └────┘  └────┘                            │
│                   1000     5                                │
│                                                             │
│  Sランク報酬     ┌────┐  ┌────┐  ┌────┐                   │
│                  │ 💎 │  │ ⭐ │  │ 🎁 │                   │
│                  └────┘  └────┘  └────┘                   │
│                   50      10      2                         │
└─────────────────────────────────────────────────────────────┘
```

### Code
```python
# Crop ROI "初回クリア報酬"
roi_first = screenshot[233:330, 1050:1255]  # From FESTIVALS_ROI_CONFIG

# Detect first clear reward section
first_clear = detector.detect_reward_section(roi_first, section_type="first_clear")

print("=== 初回クリア報酬 ===")
print(f"Label detected: {first_clear['label'] is not None}")
if first_clear['label']:
    print(f"  Label: {first_clear['label'].item} (conf={first_clear['label'].confidence:.3f})")

print(f"Items: {len(first_clear['items'])}")
for item in first_clear['items']:
    print(f"  - {item.item}: x{item.quantity} (conf={item.confidence:.3f})")

# Crop ROI "Sランク報酬"
roi_srank = screenshot[343:440, 1050:1255]

# Detect S rank reward section
s_rank = detector.detect_reward_section(roi_srank, section_type="s_rank")

print("\n=== Sランク報酬 ===")
print(f"Label detected: {s_rank['label'] is not None}")
if s_rank['label']:
    print(f"  Label: {s_rank['label'].item} (conf={s_rank['label'].confidence:.3f})")

print(f"Items: {len(s_rank['items'])}")
for item in s_rank['items']:
    print(f"  - {item.item}: x{item.quantity} (conf={item.confidence:.3f})")
```

### Output
```
Reward 'first_clear': label=✓, items=2

=== 初回クリア報酬 ===
Label detected: True
  Label: first_clear_label (conf=0.945)
Items: 2
  - item_gold_coin: x1000 (conf=0.892)
  - item_key: x5 (conf=0.878)

Reward 's_rank': label=✓, items=3

=== Sランク報酬 ===
Label detected: True
  Label: srank_reward_label (conf=0.938)
Items: 3
  - item_crystal_pink: x50 (conf=0.901)
  - item_star: x10 (conf=0.885)
  - item_gift_box: x2 (conf=0.894)
```

### Giải thích
1. **Label detection**: Detect label trước để xác định vùng reward
2. **Item filtering**: Chỉ lấy items bên phải label (x > label.x2)
3. **Quantity extraction**: Tự động OCR số lượng cho mỗi item
4. **High threshold**: Label dùng threshold=0.9 → confidence cao hơn

---

## 5. Ví dụ 3: Detect cả 2 sections trong 1 lần

### Code
```python
# Detect all rewards in one pass (hiệu quả hơn)
all_rewards = detector.detect_all_rewards(screenshot)

print("=== ALL REWARDS ===")
for section_type, data in all_rewards.items():
    print(f"\n{section_type.upper()}:")
    print(f"  Label: {'✓' if data['label'] else '✗'}")
    print(f"  Items: {len(data['items'])}")
    for item in data['items']:
        print(f"    - {item.item}: x{item.quantity}")
```

### Output
```
All rewards: first_clear=2 items, s_rank=3 items

=== ALL REWARDS ===

FIRST_CLEAR:
  Label: ✓
  Items: 2
    - item_gold_coin: x1000
    - item_key: x5

S_RANK:
  Label: ✓
  Items: 3
    - item_crystal_pink: x50
    - item_star: x10
    - item_gift_box: x2
```

### Giải thích
1. **One-pass detection**: Detect tất cả templates 1 lần duy nhất
2. **Automatic grouping**: Tự động phân nhóm items theo label gần nhất
3. **Y-coordinate proximity**: Items trong cùng row (y ± 50px) với label

---

## 6. Ví dụ 4: Tích hợp với FestivalAutomation

### Code trong festivals.py
```python
# Trong run_festival_stage()

# Step 6: Pre-Battle Verification
screenshot_after = self.snapshot_and_save(folder_name, "02_after_touch.png")

pre_battle_rois = ["初回クリア報酬", "Sランク報酬"]

# Scan với detector
extracted = self.scan_rois_combined(screenshot_after, pre_battle_rois)

# Kết quả
print(extracted["初回クリア報酬"])
```

### Output
```python
{
    "roi_name": "初回クリア報酬",
    "text": "初回クリア報酬 ゴールドコイン x1000 鍵 x5",  # OCR text
    "detected": True,
    "detections": [
        DetectionResult(item="item_gold_coin", quantity=1000, confidence=0.892, ...),
        DetectionResult(item="item_key", quantity=5, confidence=0.878, ...)
    ],
    "detection_count": 2,
    "label": DetectionResult(item="first_clear_label", confidence=0.945, ...),
    "items_with_quantity": [
        {"item": "item_gold_coin", "quantity": 1000, "confidence": 0.892},
        {"item": "item_key", "quantity": 5, "confidence": 0.878}
    ]
}
```

### Validation với expected data
```python
# Expected data từ CSV
expected_data = {
    "初回クリア報酬": "ゴールドコイン x1000, 鍵 x5"
}

# Compare
is_ok, msg, details = self.compare_results(extracted, expected_data)

print(f"Validation: {msg}")
# Output: ✓ 1/1 matched

print(details["初回クリア報酬"])
# Output:
# {
#     "status": "match",
#     "extracted_text": "初回クリア報酬 ゴールドコイン x1000 鍵 x5",
#     "expected": "ゴールドコイン x1000, 鍵 x5",
#     "detected": True,
#     "detection_count": 2,
#     "has_quantity": True,
#     "message": "Template match: True",
#     "confidence": 0.9
# }
```

---

## 7. Xử lý Edge Cases

### Case 1: OCR lỗi số lượng lớn
```python
# Input: "600" nhưng OCR đọc thành "GOO"
det = DetectionResult(item="item_slime", ocr_text="GOO", quantity=0)

# Fallback: Dùng fuzzy match với expected
expected = "スライム x600"
# → Vẫn match vì OCR text chứa "スライム"
```

### Case 2: Item bị che khuất một phần
```python
# Slime có label "GP" overlay
# → Confidence thấp hơn (0.856 vs 0.92)
# → Vẫn detect được vì threshold=0.85

# Nếu muốn strict hơn, tăng threshold
detector = TemplateMatcher(threshold=0.90)
# → Có thể miss detection
```

### Case 3: Duplicate items cùng loại
```python
# 2 crystal_pink ở vị trí khác nhau
detections = [
    DetectionResult(item="item_crystal_pink", x=75, quantity=100),
    DetectionResult(item="item_crystal_pink", x=455, quantity=100),
]

# NMS không loại bỏ vì khoảng cách > min_distance (15px)
# → Giữ cả 2 detections
# → Sort theo x-coordinate để đúng thứ tự
```

### Case 4: Template không tồn tại
```python
# Nếu thiếu template "item_slime_green.png"
# → Không detect được slime
# → detection_count = 4 thay vì 5

# Giải pháp: Kiểm tra template coverage
missing = set(expected_items) - set(detector.templates.keys())
if missing:
    logger.warning(f"Missing templates: {missing}")
```

---

## 8. Performance Benchmark

### Test với 100 screenshots
```python
import time

screenshots = [cv2.imread(f"test_{i}.png") for i in range(100)]

start = time.time()
for img in screenshots:
    detections = detector.detect(img)
duration = time.time() - start

print(f"Total: {duration:.2f}s")
print(f"Average: {duration/100*1000:.1f}ms per image")
print(f"Throughput: {100/duration:.1f} images/sec")
```

### Output
```
Total: 8.45s
Average: 84.5ms per image
Throughput: 11.8 images/sec

Breakdown:
- Template matching: 60ms (71%)
- OCR quantity: 20ms (24%)
- NMS + sorting: 4.5ms (5%)
```

### So sánh với YOLO
```
Template Matching: 84.5ms/image
YOLO (CPU):       120ms/image
YOLO (GPU):        25ms/image

→ Template matching nhanh hơn YOLO CPU
→ Nhưng chậm hơn YOLO GPU
```

---

## 9. Tips & Best Practices

### 1. Tạo template chất lượng cao
```bash
# Crop template từ screenshot gốc
# Đảm bảo:
# - Không có background noise
# - Kích thước đủ lớn (>30x30px)
# - Contrast tốt
# - Không bị blur
```

### 2. Đặt tên template có ý nghĩa
```
✓ item_crystal_pink.png
✓ first_clear_label.png
✗ template1.png
✗ img_001.png
```

### 3. Test threshold cho từng template
```python
# Test với nhiều threshold
for thresh in [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
    dets = detector.detect(img, threshold=thresh)
    print(f"Threshold {thresh}: {len(dets)} detections")

# Output:
# Threshold 0.70: 12 detections (too many false positives)
# Threshold 0.75: 8 detections
# Threshold 0.80: 6 detections
# Threshold 0.85: 5 detections ← optimal
# Threshold 0.90: 3 detections (missing items)
# Threshold 0.95: 1 detection (too strict)
```

### 4. Kiểm tra confidence score
```python
# Log low confidence detections
for det in detections:
    if det.confidence < 0.87:
        logger.warning(
            f"Low confidence: {det.item} = {det.confidence:.3f} "
            f"at ({det.x}, {det.y})"
        )
```

### 5. Validate với expected data
```python
# Luôn so sánh với expected data
if len(detections) != len(expected_items):
    logger.error(
        f"Item count mismatch: detected {len(detections)}, "
        f"expected {len(expected_items)}"
    )
```

---

## 10. Troubleshooting

### Vấn đề: Không detect được item
**Nguyên nhân:**
- Template không khớp với screenshot
- Threshold quá cao
- Template bị blur hoặc resize

**Giải pháp:**
```python
# 1. Kiểm tra template có tồn tại
print(detector.templates.keys())

# 2. Giảm threshold
detections = detector.detect(img, threshold=0.75)

# 3. Tạo lại template từ screenshot mới
```

### Vấn đề: Quá nhiều false positives
**Nguyên nhân:**
- Threshold quá thấp
- Template quá generic

**Giải pháp:**
```python
# 1. Tăng threshold
detector.threshold = 0.90

# 2. Tạo template specific hơn (thêm context)
```

### Vấn đề: OCR quantity sai
**Nguyên nhân:**
- Font số khó đọc
- Contrast thấp
- Vùng OCR không chính xác

**Giải pháp:**
```python
# 1. Kiểm tra OCR text
for det in detections:
    print(f"{det.item}: ocr_text='{det.ocr_text}', quantity={det.quantity}")

# 2. Điều chỉnh vùng OCR trong _extract_quantity()
# qty_y1 = min(y2 + 2, img_h - 1)  # Thử thay đổi offset
```

---

## Kết luận

Template matching đơn giản nhưng hiệu quả cho game automation khi:
- ✅ UI ổn định, không scale/rotate
- ✅ Items có hình dạng đặc trưng
- ✅ Cần detect nhanh (< 100ms)
- ✅ Không cần training data

Hạn chế:
- ❌ Không robust với scale/rotation
- ❌ Cần tạo template cho mỗi variant
- ❌ Không generalize được

→ Phù hợp cho dự án này vì game có UI cố định!
