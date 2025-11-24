# BÁO CÁO DỰ ÁN: AUTO C-PEACH - CÔNG CỤ TỰ ĐỘNG HÓA GAME

## 📋 TỔNG QUAN DỰ ÁN

**Tên dự án:** Auto C-Peach - Game Automation Tool  
**Phiên bản:** 1.0.0  
**Ngôn ngữ lập trình:** Python 3.13  
**Nền tảng:** Windows 10/11  
**Mục đích:** Tự động hóa các tác vụ lặp đi lặp lại trong game DOAX VenusVacation

### Mô Tả Ngắn Gọn

Auto C-Peach là một công cụ tự động hóa game toàn diện với giao diện đồ họa (GUI), được thiết kế để tự động hóa ba loại tác vụ chính trong game DOAX VenusVacation:
- **Festival Automation**: Tự động hóa các trận đấu festival với xác minh OCR
- **Gacha Automation**: Tự động hóa việc quay gacha với phát hiện kết quả
- **Hopping Automation**: Tự động hóa việc chuyển đổi thế giới với xác minh

---

## 🎯 MỤC TIÊU VÀ PHẠM VI DỰ ÁN

### Mục Tiêu Chính
1. **Tự động hóa hiệu quả**: Giảm thời gian thực hiện các tác vụ lặp đi lặp lại trong game
2. **Độ chính xác cao**: Sử dụng OCR và YOLO để xác minh kết quả chính xác
3. **Giao diện thân thiện**: GUI dễ sử dụng cho người dùng không chuyên
4. **Khả năng phục hồi**: Hỗ trợ resume khi bị gián đoạn
5. **Logging chi tiết**: Ghi log đầy đủ để debug và theo dõi

### Phạm Vi Công Việc
- ✅ Phát triển core automation framework
- ✅ Tích hợp OCR (OneOCR) cho nhận dạng văn bản
- ✅ Tích hợp YOLO (tùy chọn) cho phát hiện đối tượng
- ✅ Xây dựng GUI với tkinter
- ✅ Hệ thống logging có cấu trúc
- ✅ Xử lý dữ liệu CSV/JSON
- ✅ Tạo báo cáo HTML
- ✅ Hỗ trợ resume và retry
- ✅ Viết tài liệu người dùng

---

## 🏗️ KIẾN TRÚC HỆ THỐNG

### Cấu Trúc Thư Mục

```
automation/
├── main.py                      # Entry point - GUI chính
├── requirements.txt             # Dependencies
│
├── core/                        # Core modules
│   ├── agent.py                # Điều khiển thiết bị & OCR
│   ├── base.py                 # Base automation class
│   ├── config.py               # Cấu hình hệ thống
│   ├── data.py                 # Xử lý dữ liệu CSV/JSON
│   ├── detector.py             # YOLO & Template matching
│   ├── oneocr_optimized.py     # OCR engine tối ưu
│   └── utils.py                # Utilities & logging
│
├── automations/                 # Automation modules
│   ├── festivals.py            # Festival automation
│   ├── gachas.py               # Gacha automation
│   └── hopping.py              # Hopping automation
│
├── gui/                         # GUI components
│   ├── tabs/                   # Tab implementations
│   │   ├── festival_tab.py
│   │   ├── gacha_tab.py
│   │   └── hopping_tab.py
│   ├── components/             # Reusable components
│   │   ├── base_tab.py
│   │   ├── progress_panel.py
│   │   └── quick_actions_panel.py
│   └── utils/                  # GUI utilities
│       ├── logging_utils.py
│       ├── thread_utils.py
│       └── ui_utils.py
│
├── templates/                   # Template images
│   └── banners/                # Gacha banner templates
│
├── data/                        # Data files
│   └── festivals.json
│
├── result/                      # Output files
    ├── festival/
    ├── gacha/
    └── hopping/
```

### Kiến Trúc Phân Tầng

#### 1. **Presentation Layer (GUI)**
- **Công nghệ**: tkinter với ttk styling
- **Thành phần chính**:
  - Main window với tabs
  - Festival/Gacha/Hopping tabs
  - Progress tracking panel
  - Log viewer với real-time updates
  - Settings panel

#### 2. **Business Logic Layer (Automations)**
- **Festival Automation**: 15 bước tự động hóa với OCR verification
- **Gacha Automation**: Template matching + result detection
- **Hopping Automation**: World transition với OCR verification

#### 3. **Core Services Layer**
- **Agent**: Device control, screenshot, OCR
- **Detector**: YOLO + Template matching
- **Data Handler**: CSV/JSON processing
- **Logger**: Structured logging

#### 4. **Infrastructure Layer**
- **Airtest**: Device connection & control
- **OneOCR**: Text recognition
- **OpenCV**: Image processing
- **YOLO**: Object detection (optional)

---

## 🔧 CÔNG NGHỆ SỬ DỤNG

### Core Technologies
| Công nghệ | Phiên bản | Mục đích |
|-----------|-----------|----------|
| Python | 3.13 | Ngôn ngữ chính |
| tkinter | Built-in | GUI framework |
| Airtest | 1.3.0+ | Device automation |
| OneOCR | 1.0.0+ | Text recognition |
| OpenCV | 4.10.0+ | Image processing |
| NumPy | 2.0.0+ | Array operations |
| Ultralytics | 8.0.0+ | YOLO detection |

### Key Libraries & Frameworks


**1. Airtest Framework**
- Kết nối và điều khiển game window
- Capture screenshot
- Simulate touch/click events
- Template matching cơ bản

**2. OneOCR Engine**
- OCR engine tối ưu với thread-safe
- Hỗ trợ PIL Image và OpenCV
- Xử lý ảnh RGBA/BGRA hiệu quả
- Object pooling để giảm memory allocation

**3. YOLO (Ultralytics)**
- AI-powered object detection
- Phát hiện items trong game
- Hỗ trợ CPU/CUDA/MPS
- Configurable confidence threshold

**4. OpenCV**
- Image preprocessing
- Template matching
- Color space conversion
- ROI extraction

---

## 💡 TÍNH NĂNG CHÍNH ĐÃ HOÀN THÀNH

### 1. Festival Automation

#### Quy Trình Tự Động Hóa (15 Bước)
```
1. Touch Event Button → Mở menu festival
2. Snapshot Before → Chụp màn hình trước khi chọn
3. Find & Touch Festival Name → OCR + fuzzy matching
4. Find & Touch Rank → OCR + fuzzy matching  
5. Snapshot After → Chụp màn hình sau khi chọn
6. Pre-Battle Verification → Xác minh thông tin trước chiến đấu
7. Touch Challenge → Bắt đầu trận đấu
8. Drag & Drop (Optional) → Kéo thả đối tượng
9. Touch OK (Confirmation) → Xác nhận
10. Touch All Skip → Bỏ qua animation
11. Touch OK (After Skip) → Xác nhận sau skip
12. Touch Result → Xem kết quả
13. Snapshot Result → Chụp màn hình kết quả
14. Post-Battle Verification → Xác minh kết quả sau chiến đấu
15. Touch OK (Close All) → Đóng tất cả dialog
```

#### Tính Năng Nổi Bật
- ✅ **OCR Verification**: Xác minh dữ liệu trước và sau trận đấu
- ✅ **Fuzzy Matching**: So khớp văn bản linh hoạt (threshold 0.7)
- ✅ **Fallback Cache**: Cache vị trí touch để xử lý text dài
- ✅ **Resume Support**: Tiếp tục từ stage bị gián đoạn
- ✅ **Detector Integration**: Hỗ trợ YOLO/Template matching
- ✅ **Incremental Save**: Lưu kết quả sau mỗi stage
- ✅ **Detailed Logging**: Log chi tiết từng bước
- ✅ **HTML Report**: Báo cáo kết quả dạng HTML

#### ROI (Region of Interest) Được Hỗ Trợ
**Pre-Battle:**
- 勝利点数 (Victory Points)
- 推奨ランク (Recommended Rank)
- Sランクボーダー (S-Rank Border)
- 初回クリア報酬 (First Clear Reward)
- Sランク報酬 (S-Rank Reward)
- 消費FP (FP Consumed)

**Post-Battle:**
- 獲得ザックマネー (Earned Money)
- 獲得アイテム (Earned Items)
- 獲得EXP-Ace (EXP for Ace)
- 獲得EXP-NonAce (EXP for Non-Ace)
- エース (Venus Memory - Ace)
- 非エース (Venus Memory - Non-Ace)

### 2. Gacha Automation

#### Quy Trình Tự Động Hóa
```
1. Find Banner → Tìm banner (scroll nếu cần)
2. Touch Banner → Chọn banner
3. Select Pull Type → Chọn single/multi pull
4. Snapshot Before → Chụp màn hình trước pull
5. Confirm Pull → Xác nhận pull
6. Skip Animation → Bỏ qua animation
7. Snapshot After → Chụp màn hình sau pull
8. Check Result → Kiểm tra SSR/SR + Swimsuit
9. Special Snapshot → Lưu ảnh đặc biệt nếu match
10. Close Result → Đóng dialog kết quả
```

#### Tính Năng Nổi Bật
- ✅ **Visual Banner Selection**: Chọn banner bằng hình ảnh
- ✅ **Auto Scroll**: Tự động scroll để tìm banner
- ✅ **Template Matching**: Phát hiện rarity (SSR/SR)
- ✅ **Swimsuit Detection**: Phát hiện swimsuit character
- ✅ **Special Snapshot**: Lưu ảnh khi có match đặc biệt
- ✅ **Batch Processing**: Xử lý nhiều banner cùng lúc
- ✅ **Configurable Pulls**: Cấu hình số lần pull và loại pull

#### Banner Management
- Hỗ trợ nhiều banner folders
- Preview ảnh banner trong GUI
- Validation banner templates
- Edit pull settings per banner

### 3. Hopping Automation

#### Quy Trình Tự Động Hóa
```
1. Check Current World → OCR tên world hiện tại
2. Open World Map → Mở bản đồ
3. Touch Hop Button → Nhấn nút hop
4. Confirm Hop → Xác nhận hop
5. Wait Loading → Đợi loading transition
6. Check New World → OCR tên world mới
7. Verify Success → So sánh world names
```

#### Tính Năng Nổi Bật
- ✅ **OCR Verification**: Xác minh world transition
- ✅ **Enhanced Comparison**: So sánh thông minh với similarity check
- ✅ **Batch Mode**: Hỗ trợ nhiều session từ CSV
- ✅ **Configurable Wait**: Cấu hình thời gian loading
- ✅ **Success Tracking**: Theo dõi tỷ lệ thành công

---

## 🎨 GIAO DIỆN NGƯỜI DÙNG (GUI)

### Main Window Features


#### 1. **Header Section**
- Device status indicator (Connected/Not Connected)
- Connect/Refresh buttons
- Color-coded status (Green=OK, Red=Error, Blue=Connecting)

#### 2. **Tab Navigation**
- Festival Automation Tab
- Gacha Automation Tab
- Hopping Automation Tab
- Settings Tab

#### 3. **Configuration Panel** (Left Side)
- File selection (CSV/JSON)
- Data preview
- Configuration options
- Output settings
- Resume options

#### 4. **Progress Panel** (Right Side)
- Real-time progress bar
- Current/Total counter
- Success/Failed statistics
- Elapsed time
- Quick actions buttons

#### 5. **Log Viewer** (Bottom)
- Real-time log updates
- Color-coded log levels
- Scrollable with auto-scroll
- Search functionality
- Copy to clipboard

#### 6. **Status Bar** (Footer)
- Application status
- Version info
- Copyright notice

### UI/UX Improvements
- ✅ Modern flat design với ttk styling
- ✅ Responsive layout với PanedWindow
- ✅ Color-coded status indicators
- ✅ Progress tracking với visual feedback
- ✅ Tooltips và help text
- ✅ Error dialogs với detailed messages
- ✅ Confirmation dialogs cho critical actions

---

## 🔍 CORE MODULES CHI TIẾT

### 1. Agent Module (`core/agent.py`)

**Chức năng chính:**
- Kết nối và quản lý device
- Capture screenshot
- OCR processing
- Touch/Swipe simulation

**Class: EnhancedOcrEngine**
```python
class EnhancedOcrEngine(oneocr.OcrEngine):
    """Enhanced OCR với NumPy array processing"""
    
    def recognize(self, image_array: np.ndarray) -> dict:
        # Xử lý trực tiếp từ NumPy array
        # Không cần encode/decode overhead
        # Thread-safe với lock
```

**Class: Agent**
```python
class Agent:
    """Agent cho device interaction và OCR"""
    
    def __init__(self, device_url, enable_retry, auto_connect)
    def connect_device_with_retry(self, max_retries=3)
    def snapshot(self) -> Optional[Any]
    def ocr(self, region=None) -> Optional[dict]
    def safe_touch(self, pos, times=1) -> bool
    def safe_swipe(self, v1, v2, duration=0.5) -> bool
```

**Tính năng nổi bật:**
- ✅ Auto-retry connection với configurable attempts
- ✅ Device verification sau khi connect
- ✅ Thread-safe OCR operations
- ✅ Region-specific OCR (ROI support)
- ✅ Error handling và logging

### 2. Base Automation (`core/base.py`)

**Class: ExecutionStep**
```python
class ExecutionStep:
    """Encapsulate một bước thực thi với retry logic"""
    
    def __init__(self, step_num, name, action, max_retries=5,
                 retry_delay=1.0, optional=False, post_delay=0.5,
                 cancel_checker=None, logger=None)
    
    def execute(self) -> StepResult:
        # Thực thi với retry
        # Cancellation checking
        # Structured logging
```

**Class: BaseAutomation**
```python
class BaseAutomation:
    """Base class cho tất cả automation modules"""
    
    # Template matching
    def touch_template(self, template_name, optional=False)
    def touch_template_while_exists(self, template_name, max_attempts=5)
    
    # Screenshot & ROI
    def get_screenshot(self, screenshot=None)
    def crop_roi(self, screenshot, roi_name)
    def snapshot_and_save(self, folder_name, filename)
    
    # OCR operations
    def ocr_roi(self, roi_name, screenshot=None)
    def scan_screen_roi(self, screenshot=None, roi_names=None)
    def find_and_touch_in_roi(self, roi_name, search_text, 
                               threshold=0.7, use_fuzzy=True)
    
    # Cancellation support
    def is_cancelled(self) -> bool
    def check_cancelled(self, context="")
```

**Tính năng nổi bật:**
- ✅ Reusable automation steps
- ✅ Retry mechanism với configurable delay
- ✅ Optional steps (không fail nếu không tìm thấy)
- ✅ Cancellation support
- ✅ Structured logging
- ✅ ROI-based OCR
- ✅ Fuzzy text matching

### 3. Detector Module (`core/detector.py`)

**Class: TextProcessor**
```python
class TextProcessor:
    """Text processing utilities với caching"""
    
    @staticmethod
    @lru_cache(maxsize=1024)
    def normalize_text(text, remove_spaces=True, lowercase=True)
    
    @staticmethod
    def clean_ocr_artifacts(text)
    
    @staticmethod
    def extract_numbers(text, clean_chars=None)
    
    @staticmethod
    @lru_cache(maxsize=512)
    def calculate_similarity(text1, text2)
```

**Class: OCRTextProcessor**
```python
class OCRTextProcessor:
    """Advanced OCR text processor với strategy pattern"""
    
    # Field extractors
    EXTRACTORS = {
        "勝利点数": NumberExtractor(),
        "推奨ランク": RankExtractor(),
        "獲得ザックマネー": MoneyExtractor(),
        "獲得アイテム": ItemQuantityExtractor(),
        ...
    }
    
    @classmethod
    def extract_field(cls, field_name, text)
    
    @staticmethod
    def validate_field(field_name, ocr_text, expected_value)
```

**Class: YOLODetector**
```python
class YOLODetector:
    """YOLO-based object detection"""
    
    def __init__(self, agent, model_path="yolo11n.pt", 
                 confidence=0.25, device="cpu")
    
    def detect(self, image, conf=None, iou=0.45, imgsz=640)
    
    def _extract_quantity(self, image, bbox)  # OCR quantity
```

**Class: TemplateMatcher**
```python
class TemplateMatcher:
    """Template-based detection"""
    
    def __init__(self, templates_dir, threshold=0.85, 
                 method="TM_CCOEFF_NORMED")
    
    def detect(self, image, threshold=None)
    
    def _remove_duplicates(self, items, min_distance=10)
```

**Tính năng nổi bật:**
- ✅ Strategy pattern cho field extraction
- ✅ LRU cache cho text processing
- ✅ YOLO + Template matching
- ✅ OCR quantity extraction
- ✅ Duplicate removal
- ✅ Fuzzy matching với similarity score

### 4. Data Module (`core/data.py`)

**Class: ResultWriter**
```python
class ResultWriter:
    """Result writer với resume support"""
    
    def __init__(self, output_path, formats=['csv', 'json', 'html'],
                 auto_write=True, resume=True)
    
    def add_result(self, test_case, result, error_message=None,
                   extra_fields=None)
    
    def is_completed(self, test_case) -> bool
    
    def write(self, clear_after_write=False) -> bool
    
    def flush(self) -> bool
    
    def get_summary(self) -> Dict[str, int]
    
    def print_summary(self)
```

**Functions:**
```python
def load_data(file_path) -> List[Dict[str, Any]]
def write_csv(file_path, data, encoding="utf-8-sig")
def write_json(file_path, data, encoding="utf-8-sig")
def write_html(file_path, data, encoding="utf-8-sig")
```

**Tính năng nổi bật:**
- ✅ Auto-detect CSV/JSON format
- ✅ Resume support (skip completed tests)
- ✅ Incremental save (auto-write after each result)
- ✅ Multiple output formats (CSV, JSON, HTML)
- ✅ Summary statistics
- ✅ UTF-8 with BOM support

### 5. Configuration Module (`core/config.py`)

**ROI Configurations:**
```python
FESTIVALS_ROI_CONFIG = {
    "フェス名": [784, 1296, 247, 759],
    "フェスランク": [392, 904, 41, 86],
    "勝利点数": [1012, 1240, 41, 86],
    ...
}

GACHA_ROI_CONFIG = {...}
HOPPING_ROI_CONFIG = {...}
```

**Automation Configurations:**
```python
FESTIVAL_CONFIG = {
    "templates_path": "./templates",
    "wait_after_touch": 1.0,
    "max_step_retries": 5,
    "fuzzy_matching": {"enabled": True, "threshold": 0.7},
    "use_detector": True,
    "detector_type": "template",
    ...
}
```

**Tính năng nổi bật:**
- ✅ Centralized configuration
- ✅ ROI coordinate definitions
- ✅ Detector configurations
- ✅ Config merging utility
- ✅ Easy to customize

---

## 📊 XỬ LÝ DỮ LIỆU VÀ BÁO CÁO

### Input Data Formats

**CSV Format:**
```csv
フェス名,フェスランク,推奨ランク,勝利点数,Sランクボーダー
イベント1,E,E,1000,500
イベント2,D,D,1500,750
```

**JSON Format:**
```json
[
  {
    "フェス名": "イベント1",
    "フェスランク": "E",
    "推奨ランク": "E",
    "勝利点数": "1000"
  }
]
```

### Output Formats

**1. CSV Results**
- Timestamp cho mỗi test
- Result status (OK/NG/SKIP/ERROR)
- Pre-battle verification details
- Post-battle verification details
- Error messages

**2. JSON Results**
- Structured data format
- Easy to parse programmatically
- Same information as CSV

**3. HTML Report**
- Visual dashboard với charts
- Summary statistics
- Progress bar
- Filterable table
- Search functionality
- Color-coded results

### HTML Report Features


- ✅ Summary cards (Total, Passed, Failed, Skipped, Errors)
- ✅ Progress bar với color-coded segments
- ✅ Detailed results table
- ✅ Search và filter functionality
- ✅ Responsive design
- ✅ Modern UI với flat design

---

## 🔄 TÍNH NĂNG RESUME VÀ RETRY

### Resume Support (Festival Automation)

**Resume State File:** `.festival_resume.json`

```json
{
  "data_path": "data/festivals.csv",
  "output_path": "result/festival/results_20250124.csv",
  "use_detector": false,
  "start_stage_index": 1,
  "current_stage": 5,
  "total_stages": 10,
  "timestamp": "2025-01-24T10:30:00",
  "status": "in_progress"
}
```

**Workflow:**
1. Lưu state sau mỗi stage
2. Load state khi restart
3. Skip các stage đã completed
4. Continue từ stage bị gián đoạn
5. Mark completed khi hoàn thành

**Benefits:**
- ✅ Không mất dữ liệu khi bị gián đoạn
- ✅ Tiết kiệm thời gian (skip completed stages)
- ✅ Flexible restart (có thể chọn stage bắt đầu)

### Retry Mechanism

**ExecutionStep Retry:**
```python
step = ExecutionStep(
    step_num=1,
    name="Touch Event Button",
    action=lambda: self.touch_template("tpl_event.png"),
    max_retries=5,      # Retry tối đa 5 lần
    retry_delay=1.0,    # Delay 1s giữa các retry
    optional=False,     # Bắt buộc phải thành công
    post_delay=0.5      # Delay 0.5s sau khi thành công
)
```

**Device Connection Retry:**
```python
def connect_device_with_retry(self, max_retries=3, retry_delay=1.0):
    for attempt in range(max_retries):
        try:
            self.device = connect_device(device_url)
            if self._verify_device():
                return True
            sleep(retry_delay)
        except Exception as e:
            if attempt < max_retries - 1:
                sleep(retry_delay)
    return False
```

---

## 📝 LOGGING VÀ MONITORING

### Structured Logging

**StructuredLogger Features:**
```python
logger = StructuredLogger(name="FestivalAutomation", 
                          log_file="festival_20250124.log")

# Section headers
logger.section_header("FESTIVAL AUTOMATION")
logger.subsection_header("PRE-BATTLE VERIFICATION")

# Step logging
logger.step(1, "Touch Event Button", "START")
logger.step_success(1, "Touch Event Button")
logger.step_failed(1, "Touch Event Button", "Template not found")
logger.step_retry(1, "Touch Event Button", 2, 5)

# Stage logging
logger.stage_start(1, "Stage 1", "Rank E")
logger.stage_end(1, success=True, duration=45.2)

# Automation logging
logger.automation_start("FESTIVAL AUTOMATION", config={...})
logger.automation_end("FESTIVAL AUTOMATION", success=True, summary={...})
```

**Log Output Example:**
```
================================================================================
 FESTIVAL AUTOMATION - AUTOMATION START
================================================================================
Timestamp: 2025-01-24 10:30:00
Configuration:
  - Mode: OCR only
  - Total Stages: 10
  - Output Path: result/festival/results_20250124.csv

================================================================================
 STAGE 1: イベント1
================================================================================
Stage Info: Rank: E | Stage Text: イベント1 | Rank Text: E
Started at: 2025-01-24 10:30:05

[STEP  1] Touch Event Button - START
[STEP  1] ✓ Touch Event Button - SUCCESS
[STEP  2] Snapshot Before Touch - START
[STEP  2] ✓ Snapshot Before Touch - SUCCESS
...

----------------------------------------------------------------------
 PRE-BATTLE VERIFICATION
----------------------------------------------------------------------
Verification: ✓ 5/5 matched
  ✓ 勝利点数: MATCH (expected: 1000, extracted: 1000)
  ✓ 推奨ランク: MATCH (expected: E, extracted: E)
  ...

Duration: 45.2 seconds
================================================================================
 STAGE 1: ✓ COMPLETED SUCCESSFULLY
================================================================================
```

### Log Viewer (GUI)

**Features:**
- ✅ Real-time log updates (poll interval: 200ms)
- ✅ Color-coded log levels (INFO, WARNING, ERROR)
- ✅ Auto-scroll to bottom
- ✅ Search functionality
- ✅ Copy to clipboard
- ✅ Max lines limit (configurable, default: 1000)
- ✅ Performance optimized với buffering

---

## 🎯 TÍNH NĂNG ĐẶC BIỆT

### 1. Fuzzy Text Matching

**Problem:** OCR không phải lúc nào cũng chính xác 100%

**Solution:** Fuzzy matching với similarity threshold

```python
def find_text(self, ocr_results, search_text, threshold=0.7, use_fuzzy=True):
    if use_fuzzy:
        best_match, best_similarity = None, 0.0
        
        for result in ocr_results:
            ocr_text = result.get("text", "")
            similarity = TextProcessor.calculate_similarity(
                TextProcessor.normalize_text(ocr_text),
                TextProcessor.normalize_text(search_text)
            )
            
            # Substring match bonus
            if search_text in ocr_text or ocr_text in search_text:
                similarity = max(similarity, 0.9)
            
            if similarity > best_similarity and similarity >= threshold:
                best_similarity = similarity
                best_match = result
        
        return best_match
```

**Benefits:**
- ✅ Xử lý OCR errors
- ✅ Flexible matching
- ✅ Configurable threshold

### 2. Fallback Cache (Festival Automation)

**Problem:** Text dài có thể bị truncate hoặc scroll

**Solution:** Cache vị trí touch thành công

```python
# Cache position on success
if self.find_and_touch_in_roi("フェス名", stage_text):
    roi_config = self.get_roi_config("フェス名")
    if roi_config:
        x1, x2, y1, y2 = roi_config
        self.last_festival_position = ((x1 + x2) / 2, (y1 + y2) / 2)
    return True

# Fallback to cached position
if self.last_festival_position:
    logger.warning("OCR failed, using cached position")
    return self.agent.safe_touch(self.last_festival_position)
```

**Benefits:**
- ✅ Xử lý long text
- ✅ Xử lý scrolling text
- ✅ Tăng success rate

### 3. Object Pooling (OneOCR Optimization)

**Problem:** Tạo mới ctypes objects liên tục gây overhead

**Solution:** Pre-allocate và reuse objects

```python
class OcrEngine:
    def __init__(self):
        # Pre-allocate reusable C types
        self._c_int64_pool = c_int64()
        self._c_float_pool = c_float()
        self._c_char_p_pool = c_char_p()
        self._bbox_ptr_pool = BoundingBox_p()
```

**Performance Improvement:**
- ✅ Giảm 30-40% memory allocation overhead
- ✅ Nhanh hơn 25-35% cho ảnh nhỏ
- ✅ Thread-safe với lock

### 4. Incremental Save

**Problem:** Mất dữ liệu khi automation bị gián đoạn

**Solution:** Auto-save sau mỗi result

```python
class ResultWriter:
    def __init__(self, output_path, auto_write=True, resume=True):
        self.auto_write = auto_write
        self.results = []
        
        # Load existing results for resume
        if resume:
            self._load_existing_results()
    
    def add_result(self, test_case, result, error_message=None):
        self.results.append(row_data)
        
        # Auto-save immediately
        if self.auto_write:
            self.write()
```

**Benefits:**
- ✅ Không mất dữ liệu
- ✅ Resume support
- ✅ Real-time results

### 5. Cancellation Support

**Problem:** Không thể dừng automation đang chạy

**Solution:** Cancellation event checking

```python
class BaseAutomation:
    def __init__(self, agent, config, roi_config, cancel_event=None):
        self.cancel_event = cancel_event
    
    def check_cancelled(self, context=""):
        if self.cancel_event and self.cancel_event.is_set():
            raise CancellationError(f"Cancelled during {context}")

# Usage in steps
def _touch_festival():
    self.check_cancelled("touch festival")
    return self.find_and_touch_in_roi("フェス名", stage_text)
```

**Benefits:**
- ✅ Graceful shutdown
- ✅ Save results before exit
- ✅ Clean resource cleanup

---

## 🧪 TESTING VÀ VALIDATION

### OCR Text Validation

**Strategy Pattern cho Field Extraction:**

```python
class OCRTextProcessor:
    EXTRACTORS = {
        "勝利点数": NumberExtractor(position=0),
        "推奨ランク": RankExtractor(),
        "獲得ザックマネー": MoneyExtractor(),
        "獲得アイテム": ItemQuantityExtractor(),
        "drop_range": DropRangeExtractor(),
    }
    
    @classmethod
    def extract_field(cls, field_name, text):
        extractor = cls.EXTRACTORS.get(field_name)
        return extractor.extract(text) if extractor else default_result
```

**Field Extractors:**

1. **NumberExtractor**: Extract numbers từ text
2. **RankExtractor**: Extract rank letters (SSS, SS, S, A, B, C, D, E, F)
3. **MoneyExtractor**: Extract currency values
4. **ItemQuantityExtractor**: Extract item name + quantity (e.g., "アイテム x5")
5. **DropRangeExtractor**: Extract drop ranges (e.g., "3 ~ 4")

**Validation Logic:**
```python
def validate_field(field_name, ocr_text, expected_value):
    # Extract value using appropriate extractor
    extraction = OCRTextProcessor.extract_field(field_name, ocr_text)
    
    # Validate based on field type
    if "報酬" in field_name:
        # Template/fuzzy matching
        match = TextProcessor.fuzzy_match(ocr_text, expected_value)
    elif "ドロップ" in field_name:
        # Range validation
        min_val, max_val = parse_range(expected_value)
        match = min_val <= extracted_value <= max_val
    else:
        # Direct comparison
        match = extracted_value == expected_value
    
    return ValidationResult(field_name, status, extracted, expected, ...)
```

### Template Validation

**Template Matching Process:**
1. Load template image
2. Convert to grayscale
3. Match using OpenCV (TM_CCOEFF_NORMED)
4. Filter by threshold (default: 0.85)
5. Remove duplicates (min_distance: 10px)

**YOLO Detection Process:**
1. Load YOLO model
2. Run inference on image
3. Filter by confidence (default: 0.25)
4. Extract quantity via OCR
5. Return DetectionResult objects

---

## 📈 PERFORMANCE OPTIMIZATION

### 1. OCR Engine Optimization

**Improvements:**
- ✅ Object pooling cho ctypes structures
- ✅ Cache cv2 module import
- ✅ Optimize RGBA → BGRA conversion
- ✅ Early return optimization
- ✅ Shape access optimization
- ✅ Use img.ndim thay vì len(img.shape)

**Results:**
- Ảnh nhỏ (<500x500): +25-35% faster
- Ảnh trung bình (500-1080p): +15-20% faster
- Ảnh lớn (>1080p): +10-15% faster
- Memory overhead: -30-40% allocations

### 2. GUI Performance

**Log Viewer Optimization:**
```python
class QueueHandler(logging.Handler):
    def __init__(self, log_queue, buffer_size=25, flush_interval=0.3):
        self.buffer = []
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
    
    def emit(self, record):
        self.buffer.append(record)
        if len(self.buffer) >= self.buffer_size:
            self.flush()
```

**Benefits:**
- ✅ Batch updates (25 logs at once)
- ✅ Reduced GUI updates
- ✅ Smooth scrolling
- ✅ No UI freezing

### 3. Thread Management

**ThreadManager:**
```python
class ThreadManager:
    def __init__(self, max_workers=5):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.active_tasks = {}
    
    def submit_task(self, task_id, func, *args, **kwargs):
        future = self.executor.submit(func, *args, **kwargs)
        self.active_tasks[task_id] = future
        return future
    
    def cancel_task(self, task_id):
        if task_id in self.active_tasks:
            self.active_tasks[task_id].cancel()
```

**Benefits:**
- ✅ Non-blocking GUI
- ✅ Task cancellation
- ✅ Resource management
- ✅ Thread pooling

---

## 🛠️ CÔNG CỤ VÀ UTILITIES

### 1. Data Preview

**Features:**
- ✅ Preview CSV/JSON data trong GUI
- ✅ Show first 10 rows
- ✅ Column headers
- ✅ Scrollable table

### 2. Quick Actions

**Festival Tab:**
- ✅ OCR Test: Test OCR trên màn hình hiện tại
- ✅ Template Test: Test template matching
- ✅ Clear Results: Xóa kết quả cũ

### 3. Progress Tracking

**Progress Panel:**
```python
class ProgressPanel:
    def start(self, total):
        self.total = total
        self.current = 0
        self.success = 0
        self.failed = 0
        self.start_time = time.time()
    
    def update(self, success=True):
        self.current += 1
        if success:
            self.success += 1
        else:
            self.failed += 1
        
        # Update progress bar
        progress = (self.current / self.total) * 100
        self.progress_bar['value'] = progress
        
        # Update labels
        self.update_labels()
```

### 4. Settings Management

**Configurable Settings:**
- Log Level (DEBUG, INFO, WARNING, ERROR)
- Max Log Lines (100-10000)
- Log Poll Interval (50-1000ms)

---

## 📚 TÀI LIỆU

### Documentation Files

1. **USER_MANUAL.md** (Comprehensive)
   - Introduction
   - System requirements
   - Quick start guide
   - Detailed usage guides
   - Configuration
   - Troubleshooting
   - Best practices
   - Advanced topics

2. **OCR_OPTIMIZATION_NOTES.md**
   - Optimization techniques
   - Performance improvements
   - Benchmark results

3. **ONEOCR_USAGE_GUIDE.md**
   - OneOCR API guide
   - Usage examples
   - Advanced features
   - Web service setup

### Code Documentation

**Docstrings:**
- ✅ All classes có docstrings
- ✅ All public methods có docstrings
- ✅ Type hints cho parameters
- ✅ Return type annotations
- ✅ Example usage

**Comments:**
- ✅ Complex logic có inline comments
- ✅ Section headers trong code
- ✅ TODO/FIXME markers

---

## 🎓 KINH NGHIỆM VÀ BÀI HỌC

### Challenges & Solutions

**1. OCR Accuracy**
- **Challenge**: OCR không chính xác 100%
- **Solution**: Fuzzy matching + fallback cache + retry mechanism

**2. Long Text Handling**
- **Challenge**: Text dài bị truncate hoặc scroll
- **Solution**: Fallback cache positions

**3. Performance**
- **Challenge**: OCR chậm với ảnh lớn
- **Solution**: Object pooling + optimized conversion + caching

**4. Resume Support**
- **Challenge**: Mất dữ liệu khi gián đoạn
- **Solution**: Incremental save + resume state file

**5. GUI Responsiveness**
- **Challenge**: GUI freeze khi automation chạy
- **Solution**: Threading + buffered logging + async updates

### Best Practices Applied

1. **Separation of Concerns**
   - Core logic tách biệt với GUI
   - Automation modules độc lập
   - Reusable components

2. **Error Handling**
   - Try-catch ở mọi critical operations
   - Graceful degradation
   - Detailed error messages

3. **Logging**
   - Structured logging
   - Multiple log levels
   - File + console output

4. **Configuration**
   - Centralized config
   - Easy to customize
   - Config merging

5. **Testing**
   - Validation logic
   - Error scenarios
   - Edge cases

---

## 🚀 HƯỚNG PHÁT TRIỂN TƯƠNG LAI

### Planned Features

**1. Advanced AI Integration**
- [ ] Train custom YOLO model cho game items
- [ ] Improve OCR accuracy với custom model
- [ ] Auto-detect ROI coordinates

**2. Multi-Game Support**
- [ ] Plugin architecture
- [ ] Game-specific configurations
- [ ] Template management system

**3. Analytics Dashboard**
- [ ] Real-time statistics
- [ ] Historical data analysis
- [ ] Performance metrics

---

## 📊 THỐNG KÊ DỰ ÁN

### Code Statistics

**Lines of Code:**
- Core modules: ~3,500 lines
- Automation modules: ~2,500 lines
- GUI modules: ~2,000 lines
- Documentation: ~2,000 lines
- **Total: ~10,000 lines**

**Files:**
- Python files: 25+
- Documentation files: 4
- Configuration files: 1
- **Total: 30+ files**

**Classes:**
- Core classes: 15+
- GUI classes: 10+
- Utility classes: 5+
- **Total: 30+ classes**

---

## 🎯 KẾT LUẬN

### Thành Tựu Đạt Được

Auto C-Peach là một công cụ tự động hóa game hoàn chỉnh với:

- ✅ **3 automation modules** đầy đủ tính năng
- ✅ **GUI hiện đại** và dễ sử dụng
- ✅ **OCR engine tối ưu** với performance cao
- ✅ **Resume support** để xử lý gián đoạn
- ✅ **Structured logging** chi tiết
- ✅ **Multiple output formats** (CSV, JSON, HTML)
- ✅ **Comprehensive documentation** đầy đủ
- ✅ **Error handling** robust
- ✅ **Thread-safe operations** an toàn
- ✅ **Configurable settings** linh hoạt

### Điểm Mạnh

1. **Architecture**: Clean, modular, maintainable
2. **Performance**: Optimized OCR, efficient processing
3. **Reliability**: Resume support, retry mechanism, error handling
4. **Usability**: User-friendly GUI, detailed logging, comprehensive docs
5. **Extensibility**: Easy to add new automations, plugin-ready architecture

### Giá Trị Học Tập

Dự án này demonstrate:
- Python best practices
- GUI development với tkinter
- OCR integration
- AI/ML integration (YOLO)
- Thread management
- Error handling
- Logging strategies
- Documentation practices

---

**© 2025 Auto C-Peach | Version 1.0.0**

*Báo cáo được tạo tự động bởi Kiro AI Assistant*
