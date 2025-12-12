# Hướng Dẫn Sử Dụng Auto C-Peach

## Mục Lục

1. [Giới Thiệu](#giới-thiệu)
2. [Khởi Động & Kết Nối](#khởi-động--kết-nối)
3. [Giao Diện Chính](#giao-diện-chính)
4. [Festival Automation](#tab-festival-automation)
5. [Gacha Automation](#tab-gacha-automation)
6. [Hopping Automation](#tab-hopping-automation)
7. [Settings](#tab-settings)
8. [Phím Tắt](#phím-tắt)
9. [Kết Quả & Ảnh Chụp](#kết-quả--ảnh-chụp)
10. [Xử Lý Sự Cố](#xử-lý-sự-cố)
11. [FAQ](#faq)

---

## Giới Thiệu

Auto C-Peach là công cụ tự động hóa game DOAX Venus Vacation với 3 chức năng chính:

| Chức năng | Mô tả |
|-----------|-------|
| **Festival Automation** | Tự động chơi các màn Festival với xác minh OCR |
| **Gacha Automation** | Tự động quay gacha với nhận diện template |
| **Hopping Automation** | Tự động Pool Hopping với xác minh item |

---

## Khởi Động & Kết Nối

### Bước 1: Mở Game
1. Khởi động DOAX Venus Vacation
2. Đảm bảo cửa sổ game hiển thị đầy đủ (không minimize)
3. Đặt game ở chế độ Windowed hoặc Borderless

### Bước 2: Chạy Auto C-Peach
Mở file **Auto C-Peach.exe** để khởi động ứng dụng.

### Bước 3: Kết Nối Device
1. Nhấn nút **"Connect Device"** ở góc trên bên phải
2. Chờ trạng thái chuyển sang **"Device Connected"** (màu xanh)
3. Nếu kết nối thất bại, nhấn **"Refresh"** để thử lại

### Trạng Thái Kết Nối

| Trạng thái | Màu | Ý nghĩa |
|------------|-----|---------|
| Not Connected | Đỏ | Chưa kết nối device |
| Connecting... | Xanh dương | Đang kết nối |
| Device Connected | Xanh lá | Kết nối thành công |
| Connection Failed | Đỏ | Kết nối thất bại |

---

## Giao Diện Chính

### Bố Cục Tổng Quan

```
┌─────────────────────────────────────────────────────────────┐
│  Header: Auto C-Peach    [Device Status] [Connect] [Refresh]│
├─────────────────────────────────────────────────────────────┤
│  [Festival] [Gacha] [Hopping] [Settings]  ← Tabs            │
├───────────────────────────────────┬─────────────────────────┤
│                                   │  Progress Panel         │
│  Configuration Panel              │  ├─ Progress bar        │
│  ├─ File Selection                │  └─ Statistics          │
│  ├─ Automation Settings           │─────────────────────────│
│  └─ Action Buttons                │  Quick Actions          │
│      [▶ Start] [⏸ Pause] [⏹ Stop]│  └─ Device/OCR test     │
│                                   │─────────────────────────│
│                                   │  Status                 │
│                                   │  └─ Current status      │
├───────────────────────────────────┴─────────────────────────┤
│  Activity Logs                    │  Error History          │
│  └─ Real-time log display         │  └─ Error tracking      │
├─────────────────────────────────────────────────────────────┤
│  Footer: © 2025 Auto C-Peach | Version 1.0        [Status]  │
└─────────────────────────────────────────────────────────────┘
```

### Các Thành Phần UI

#### 1. Pause/Resume Button
- Nhấn **"⏸ Pause"** để tạm dừng automation
- Nhấn **"▶ Resume"** để tiếp tục
- Phím tắt: `Ctrl+P`
- Hữu ích khi cần can thiệp thủ công giữa chừng

#### 2. Progress Panel
- **Progress Bar**: Thanh tiến trình với phần trăm
- **Statistics**: Tổng/Thành công/Thất bại/Bỏ qua
- **Time Info**: Thời gian đã chạy, ETA, thời gian trung bình
- **Current Item**: Hiển thị item đang xử lý

#### 3. Toast Notifications
Thông báo không chặn hiển thị ở góc màn hình:
- **Info** (xanh dương): Thông tin chung
- **Success** (xanh lá): Hoàn thành thành công
- **Warning** (cam): Cảnh báo
- **Error** (đỏ): Lỗi

#### 4. Error History Panel
- Hiển thị lịch sử lỗi với timestamp
- Phân loại theo severity (ERROR/WARNING/INFO)
- Nút **Clear** để xóa lịch sử

#### 5. Tooltips
Di chuột vào các nút để xem hướng dẫn:
- Start: "Start automation (Ctrl+Enter)"
- Stop: "Stop automation (Ctrl+Q, ESC, F9)"
- Pause: "Pause/Resume (Ctrl+P)"
- Browse: "Select data file (CSV/JSON)"

---

## Tab Festival Automation

### Tổng Quan
Festival Automation tự động hóa việc chơi các màn Festival trong game.

### Cơ Chế Hoạt Động

```
┌─────────────────────────────────────────────────────────────┐
│                    FESTIVAL AUTOMATION FLOW                  │
├─────────────────────────────────────────────────────────────┤
│  1. Touch Event Button                                      │
│  2. Snapshot Before                                         │
│  3. OCR Find & Touch Festival Name                          │
│     └─ Fallback: Use cached position if OCR fails          │
│  4. OCR Find & Touch Rank                                   │
│  5. Snapshot After                                          │
│  6. Pre-Battle Verification (OCR scan)                      │
│  7. Touch Challenge Button                                  │
│  8. Optional: Drag & Drop mini-game                         │
│  9. Touch OK/Skip buttons                                   │
│ 10. Touch Result Button                                     │
│ 11. Snapshot Result                                         │
│ 12. Post-Battle Verification                                │
│ 13. Close result dialogs                                    │
│                                                             │
│  ※ Pause/Resume available at any step                      │
└─────────────────────────────────────────────────────────────┘
```

### Tính Năng Đặc Biệt

- **Resume**: Tự động lưu tiến trình, có thể tiếp tục nếu bị gián đoạn
- **Fallback Cache**: Lưu vị trí chạm thành công để dùng khi OCR thất bại
- **Fuzzy Matching**: So khớp mờ để xử lý OCR sai ký tự

### Chuẩn Bị File Dữ Liệu

#### Format CSV
```csv
フェス名,フェスランク,推奨ランク,勝利点数,Sランクボーダー,初回クリア報酬,Sランク報酬
Festival Name 1,Rank A,E,1000,5000,Item A,Item B
```

### Các Bước Thực Hiện

1. **Chọn File Dữ Liệu**: Nhấn Browse → Preview để xem trước
2. **Chọn Stage Bắt Đầu**: Dropdown "Start Stage"
3. **Tùy Chọn Resume**: Tick "Force New Session" nếu muốn bắt đầu mới
4. **Bắt Đầu**: Nhấn **"▶ Start Festival"**
5. **Tạm Dừng**: Nhấn **"⏸ Pause"** hoặc `Ctrl+P`
6. **Dừng**: Nhấn **"⏹ Stop"** hoặc `Ctrl+Q`/`ESC`/`F9`

---

## Tab Gacha Automation

### Tổng Quan
Gacha Automation tự động hóa việc quay gacha với xác minh kết quả.

### Cơ Chế Hoạt Động

```
┌─────────────────────────────────────────────────────────────┐
│                    GACHA AUTOMATION FLOW                     │
├─────────────────────────────────────────────────────────────┤
│  1. Find Banner (template matching + auto-scroll)           │
│  2. Touch Banner to open                                    │
│  3. Select Pull Type (single/multi)                         │
│  4. Snapshot Before                                         │
│  5. Confirm Pull                                            │
│  6. Wait for animation                                      │
│  7. Skip Animation (optional)                               │
│  8. Snapshot After                                          │
│  9. Result Verification (Rarity + Swimsuit)                 │
│ 10. Repeat for remaining pulls                              │
│                                                             │
│  ※ Pause/Resume available between pulls                    │
└─────────────────────────────────────────────────────────────┘
```

### Chuẩn Bị Templates

```
templates/
└── jp/
    ├── tpl_event.png, tpl_ok.png, tpl_skip.png
    ├── tpl_ssr.png, tpl_sr.png
    └── banners/
        └── banner_name/
            ├── banner.png      # Ảnh banner chính
            └── swimsuit.png    # Ảnh swimsuit để xác minh
```

### Các Bước Thực Hiện

1. **Chọn Thư Mục Templates**: Browse → Refresh
2. **Cấu Hình Pull**: Rarity, Pulls, Type
3. **Thêm Banner Vào Queue**: Add to Queue
4. **Bắt Đầu**: Start Gacha
5. **Điều Khiển**: Pause/Resume/Stop

---

## Tab Hopping Automation

### Tổng Quan
Hopping Automation tự động hóa Pool Hopping với xác minh item.

### Cơ Chế Hoạt Động

```
┌─────────────────────────────────────────────────────────────┐
│                   HOPPING AUTOMATION FLOW                    │
├─────────────────────────────────────────────────────────────┤
│  1. Snapshot Before                                         │
│  2. Touch Use Button                                        │
│  3. Touch OK Button                                         │
│  4. Snapshot Item                                           │
│  5. Verification (OCR scan)                                 │
│  6. Compare with CSV data                                   │
│  7. Record OK/NG/Draw Unchecked                             │
│  8. Repeat for remaining courses                            │
│                                                             │
│  ※ Pause/Resume available at any step                      │
└─────────────────────────────────────────────────────────────┘
```

### Kết Quả Xác Minh

| Kết quả | Ý nghĩa |
|---------|---------|
| **OK** | Kết quả đúng, đã xác minh |
| **NG** | Kết quả sai, đã xác minh |
| **Draw Unchecked** | Không thể xác minh (OCR thất bại) |

### Các Bước Thực Hiện

1. **Chọn File Dữ Liệu**: Browse → Preview
2. **Chọn Course Bắt Đầu**: Dropdown
3. **Bắt Đầu**: Start Hopping
4. **Điều Khiển**: Pause/Resume/Stop

---

## Tab Settings

### Log Level

| Level | Mô tả | Khi nào dùng |
|-------|-------|--------------|
| DEBUG | Chi tiết nhất | Debug, tìm lỗi |
| INFO | Thông tin chung | Sử dụng bình thường |
| WARNING | Chỉ cảnh báo | Giảm log |
| ERROR | Chỉ lỗi | Chỉ quan tâm lỗi |

### Performance Settings

| Setting | Mô tả | Giá trị khuyến nghị |
|---------|-------|---------------------|
| Max Log Lines | Số dòng log tối đa | 1000 |
| Log Poll Interval | Tần suất cập nhật log (ms) | 200 |

---

## Phím Tắt

| Phím | Chức Năng |
|------|-----------|
| `Ctrl+Q` | Dừng automation (an toàn) |
| `ESC` | Dừng khẩn cấp |
| `F9` | Dừng automation |
| `Ctrl+P` | Pause/Resume |
| `Ctrl+Enter` | Bắt đầu automation |

---

## Kết Quả & Ảnh Chụp

### Cấu Trúc Thư Mục Result

```
result/
├── festival/
│   ├── snapshots/          # Ảnh chụp màn hình
│   ├── results/            # File kết quả (CSV/JSON/HTML)
│   ├── logs/               # Log chi tiết
│   └── .festival_resume.json
├── gacha/
│   ├── snapshots/
│   ├── results/
│   └── logs/
└── hopping/
    ├── snapshots/
    ├── results/
    ├── logs/
    └── .hopping_resume.json
```

### Output Formats

- **CSV**: Dễ mở bằng Excel
- **JSON**: Dễ xử lý bằng code
- **HTML**: Xem trực quan trong browser

---

## Xử Lý Sự Cố

### Lỗi Thường Gặp

| Lỗi | Nguyên nhân | Giải pháp |
|-----|-------------|-----------|
| Device not connected | Game chưa mở | Mở game, đặt ở foreground |
| OCR failed | Text không rõ | Kiểm tra độ phân giải game |
| Template not found | Ảnh không khớp | Chụp lại template |
| Connection timeout | Game không phản hồi | Restart game và tool |

### Tips Tối Ưu

1. **Giữ game ở foreground** - Không minimize
2. **Kiểm tra Error History** - Phát hiện lỗi sớm
3. **Sử dụng Pause** - Tạm dừng khi cần can thiệp

---

## FAQ

### Q: Khi nào nên dùng Pause?
A: Khi cần can thiệp thủ công (ví dụ: game hiện popup bất ngờ).

### Q: Làm sao để tiếp tục session bị gián đoạn?
A: Tool tự động lưu tiến trình. Khi chạy lại, sẽ hỏi có muốn tiếp tục không.

### Q: Dữ liệu có an toàn không?
A: Tool chỉ đọc file bạn cung cấp và lưu kết quả local. Không gửi dữ liệu ra ngoài.

---
