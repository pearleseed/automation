# Gacha Automation Guide

## 📁 Folder Structure

```
templates/
  ├── tpl_ssr.png                    # SSR rarity template
  ├── tpl_sr.png                     # SR rarity template
  ├── tpl_ok.png                     # OK/Confirm button
  ├── tpl_allskip.png                # Skip animation button
  ├── tpl_single_pull.png            # Single pull button
  ├── tpl_multi_pull.png             # Multi pull button
  ├── tpl_button_down.png            # Scroll down button
  └── banners/
      ├── summer_gacha/              # Summer banner folder
      │   ├── banner.png             # ← Ảnh banner (hiển thị trong game)
      │   ├── swimsuit_red.png       # ← Swimsuit muốn tìm
      │   ├── swimsuit_blue.png
      │   └── swimsuit_white.png
      ├── winter_gacha/              # Winter banner folder
      │   ├── banner.png
      │   ├── swimsuit_black.png
      │   └── swimsuit_gold.png
      └── special_limited/           # Special banner folder
          ├── banner.png
          ├── swimsuit_limited_1.png
          └── swimsuit_limited_2.png
```

## 🚀 Quick Start

### 1. Chuẩn bị Templates

1. Chụp ảnh các **nút bấm cơ bản** → lưu vào `templates/`:
   - `tpl_ssr.png` - icon SSR (5 sao)
   - `tpl_sr.png` - icon SR (4 sao)
   - `tpl_ok.png` - nút OK/Confirm
   - `tpl_allskip.png` - nút Skip All
   - `tpl_single_pull.png` - nút Single Pull
   - `tpl_multi_pull.png` - nút Multi Pull (10x)
   - `tpl_button_down.png` - nút scroll xuống

2. **Cho mỗi banner gacha:**
   - Tạo folder trong `templates/banners/`
     - Ví dụ: `templates/banners/summer_gacha/`
   
   - Chụp ảnh **banner** (banner hiển thị trong game) → lưu vào folder với tên `banner.png`
     - `templates/banners/summer_gacha/banner.png`
   
   - Chụp tất cả **swimsuit muốn tìm** → lưu vào cùng folder
     - `templates/banners/summer_gacha/swimsuit_red.png`
     - `templates/banners/summer_gacha/swimsuit_blue.png`
     - `templates/banners/summer_gacha/swimsuit_white.png`

**Ví dụ đầy đủ:**
```
templates/banners/summer_gacha/
  ├── banner.png              # ← Ảnh banner (REQUIRED)
  ├── swimsuit_red.png        # ← Swimsuit muốn tìm
  ├── swimsuit_blue.png
  └── swimsuit_white.png
```

**⚠️ Lưu ý:** File banner PHẢI đặt tên là `banner.png` (hoặc `.jpg`)

### 2. Sử dụng GUI

1. Mở **Gacha Automation** tab
2. Chọn **Templates Folder** (nếu chưa đúng)
3. UI sẽ tự động scan folder `templates/banners/` và hiển thị tất cả banner
4. Chọn **Target Rarity**: SSR hoặc SR
5. Nhập **Number of Pulls**: số lần pull
6. Chọn **Pull Type**: Single hoặc Multi (10x)
7. Click **Add** trên banner gacha muốn pull
8. Lặp lại bước 4-7 cho các banner khác (nếu có)
9. Click **▶ Start** để bắt đầu

**💡 Tips:**
- UI tự động detect banner và swimsuit trong cùng folder
- Icon ✓ (màu xanh) = folder có đủ banner + swimsuit
- Icon ? (màu cam) = folder thiếu file hoặc chưa đúng

### 3. Kết quả

- Screenshots được lưu trong: `result/gacha/snapshots/`
- Mỗi gacha có folder riêng: `01_gacha_name_timestamp/`
- File có `_SPECIAL.png` = tìm thấy cả SSR/SR + Swimsuit
- CSV results: `result/gacha/results/gacha_YYYYMMDD_HHMMSS.csv`
- Logs: `result/gacha/results/logs/gacha_YYYYMMDD_HHMMSS.log`

## 💡 Tips

- **Preview Swimsuit**: Click "Preview" để xem tất cả swimsuit trong folder
- **Edit Gacha**: Click "Edit" để sửa số pulls/rarity/type cho banner đã thêm
- **Multiple Gachas**: Có thể thêm nhiều banner với config khác nhau
- **Scroll Auto**: Nếu không thấy banner, sẽ tự động scroll down để tìm
- **Special Match**: Khi tìm thấy cả Rarity + Swimsuit → tự động snapshot đặc biệt

## 🔧 Config Parameters

```python
{
    'templates_path': './templates',      # Folder chứa templates
    'wait_after_pull': 2.0,               # Đợi sau khi pull (giây)
    'max_scroll_attempts': 10,            # Số lần scroll tối đa để tìm banner
}
```

## ❓ FAQ

**Q: Làm sao biết template nào cần chụp?**
A: Chạy thử 1 lần, xem log để biết template nào missing

**Q: Banner folder đặt tên gì?**
A: Tên gì cũng được, miễn dễ nhớ. Ví dụ: `summer_gacha`, `winter_event`, `limited_2024`, etc.

**Q: Phải có bao nhiêu swimsuit trong folder?**
A: Ít nhất 1 ảnh swimsuit. Automation sẽ check tất cả swimsuit trong folder.

**Q: Có thể pull nhiều banner cùng lúc?**
A: Có, thêm tất cả banner vào list rồi Start

**Q: Làm sao để automation tìm đúng swimsuit?**
A: Chụp ảnh swimsuit rõ ràng, đủ lớn để template matching chính xác

**Q: Multi pull khác gì Single pull?**
A: Multi = 10 pulls 1 lần, Single = từng pull 1

