# Hướng Dẫn Sử Dụng Auto C-Peach

## Giới Thiệu

Auto C-Peach là công cụ tự động hóa game DOAX Venus Vacation, hỗ trợ:
- **Festival Automation**: Tự động chơi các màn Festival
- **Gacha Automation**: Tự động quay gacha
- **Hopping Automation**: Tự động Pool Hopping

---

## Khởi Động

1. Mở game DOAX Venus Vacation
2. Chạy Auto C-Peach
3. Nhấn **"Connect Device"** ở góc trên bên phải
4. Chờ trạng thái hiển thị **"Device Connected"** (màu xanh)

---

## Tab Festival Automation

### Cơ Chế Hoạt Động

Tool sẽ tự động thực hiện quy trình sau cho mỗi stage:
1. Chạm vào nút Event để mở menu Festival
2. Chụp ảnh màn hình trước khi chọn stage
3. Sử dụng OCR để tìm và chạm vào tên Festival trong danh sách
4. Sử dụng OCR để tìm và chạm vào Rank tương ứng
5. Chụp ảnh màn hình sau khi chọn
6. Xác minh thông tin hiển thị (điểm, rank, phần thưởng) với dữ liệu CSV
7. Chạm nút Challenge để bắt đầu trận đấu
8. Tự động kéo thả object nếu có (mini-game)
9. Chạm các nút OK, Skip để hoàn thành trận
10. Chụp ảnh kết quả và xác minh phần thưởng nhận được

**Tính năng Resume**: Nếu bị gián đoạn (tắt app, lỗi mạng...), tool sẽ tự động lưu tiến trình và cho phép tiếp tục từ stage đang dở khi chạy lại.

**Fallback Cache**: Nếu OCR không nhận diện được text (do text quá dài, bị cắt...), tool sẽ sử dụng vị trí đã lưu từ lần chạm thành công trước đó.

### Chuẩn Bị
Chuẩn bị file CSV/JSON chứa thông tin các stage Festival cần chạy.

### Các Bước Thực Hiện

1. **Chọn File Dữ Liệu**
   - Nhấn **"Browse"** để chọn file dữ liệu
   - Nhấn **"Preview"** để xem trước nội dung

2. **Chọn Stage Bắt Đầu**
   - Chọn stage muốn bắt đầu từ dropdown "Start Stage"
   - Hiển thị format: `[Rank] Tên Festival - Rank Festival`

3. **Cấu Hình Output** (tùy chọn)
   - Nhấn **"..."** để chọn nơi lưu kết quả
   - Bỏ trống để tự động tạo file với timestamp

4. **Tùy Chọn Resume**
   - Mặc định sẽ tự động tiếp tục session bị gián đoạn
   - Tick **"Force New Session"** để bắt đầu mới hoàn toàn

5. **Bắt Đầu**
   - Nhấn **"Start"** để chạy
   - Nếu có session cũ, sẽ hiện hộp thoại hỏi có muốn tiếp tục không
   - Theo dõi tiến trình qua Progress Panel

6. **Dừng**
   - Nhấn **"Stop"** hoặc phím tắt `Ctrl+Q` / `ESC` / `F9`

---

## Tab Gacha Automation

### Cơ Chế Hoạt Động

Tool sẽ tự động thực hiện quy trình sau cho mỗi lần pull:
1. Tìm banner trong màn hình bằng template matching (so khớp ảnh)
2. Nếu không thấy, tự động cuộn xuống và tìm tiếp (tối đa 10 lần)
3. Chạm vào banner để mở
4. Chọn loại pull (Single hoặc Multi 10x)
5. Chụp ảnh trước khi pull
6. Xác nhận pull và chờ animation
7. Tự động skip animation nếu có thể
8. Chụp ảnh kết quả
9. Xác minh kết quả:
   - Kiểm tra có đúng rarity (SSR/SR) không bằng template matching
   - Kiểm tra có swimsuit character không bằng template matching
   - Nếu cả hai đều match → Lưu ảnh đặc biệt đánh dấu "SPECIAL"

**Template Matching**: Tool so sánh ảnh màn hình với các ảnh mẫu trong folder banner để nhận diện. Độ chính xác phụ thuộc vào chất lượng ảnh mẫu.

### Chuẩn Bị
Tổ chức thư mục templates với các folder banner:
- Mỗi folder là một banner
- Chứa ảnh banner chính để nhận diện
- Chứa các ảnh character swimsuit để xác minh kết quả

### Các Bước Thực Hiện

1. **Chọn Thư Mục Templates**
   - Nhấn **"Browse"** để chọn thư mục templates
   - Nhấn **"Refresh"** để tải danh sách banner
   - Mỗi banner hiển thị preview và số lượng ảnh trong folder

2. **Cấu Hình Pull**
   - **Rarity**: Chọn SSR hoặc SR để xác minh kết quả
   - **Pulls**: Nhập số lần pull (1-100)
   - **Type**: Chọn Single (1 lần) hoặc Multi (10 lần)

3. **Thêm Banner Vào Queue**
   - Nhấn **"Add to Queue"** trên banner muốn quay
   - Nếu folder có nhiều ảnh, chọn ảnh banner chính
   - Xem tổng số banner và pull trong Queue

4. **Quản Lý Queue**
   - **Edit**: Chỉnh sửa số pull, type, rarity của banner
   - **Remove**: Xóa banner khỏi queue
   - **Clear**: Xóa toàn bộ queue

5. **Bắt Đầu**
   - Nhấn **"Start Gacha"**
   - Xác nhận số lượng gacha và pull trong hộp thoại

6. **Dừng**
   - Nhấn **"Stop"** hoặc phím tắt

---

## Tab Hopping Automation

### Cơ Chế Hoạt Động

Tool sẽ tự động thực hiện quy trình sau cho mỗi course:
1. Chụp ảnh màn hình trước khi sử dụng item
2. Chạm nút Use để sử dụng item
3. Chạm OK để xác nhận
4. Chụp ảnh item nhận được
5. Sử dụng OCR để đọc tên item và số lượng
6. So sánh với dữ liệu trong CSV để xác minh đúng/sai
7. Ghi kết quả vào file output

**Tính năng Resume**: Tương tự Festival, tool lưu tiến trình và cho phép tiếp tục từ course đang dở.

**Xác minh OCR**: Tool đọc text trên màn hình và so sánh với dữ liệu mong đợi trong CSV. Hỗ trợ fuzzy matching (so khớp mờ) để xử lý các trường hợp OCR đọc sai một vài ký tự.

### Chuẩn Bị
Chuẩn bị file CSV/JSON chứa thông tin các course cần chạy.

### Các Bước Thực Hiện

1. **Chọn File Dữ Liệu**
   - Nhấn **"Browse"** để chọn file
   - Nhấn **"Preview"** để xem trước

2. **Chọn Course Bắt Đầu**
   - Chọn course từ dropdown "Start Course"

3. **Cấu Hình Output** (tùy chọn)
   - Nhấn **"..."** để chọn nơi lưu kết quả

4. **Tùy Chọn Resume**
   - Tick **"Force New Session"** nếu muốn bắt đầu mới

5. **Bắt Đầu**
   - Nhấn **"Start"**

6. **Dừng**
   - Nhấn **"Stop"** hoặc phím tắt

---

## Tab Settings

- **Log Level**: Điều chỉnh mức độ chi tiết của log
  - DEBUG: Hiển thị tất cả thông tin chi tiết
  - INFO: Thông tin chung (khuyến nghị)
  - WARNING: Chỉ hiện cảnh báo
  - ERROR: Chỉ hiện lỗi

- **Max Log Lines**: Số dòng log tối đa hiển thị (giảm để tăng hiệu suất)
- **Log Poll Interval**: Tần suất cập nhật log (tăng để giảm tải CPU)

---

## Phím Tắt

| Phím | Chức Năng |
|------|-----------|
| `Ctrl+Q` | Dừng automation |
| `ESC` | Dừng khẩn cấp |
| `F9` | Dừng automation |

---

## Kết Quả & Ảnh Chụp

Tất cả kết quả được lưu trong thư mục `result/`:
- **Snapshots**: Ảnh chụp màn hình tại các bước quan trọng
- **Results**: File CSV/JSON chứa kết quả xác minh
- **Logs**: File log chi tiết quá trình chạy

Mỗi automation có thư mục riêng (festival/, gacha/, hopping/) để dễ quản lý.

---

## Lưu Ý Quan Trọng

- **Không di chuyển hoặc minimize** cửa sổ game khi đang chạy
- **Giữ game ở foreground** để tool có thể chụp màn hình và điều khiển
- Sử dụng **Preview** để kiểm tra dữ liệu trước khi chạy
- Nếu bị treo, dùng **ESC** để dừng khẩn cấp
- Kiểm tra **Activity Log** ở cuối màn hình để theo dõi trạng thái và lỗi
- Có thể kéo thanh phân cách giữa tab và log để điều chỉnh kích thước
