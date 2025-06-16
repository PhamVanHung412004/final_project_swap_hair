# Ứng dụng Ghép ảnh AI

Ứng dụng ghép ảnh sử dụng trí tuệ nhân tạo với giao diện web thân thiện.

## Yêu cầu hệ thống

- Python 3.10
- Conda (Anaconda hoặc Miniconda)
- Git

## Hướng dẫn cài đặt

### Bước 1: Clone repository

```bash
git clone https://github.com/PhamVanHung412004/final_project_swap_hair.git
cd final_project_swap_hair
```

### Bước 2: Tạo môi trường conda

```bash
# Tạo môi trường mới với Python 3.9
conda create -n image_blend python=3.9

# Kích hoạt môi trường
conda activate image_blend
```

### Bước 3: Cài đặt các package cần thiết

```bash
# Cài đặt các thư viện từ requirements.txt
pip install -r requirements.txt
```

### Bước 4: Tải model AI

```bash
# Tải model cần thiết cho ứng dụng
python download_model.py
```

## Chạy ứng dụng

### Khởi động giao diện web

```bash
# Chạy ứng dụng Streamlit
streamlit run app.py
```

Sau khi chạy lệnh trên, ứng dụng sẽ tự động mở trong trình duyệt tại địa chỉ `http://localhost:8501`

## Các lệnh conda hữu ích

```bash
# Xem danh sách môi trường
conda env list

# Kích hoạt môi trường
conda activate image_blend

# Thoát môi trường
conda deactivate

# Xóa môi trường (nếu cần)
conda env remove -n image_blend

# Cập nhật conda
conda update conda
```

## Gỡ lỗi thường gặp

### Lỗi không tìm thấy module
```bash
# Đảm bảo đã kích hoạt đúng môi trường
conda activate image_blend

# Cài đặt lại requirements
pip install -r requirements.txt
```