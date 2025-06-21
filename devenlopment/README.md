# Ứng dụng Ghép ảnh AI

Ứng dụng ghép ảnh sử dụng trí tuệ nhân tạo với giao diện web thân thiện.

## Hướng dẫn cài đặt

### Bước 1: Clone repository

```bash
git clone https://github.com/PhamVanHung412004/final_project_swap_hair.git
cd final_project_swap_hair
```



### Bước 2: Cài đặt các package cần thiết
 

```bash
# Tạo môi trường với conda (yêu cầu Miniconda or Anaconda)
conda create -n tên_môi_trường python=3.10
```

```bash
# Cài đặt các thư viện từ requirements.txt
pip install -r requirements.txt
```

### Bước 3: Tải model AI

```bash
# Tải model cần thiết cho ứng dụng
python download_model.py
```

## Chạy ứng dụng

### Khởi động giao diện web

```bash
# KHởi dộng server với Flask
python api_server.py
```
### Test API 
```bash
# Sử dụng câu lệnh sau để test API
python test_api.py
```

Sau khi chạy lệnh trên, ứng dụng sẽ tự động mở trong trình duyệt tại địa chỉ: `http://localhost:8000`
