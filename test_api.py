import requests
from pathlib import Path
import mimetypes

# print(Path(__file__).parent)

# Đường dẫn
path_face = "image/anh_truoc/_DC_0947.jpg"
path_hair = "image/anh_face/_DC_0882.jpg"

# # Tự động lấy MIME type từ đuôi file
# mime_face = mimetypes.guess_type(path_face)[0] or 'application/octet-stream'
# mime_hair = mimetypes.guess_type(path_hair)[0] or 'application/octet-stream'

files = {
    'source_image': open(path_face, 'rb'),      # Khuôn mặt cần ghép tóc
    'reference_image': open(path_hair, 'rb')    # Ảnh có tóc đẹp
}

data = {
    'random_seed': -1,
    'step': 20,
    'guidance_scale': 1.5,
    'controlnet_conditioning_scale': 1.0,
    'scale': 1.5,
    'size': 512
}

response = requests.post("http://localhost:8000/transfer-hair", files=files, data=data)
print(response.json())

