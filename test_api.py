import requests
from pathlib import Path
import mimetypes
import time
from datetime import datetime

start_time = time.time()
path_face = "image/anh_truoc/_DC_0948.jpg"
path_hair = "image/anh_face/_DC_0920.jpg"

mime_face = mimetypes.guess_type(path_face)[0] or 'application/octet-stream'
mime_hair = mimetypes.guess_type(path_hair)[0] or 'application/octet-stream'

data = {
    'random_seed': -1,
    'step': 20,
    'guidance_scale': 1.5,
    'controlnet_conditioning_scale': 1.0,
    'scale': 1.2,
    'size': 512
}

with open(path_face, 'rb') as f_face, open(path_hair, 'rb') as f_hair:
    files = {
        'source_image': (Path(path_face).name, f_face, mime_face),
        'reference_image': (Path(path_hair).name, f_hair, mime_hair)
    }
    response = requests.post("http://localhost:8000/transfer-hair", files=files, data=data)
    try:
        print(response.json())
    except Exception as e:
        print("Error parsing JSON:", e)
        print("Response content:", response.text)

end_time = time.time()  # End tracking
print(f"\nTổng thời gian thưc thi: {end_time - start_time:.2f} seconds")

