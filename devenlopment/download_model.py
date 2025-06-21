import gdown
from pathlib import Path
output_directory = str(Path(__file__).parent / "models")

# Tải toàn bộ folder vào thư mục chỉ định
folder_url = "https://drive.google.com/drive/folders/1E-8Udfw8S8IorCWhBgS4FajIbqlrWRbQ"
gdown.download_folder(folder_url, output=output_directory, quiet=False, use_cookies=False)