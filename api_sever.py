from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import io
import uuid
from PIL import Image
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
import torch
from solution import StableHair
import cv2

# Set CPU optimizations
torch.set_num_threads(os.cpu_count())  # Use all CPU cores
torch.backends.quantized.engine = 'fbgemm'
os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
path = str(Path(__file__).parent / "configs/hair_transfer.yaml")

# Khởi tạo Flask app
app = Flask(__name__)
CORS(app)

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

path = str(Path(__file__).parent / "configs/hair_transfer.yaml")

model = StableHair(config=path, weight_dtype=torch.float32)

# ==============================================

# Tạo thư mục lưu trữ nếu chưa tồn tại
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Cấu hình giới hạn file upload
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Danh sách định dạng file được phép
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed_file(filename):
    """Kiểm tra file có định dạng hợp lệ"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_uploaded_file(file, prefix=''):
    print("File", file)
    """Lưu file upload và trả về đường dẫn"""
    if file and allowed_file(file.filename):
        ext = file.filename.rsplit('.', 1)[1].lower()
        filename = f"{prefix}_{uuid.uuid4()}.{ext}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        return filepath
    return None



@app.route('/transfer-hair', methods=['POST'])
def transfer_hair():
    try:
        
        if 'source_image' not in request.files:
            return jsonify({'error': 'Missing source_image file'}), 400
        if 'reference_image' not in request.files:
            return jsonify({'error': 'Missing reference_image file'}), 400
        
        source_file = request.files['source_image']
        reference_file = request.files['reference_image']
        
        if source_file.filename == '':
            return jsonify({'error': 'No source image selected'}), 400
        if reference_file.filename == '':
            return jsonify({'error': 'No reference image selected'}), 400
        
        source_path = save_uploaded_file(source_file, 'source')
        reference_path = save_uploaded_file(reference_file, 'reference')
        
        if not source_path or not reference_path:
            return jsonify({'error': 'Invalid file format'}), 400
        
        params = {
            "source_image": source_path,
            "reference_image": reference_path,
            "random_seed": int(request.form.get("random_seed", -1)),
            "step": int(request.form.get("step", 20)),
            "guidance_scale": float(request.form.get("guidance_scale", 1.5)),
            "controlnet_conditioning_scale": float(request.form.get("controlnet_conditioning_scale", 1.0)),
            "scale": float(request.form.get("scale", 1.5)),
            "size": int(request.form.get("size", 512))
        }
        
        logger.info(f"Processing with params: {params}")
        
        result_img = model.Hair_Transfer(**params)  # Giả sử trả numpy array float
        
        result_img = (result_img * 255.).astype(np.uint8)
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        
        result_filename = f"result_{uuid.uuid4()}.png"
        result_filepath = os.path.join(RESULT_FOLDER, result_filename)
        cv2.imwrite(result_filepath, result_img)
        
        return send_file(
            result_filepath,
            mimetype='results/png',
            as_attachment=True,
            download_name=f'hair_transfer_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        )
    except Exception as e:
        logger.error(f"API Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

    finally:
        # Cleanup files sau khi xử lý (optional)
        try:
            if 'source_path' in locals() and os.path.exists(source_path):
                os.remove(source_path)
            if 'reference_path' in locals() and os.path.exists(reference_path):
                os.remove(reference_path)
        except:
            pass

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/', methods=['GET'])
def index():
    """Root endpoint với thông tin API"""
    return jsonify({
        'service': 'Hair Transfer API',
        'version': '1.0.0',
        'endpoints': {
            '/transfer-hair': 'POST - Transfer hair from reference to source image',
            '/health': 'GET - Health check'
        },
        'parameters': {
            'source_image': 'File - Face image to add hair to',
            'reference_image': 'File - Image with desired hair style',
            'random_seed': 'Integer - Random seed for reproducibility',
            'step': 'Integer - Number of processing steps',
            'guidance_scale': 'Float - Guidance scale for generation',
            'scale': 'Float - Scale factor',
            'controlnet_conditioning_scale': 'Float - ControlNet conditioning scale',
            'size': 'Integer - Output image size'
        }
    })

if __name__ == '__main__':
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    app.run(host='0.0.0.0', port=8000, debug=True, use_reloader=False)
