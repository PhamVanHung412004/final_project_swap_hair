import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import base64
import torch
from typing import Tuple
from solution import StableHair
from pathlib import Path

path = str(Path(__file__).parent / "configs/hair_transfer.yaml")
@st.cache_resource
def load_model():
    model = StableHair(config=path, weight_dtype=torch.float32)
    return model

# Set page configuration
st.set_page_config(
    page_title="Ứng dụng Ghép Tóc",
    page_icon="💇",
    layout="wide"
)



def main():
    st.title("💇 Ứng dụng Ghép Tóc")
    st.markdown("Ghép tóc từ ảnh này sang ảnh khác bằng công nghệ thị giác máy tính")
    
    # Initialize hair transfer processor
    # hair_processor = HairTransfer()
    
    # Create two columns for image uploads
    size = st.selectbox(
            "Size image",
            options=[128,256,512, 768, 1024],
            index=0,
            help="Chọn kích thước"
        )
    col1, col2 = st.columns(2)
    
    # Initialize variables
    source_image = None
    target_image = None
    
    with col1:
        st.subheader("Ảnh của bạn")
        st.markdown("Tải lên ảnh mà bạn muốn ghép tóc")
        source_file = st.file_uploader(
            "Chọn ảnh nguồn...",
            type=['jpg', 'jpeg', 'png'],
            key="source"
        )
        
        if source_file is not None:
            # QUAN TRỌNG: Reset file pointer
            source_file.seek(0)
            source_image = Image.open(source_file)
            
            # Chuyển về RGB nếu cần
            if source_image.mode != 'RGB':
                source_image = source_image.convert('RGB')
            source_image = source_image.resize((int(size),int(size)))

            st.image(source_image, caption="Ảnh Nguồn", use_container_width=True)

    with col2:
        st.subheader("Ảnh có kiểu tóc")
        st.markdown("Tải lên ảnh tóc mà bạn muốn ghép")
        target_file = st.file_uploader(
            "Chọn ảnh đích...",
            type=['jpg', 'jpeg', 'png'],
            key="target"
        )
        
        if target_file is not None:
            # QUAN TRỌNG: Reset file pointer
            target_file.seek(0)
            target_image = Image.open(target_file)
            
            # Chuyển về RGB nếu cần
            if target_image.mode != 'RGB':
                target_image = target_image.convert('RGB')

            target_image = target_image.resize((int(size),int(size)))
                
            st.image(target_image, caption="Ảnh Đích", use_container_width=True)    

    # Always show processing options
    st.markdown("---")
    st.subheader("⚙️ Tùy Chỉnh Thông Số")
    model = load_model()
    col3, col4 = st.columns(2)
    col5, col6 = st.columns(2)
    
    with col3:
        random_seed = st.slider(
            "random_seed",
            min_value=-2.0,
            max_value=2.0,
            value=-1.0,
            step=0.1,
            help=""
        )
    
    with col4:
        guidance_scale = st.slider(
            "guidance_scale",
            min_value=0.0,
            max_value=3.0,
            value=1.5,
            step=0.1,
            help=""
        )

    
    with col5:
        controlnet_conditioning_scale = st.slider(
            "controlnet_conditioning_scale",
            min_value=0.0,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help=""
        )
        
    
    # Advanced settings in an expander
    with st.expander("🔧 Cài Đặt Nâng Cao"):
        col7, col8 = st.columns(2)
        
        with col7:
            step = st.slider(
                "step",
                min_value=1,
                max_value=50,
                value=20,
                step=1,
                help=""
            )
        
        with col8:
            scale = st.slider(
                "scale",
                min_value=0.0,
                max_value=2.0,
                value=1.2,
                step=0.1,
                help=""
            )
                
    # Processing section
    if source_file is not None and target_file is not None:
        # Process button
        if st.button("🎨 Ghép Tóc", type="primary"):
            # try:
            with st.spinner(f"Đang xử lý ảnh với {step} bước... Vui lòng đợi một chút."):
                # Validate images
                if source_image is None or target_image is None:
                    st.error("Vui lòng tải lên cả hai ảnh trước khi xử lý")
                    return
                
                source_cv = source_image.convert("RGB").resize((int(size),int(size)))
                target_cv = target_image.convert("RGB").resize((int(size),int(size)))
                
                # Perform hair transfer
                data_new = {     
                    "source_image": source_cv,
                    "reference_image": target_cv,
                    "random_seed": random_seed,
                    "step": step,
                    "guidance_scale": guidance_scale,
                    "controlnet_conditioning_scale": controlnet_conditioning_scale,
                    "scale": scale,
                    "size" : size
                }

                image = model.Hair_Transfer(**data_new)
                print(type(image))
                image_result = (image*255.).astype(np.uint8)
                image_result = cv2.cvtColor(image_result, cv2.COLOR_BGR2RGB)
                # if image_result is not None:
                #     # Convert back to PIL format for display
                try:
                    result_pil = Image.fromarray(cv2.cvtColor(image_result, cv2.COLOR_BGR2RGB))
                    # Hiển thị kết quả
                    st.success("Ghép tóc thành công!")
                    _, result, __ = st.columns([1,2,1])
                    with result:
                        st.image(result_pil, caption="Kết quả", use_container_width=True)

                        # Download button
                        buf = io.BytesIO()
                        result_pil.save(buf, format='PNG')
                        byte_im = buf.getvalue()
                        
                        st.download_button(
                            label="📥 Tải Xuống Kết Quả",
                            data=byte_im,
                            file_name="ket_qua_ghep_toc.png",
                            mime="image/png"
                        )

                except Exception as e:
                    st.error(f"Lỗi chuyển đổi kết quả: {str(e)}")
                    st.write(f"Result type: {type(image_result)}")
                    
    else:
        st.info("👆 Vui lòng tải lên cả hai ảnh để bắt đầu ghép tóc")
    
if __name__ == "__main__":
    main()
