import torch
from PIL import Image
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import os
import cv2
from diffusers import DDIMScheduler, UniPCMultistepScheduler
from diffusers.models import UNet2DConditionModel
from ref_encoder.latent_controlnet import ControlNetModel
from ref_encoder.adapter import *
from ref_encoder.reference_unet import ref_unet
from utils.pipeline import StableHairPipeline
from utils.pipeline_cn import StableDiffusionControlNetPipeline
from rembg import remove  # pip install rembg

def concatenate_images(image_files, output_file, type="pil"):
    if type == "np":
        image_files = [Image.fromarray(img) for img in image_files]
    images = image_files  # list
    max_height = max(img.height for img in images)
    images = [img.resize((img.width, max_height)) for img in images]
    total_width = sum(img.width for img in images)
    combined = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for img in images:
        combined.paste(img, (x_offset, 0))
        x_offset += img.width
    combined.save(output_file)

class StableHair:
    def __init__(self, config="stable_hair/configs/hair_transfer.yaml", device="cpu", weight_dtype=torch.float32) -> None:
        print("Initializing Stable Hair Pipeline...")
        self.config = OmegaConf.load(config)
        self.device = device

        ### Load controlnet
        unet = UNet2DConditionModel.from_pretrained(self.config.pretrained_model_path, subfolder="unet").to(device)
        controlnet = ControlNetModel.from_unet(unet).to(device)
        _state_dict = torch.load(os.path.join(self.config.pretrained_folder, self.config.controlnet_path),  map_location=torch.device('cpu'))
        controlnet.load_state_dict(_state_dict, strict=False)
        controlnet.to(weight_dtype)

        ### >>> create pipeline >>> ###
        self.pipeline = StableHairPipeline.from_pretrained(
            self.config.pretrained_model_path,
            controlnet=controlnet,
            safety_checker=None,
            torch_dtype=weight_dtype,
        ).to(device)
        self.pipeline.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)

        # turn on no-grad defaults
        # self.pipeline.enable_attention_slicing()
        # self.pipeline.enable_vae_slicing()
        # self.pipeline.enable_model_cpu_offload()
        # self.pipeline.enable_sequential_cpu_offload()

        

        ### load Hair encoder/adapter
        self.hair_encoder = ref_unet.from_pretrained(self.config.pretrained_model_path, subfolder="unet").to(device)
        _state_dict = torch.load(os.path.join(self.config.pretrained_folder, self.config.encoder_path), map_location=torch.device('cpu'))
        self.hair_encoder.load_state_dict(_state_dict, strict=False)
        self.hair_adapter = adapter_injection(self.pipeline.unet, device=self.device, dtype=torch.float32, use_resampler=False)
        _state_dict = torch.load(os.path.join(self.config.pretrained_folder, self.config.adapter_path),  map_location=torch.device('cpu'))
        self.hair_adapter.load_state_dict(_state_dict, strict=False)

        ### load bald converter
        bald_converter = ControlNetModel.from_unet(unet).to(device)
        _state_dict = torch.load(self.config.bald_converter_path,  map_location=torch.device('cpu'))
        bald_converter.load_state_dict(_state_dict, strict=False)
        bald_converter.to(dtype=weight_dtype)
        del unet

        ### create pipeline for hair removal
        self.remove_hair_pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            self.config.pretrained_model_path,
            controlnet=bald_converter,
            safety_checker=None,
            torch_dtype=weight_dtype,
        )
        self.remove_hair_pipeline.scheduler = DDIMScheduler.from_config(
            self.remove_hair_pipeline.scheduler.config)
        self.remove_hair_pipeline = self.remove_hair_pipeline.to(device)

        # For the hair removal pipeline, add the same optimizations:
        # self.remove_hair_pipeline.enable_attention_slicing()
        # self.remove_hair_pipeline.enable_vae_slicing()

        ### move to fp16
        self.hair_encoder.to(weight_dtype)
        self.hair_adapter.to(weight_dtype)

        print("Initialization Done!")

    def Hair_Transfer(self, source_image, reference_image, random_seed, step, guidance_scale, scale, controlnet_conditioning_scale, size):
        with torch.inference_mode():
            prompt = ""
            n_prompt = ""
            random_seed = int(random_seed)
            step = int(step)
            guidance_scale = float(guidance_scale)
            scale = float(scale)

            # load imgs
            source_image = Image.open(source_image).convert("RGB").resize((size, size))
            id = np.array(source_image)
            reference_image = np.array(Image.open(reference_image).convert("RGB").resize((size, size)))
            source_image_bald = np.array(self.get_bald(source_image, scale=0.9))
            H, W, C = source_image_bald.shape

            # generate images
            set_scale(self.pipeline.unet, scale)
            generator = torch.Generator(device="cpu")
            generator.manual_seed(random_seed)
            sample = self.pipeline(
                prompt,
                negative_prompt=n_prompt,
                num_inference_steps=step,
                guidance_scale=guidance_scale,
                width=W,
                height=H,
                controlnet_condition=source_image_bald,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                generator=generator,
                reference_encoder=self.hair_encoder,
                ref_image=reference_image,
            ).samples
        return sample

    def get_bald(self, id_image, scale):
        H, W = id_image.size
        scale = float(scale)
        image = self.remove_hair_pipeline(
            prompt="",
            negative_prompt="",
            num_inference_steps=20,
            guidance_scale=1.5,
            width=W,
            height=H,
            image=id_image,
            controlnet_conditioning_scale=scale,
            generator=None,
        ).images[0]

        return image

# import time
# from datetime import datetime
# if __name__ == '__main__':
#     start_time = time.time()  # Start tracking
#     model = StableHair(config="../configs/hair_transfer.yaml", weight_dtype=torch.float32)
#     kwargs = OmegaConf.to_container(model.config.inference_kwargs)
#     print("Starting hair transfer:")
#     start_time_2 = time.time()
#     # import torch.profiler
#     # with torch.profiler.profile(
#     #     activities=[torch.profiler.ProfilerActivity.CPU],
#     #     record_shapes=True
#     #     ) as prof:
#     #     with torch.inference_mode():
#     #         _ = model.Hair_Transfer(**kwargs)
#     # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
#     data_new = {     
#         "source_image": r"E:\final_project_test_hair\image\anh_truoc\_DC_0946.jpg",
#         "reference_image": r"E:\final_project_test_hair\image\anh_face\_DC_0885.jpg",
#         "random_seed": -1,
#         "step": 20,   
#         "guidance_scale": 1.5,
#         "controlnet_conditioning_scale": 1,
#         "scale": 1.2,
#         "size" : 128

#     }

#     id, image, source_image_bald, reference_image = model.Hair_Transfer(**data_new)
#     os.makedirs(model.config.output_path, exist_ok=True)
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#     model.config.save_name = f"{timestamp}_{model.config.save_name}"
#     output_file = os.path.join(model.config.output_path, model.config.save_name)
#     concatenate_images([id, source_image_bald, reference_image, (image*255.).astype(np.uint8)], output_file=output_file, type="np")
#     image_result = (image*255.).astype(np.uint8)
#     image_result = cv2.cvtColor(image_result, cv2.COLOR_BGR2RGB)
#     id = cv2.cvtColor(id, cv2.COLOR_BGR2RGB)
#     source_image_bald = cv2.cvtColor(source_image_bald, cv2.COLOR_BGR2RGB)
#     reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)

#     cv2.imwrite("./result/anh_goc.jpg",id)
#     cv2.imwrite("./result/anh_dau_troc.jpg",source_image_bald)
#     cv2.imwrite("./result/anh_toc_se_ghep.jpg",reference_image)
#     cv2.imwrite("./result/anh_ket_qua.jpg",image_result)

#     end_time = time.time()  # End tracking
#     print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
