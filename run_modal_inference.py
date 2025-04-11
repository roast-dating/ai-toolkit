import os
import modal
import sys
import yaml
import argparse
from typing import Dict
import torch
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, snapshot_download
from modal import fastapi_endpoint
from fastapi import Request
from utils import utils
from src.image import create_image_remix
from src.volumes import setup_volumes
from pathlib import Path
import boto3
from dotenv import dotenv_values
from rdbc.utils import enums
import random

sys.path.insert(0, "/root/ai-toolkit")
# Create image and volumes
image = create_image_remix()
model_volume, MOUNT_DIR_MODELS, outputs_volume, MOUNT_DIR_OUTPUTS, MOUNT_PHOTOSHOOT_BUCKET_DIR, MOUNT_DIR_PHOTOSHOOT_MODELS = setup_volumes()
env_dict = dotenv_values("/Users/benoitbaylin/Documents/code/ai-toolkit/.env")
MOUNT_DIR_OUTPUTS = "/root/ai-toolkit/outputs"
MOUNT_DIR_MODELS = "/root/ai-toolkit/models"
MOUNT_DIR_PHOTOSHOOT_TECH = "/root/ai-toolkit/roast-photoshoot-ai-tech"
MOUNT_ROAST_IMAGE_BUCKET_DIR = Path("/roast-images")
print(f"Env dict: {env_dict}")
# Setup code mount
# code_mount = modal.Mount.from_local_dir(
#     str(Path.cwd()), 
#     remote_path="/root/ai-toolkit"
# )
image.add_local_dir("/Users/benoitbaylin/Documents/code/ai-toolkit", remote_path="/root/ai-toolkit")
image.add_local_python_source("main_debug", "remix_app", "src", "utils")
# Setup AWS secret
aws_secret = modal.Secret.from_dict({
    "AWS_ACCESS_KEY_ID": os.environ.get('AWS_ACCESS_KEY_ID'),
    "AWS_SECRET_ACCESS_KEY": os.environ.get('AWS_SECRET_ACCESS_KEY'),
    "AWS_REGION": os.environ.get('AWS_REGION')
})


app = modal.App(
    name="inference-flux",
    image=image,
    mounts=[],
    secrets=[
        modal.Secret.from_name("roast-aws-secret"),
        modal.Secret.from_name("backend-keys"),
        aws_secret
    ]
)

@app.function(
    timeout=7200,  # 2 hours, increase or decrease if needed
    image=image,
    secrets=[modal.Secret.from_name("roast-aws-secret"), modal.Secret.from_name("backend-keys")],
    volumes={MOUNT_DIR_OUTPUTS: outputs_volume,
            MOUNT_DIR_PHOTOSHOOT_MODELS: modal.CloudBucketMount(
            bucket_name="roast-photoshoot-ai-models",
            secret=aws_secret,
            read_only=False
        ),
            MOUNT_PHOTOSHOOT_BUCKET_DIR: modal.CloudBucketMount(
            bucket_name="roast-photoshoot-ai",
            secret=aws_secret,
            read_only=False
        ),
            MOUNT_ROAST_IMAGE_BUCKET_DIR: modal.CloudBucketMount(
            bucket_name="roast-images-ai",
            secret=aws_secret,
            read_only=False)
    },
)
@fastapi_endpoint(method="POST")
async def infer_user_endpoint(request: Request):
    print("Inference request: ", request)
    data = await request.json()
    user_id = data['user_id']
    model_id = data['model_id']
    photoshoot_id = data['photoshoot_id']
    mode = data.get('mode', 'flux')
    print("Starting inference")
    infer_user.spawn(user_id=user_id, model_id=model_id,
                      photoshoot_id=photoshoot_id, mode=mode)
    return {"status": "success", "user_id": user_id,
             "model_id": model_id, "mode": mode}


@app.function(
    gpu="A100",  # gpu="H100"
    timeout=7200,  # 2 hours, increase or decrease if needed
    image=image,
    secrets=[modal.Secret.from_name("roast-aws-secret")],
    volumes={MOUNT_DIR_OUTPUTS: outputs_volume,
             MOUNT_DIR_MODELS: model_volume,
            MOUNT_DIR_PHOTOSHOOT_TECH: modal.CloudBucketMount(
            bucket_name="roast-photoshoot-ai-tech",
            secret=aws_secret,
            read_only=False
        ),
            MOUNT_DIR_PHOTOSHOOT_MODELS: modal.CloudBucketMount(
            bucket_name="roast-photoshoot-ai-models",
            secret=aws_secret,
            read_only=False
        ),
            MOUNT_PHOTOSHOOT_BUCKET_DIR: modal.CloudBucketMount(
            bucket_name="roast-photoshoot-ai",
            secret=aws_secret,
            read_only=False
        ),
            MOUNT_ROAST_IMAGE_BUCKET_DIR: modal.CloudBucketMount(
            bucket_name="roast-images-ai",
            secret=aws_secret,
            read_only=False)

         },
         max_containers=20
)
def infer_user(user_id: str, model_id: str,
                photoshoot_id: str, mode: str,
                  test_pics: bool = False, inference_mode: str = "prod"):
    # Load the config file
    print(f"Inside infer_user: {mode}")
    config_path = "/root/ai-toolkit/config/flux_inference.yaml"

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Update the config with the specific model_id
    config['run_name'] = model_id
    if mode == "flux":
        config['loras'][mode]['user']['path'] = config['loras'][mode]['user']['path'].format(user_id, model_id, model_id)
    else:
        model_name = f"lora_xl_{user_id}_{model_id}_udaxihhe.safetensors"
        config['loras'][mode]['user']['path'] = os.path.join(MOUNT_PHOTOSHOOT_BUCKET_DIR, user_id, "models", model_id, model_name)
    config['mode'] = mode
    # Run inference
    run_inference(config, user_id=user_id,
                   model_id=model_id,
                     photoshoot_id=photoshoot_id,
                       test_pics=test_pics,
                       inference_mode=inference_mode)

    # Commit the output volume after inference
    # outputs_volume.commit()
    # model_volume.commit()

    print(f"Inference completed for user {user_id}, model {model_id}, photoshoot {photoshoot_id}")

def pil_to_tensor(image):
    """Convert PIL image to normalized tensor"""
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    return transform(image)

def run_inference(config, user_id: str, model_id: str, 
                  photoshoot_id: str, test_pics: bool = False,
                  inference_mode: str = "prod"):
    import diffusers
    print(f"Diffusers version: {diffusers.__version__}")
    from diffusers import FluxPipeline, StableDiffusionXLPipeline, FluxControlNetPipeline, FluxControlNetModel
    from rdbc.db.mongodb.mongo_engine import MongoDBEngine
    from controlnet_aux import CannyDetector
    from diffusers.utils import load_image
    from PIL import Image
    from pruna_pro import smash, SmashConfig
    from rdbc.utils import enums

    try:
        print(f"Inside infer_user: {os.environ.get('MONGODB_URI')}")
        print(f"Inside infer_user: {os.environ.get('HF_TOKEN')}")
        print(f"Inside run inference: test pics: {test_pics}")
        engine = MongoDBEngine(uri=os.environ.get("MONGODB_URI"))
        photoshoot = engine.repository("Photoshoot").get_photoshoot(id=photoshoot_id)
        photoshoot.update(status="shooting")

        try:
            user = engine.repository("User").get_user(id=photoshoot.user_id)
        except Exception as e:
            print(f"Error getting user: {e} by id, trying firebase: {photoshoot.user_firebase_id}")
            user = engine.repository("User").get_user(user_firebase_id=photoshoot.user_firebase_id)
        
        username = "udaxihhe"
        user_info = photoshoot.user_info
        user_info.update({"username": username, "gender": photoshoot.gender, "ethnicity": photoshoot.ethnicity})
        # canny_detector = CannyDetector()

        # controlnet_model_canny = FluxControlNetModel.from_pretrained("InstantX/FLUX.1-dev-controlnet-canny",
        #                                                             torch_dtype=torch.bfloat16,
                                                                    
        #                                                             cache_dir=os.environ["HF_HOME"] # Use the cache directory
        #                                                             )

        print(f"What is in roast-photoshoot-ai-tech: {os.listdir(MOUNT_DIR_PHOTOSHOOT_TECH)}")
        print(f"List of controlnet images: {os.listdir(os.path.join(MOUNT_DIR_PHOTOSHOOT_TECH, 'controlnet-pics-xl'))}")
            
        s3_resource = boto3.resource('s3',
                    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
                    aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
                    region_name=os.environ.get('AWS_REGION'))
        if photoshoot.photoshoot_batch is None:
            result_batch_id_payload = {
                        "user_id": user_id,
                        "user_firebase_id": photoshoot.user_firebase_id,
                        "photoshoot_id": photoshoot_id,
                        "operation": "flux_inference",
                        "status": "pending"
                    }
            photoshoot_batch, success = engine.repository("PhotoshootBatch").post_batch(**result_batch_id_payload)
            photoshoot = engine.repository("Photoshoot").update_photoshoot(photoshoot_id=photoshoot_id, fields_update={"photoshoot_batch": photoshoot_batch.id})
        
        photoshoot_batch = photoshoot.photoshoot_batch
        images_generated = []

        ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
        SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
        REGION_NAME = os.environ.get('AWS_REGION')

        s3_client = boto3.client('s3',
                aws_access_key_id=ACCESS_KEY_ID,
                aws_secret_access_key=SECRET_ACCESS_KEY,
                region_name=REGION_NAME)
        
        sqs = boto3.client("sqs",
                                    aws_access_key_id=ACCESS_KEY_ID,
                                    aws_secret_access_key=SECRET_ACCESS_KEY,
                                    region_name=REGION_NAME)

        # Load appropriate pipeline based on mode
        mode = config['mode']
        print(f"Mode: {mode}")

        model_config = config['model'][mode]
        print(f"Loading {mode} pipeline: {model_config['name']}")

        if mode == "flux":
            print("Loading flux pipeline")
            pipe = FluxPipeline.from_pretrained(
                model_config['name'],
                torch_dtype=torch.bfloat16,
                device_map='balanced'
            )
            # Initialize the SmashConfig
            smash_config = SmashConfig()
            smash_config['compiler'] = 'torch_compile'
            smash_config['cacher'] = 'taylor_auto'
            smash_config['taylor_auto_speed_factor'] = 0.5
            smash_config._prepare_saving = False

            pipe = smash(
                model=pipe,
            token="pruna_9dcd67c48831b7601f2292a9384cd6d1",
                smash_config=smash_config,
            )
            # pipe = FluxControlNetPipeline.from_pretrained(
            #     model_config['name'],
            #     controlnet=controlnet_model_canny,
            #     torch_dtype=torch.bfloat16,
            #     cache_dir=os.environ["HF_HOME"] # Use the cache directory

            # )
        elif mode == "sdxl": 
            print("Loading sdxl pipeline") # sdxl
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_config['name'],
                torch_dtype=torch.bfloat16,
                device_map='balanced'
            )

        # Setup paths
        base_output_path = f"/root/ai-toolkit/outputs/{config['run_name']}"
        base_output_path_s3 = os.path.join(MOUNT_ROAST_IMAGE_BUCKET_DIR)
        os.makedirs(base_output_path, exist_ok=True)
        os.makedirs(base_output_path_s3, exist_ok=True)
        print(f"Pipe initiated:{pipe}")

        # Load the bntbln LoRA (always applied)
        user_lora_path = config['loras'][mode]['user']['path']
        if not test_pics:
            lora_config = config['loras'][mode]['user']
            if mode == "flux":
                lora_path = lora_config['path'].format(model_id, model_id)
            else:
                lora_path = lora_config['path'].format(user_id, model_id, f"lora_xl_{user_id}_{model_id}_udaxihhe")
            
            pipe.load_lora_weights(lora_path, adapter_name="user")

        # Get inference parameters based on mode
        inference_params = config['inference'][mode]
        if inference_mode == "test_pics":
            col = "prompts_example"
        elif inference_mode == "rd":
            col = "prompts_rd"
        else:
            col = "prompts"
        
        list_prompts = config[col]

        # Generate images
        images_generated = []
        infos_backend = []
        # generator = torch.Generator("cpu").manual_seed(random.randint(0, 1000000))
        generator = torch.Generator("cpu").manual_seed(42)

        for i, prompt_config in enumerate(list_prompts):
            print(f"Processing prompt_config: {prompt_config}")
            if photoshoot.bald:
                print(f"Photoshoot bald case")                
                prompt_config['prompt'] = prompt_config['prompt'].replace("udaxihhe an ETHNICITY GENDER", "udaxihhe a bald ETHNICITY GENDER")
            else:
                print(f"Photoshoot not bald case")
            for _keyword in ["USERNAME", "GENDER", "ETHNICITY"]:
                prompt_config['prompt'] = prompt_config['prompt'].replace(_keyword, user_info[_keyword.lower()])

            print(f"Prompt after replacing keywords: {prompt_config['prompt']}")
            
            # for _variation in range(3):
            # Reset to original weights (including bntbln)
            # pipe.unet.load_state_dict(original_weights)
            # Apply bntbln LoRA
            # Apply additional LoRAs for this prompt
            print(f"processing prompt_config: {prompt_config}")
            ## Put back for normal infernece
            if test_pics:
                lora_names = []
                lora_weights = []
            else:
                lora_names = ["user"]
                lora_weights = [1.0]

            if 'loras' in prompt_config:
                for lora in prompt_config['loras']:
                    lora_names.append(lora['name'])
                    lora_weights.append(lora['weight'])

            pipe.set_adapters(lora_names, adapter_weights=lora_weights)
            
            for _i in range(3):
                if prompt_config.get('controlnet_pic', None) is not None:
                    print(f"Controlnet case!: {prompt_config.get('controlnet_pic', None)}")
                    # init_image = load_image(os.path.join(MOUNT_DIR_PHOTOSHOOT_TECH,"controlnet-pics-xl", prompt_config.get('controlnet_pic', None)))
                    # init_image = utils.load_image_s3(image_key=f"controlnet-pics-xl/{prompt_config.get('controlnet_pic', None)}", bucket="roast-photoshoot-ai-tech", s3_resource=s3_resource)

                    # print(f"Image laoded, canny detector")
                    # # init_image = pil_to_tensor(init_image).to(device="cuda", dtype=torch.bfloat16)
                    
                    # control_image = canny_detector(init_image)

                    
                    # if isinstance(control_image, list):
                    #     control_image = control_image[0]
                    # # Ensure control image is RGB
                    # if control_image.mode != 'RGB':
                    #     control_image = control_image.convert('RGB')

                    print(f"Image cannied, generating image")
                    images = pipe(
                        prompt=prompt_config['prompt'],
                        num_inference_steps=inference_params['n_steps'],
                        guidance_scale=inference_params['guidance_scale'],
                        width=inference_params['width'],
                        height=inference_params['height'],
                        generator=generator,
                        num_images_per_prompt=3,
                        # control_image=control_image,
                        # controlnet_conditioning_scale=0.7
                        ).images

                else:

                # Generate image with mode-specific parameters
                    images = pipe(
                        prompt=prompt_config['prompt'],
                        num_inference_steps=inference_params['n_steps'],
                        guidance_scale=inference_params['guidance_scale'],
                        width=inference_params['width'],
                        height=inference_params['height'],
                        generator=generator,
                        num_images_per_prompt=1,
                    ).images

                for image in images:
                # Create filename with adapter names and weights
                    adapter_info = "_".join([f"{name}_{weight}" for name, weight in zip(lora_names, lora_weights)])
                    seed = generator.initial_seed()
                    if not test_pics:
                        filename = f"{prompt_config['filename']}_{adapter_info}_{mode}_seed{seed}"
                    else:
                        filename = f"{prompt_config['niche']}_{prompt_config['pose']}_{random.randint(0,1000)}_{mode}_seed{seed}"
                    # filename = f"{prompt_config['niche']}_{i}"
                    infos_process_backend = {
                        "images_generated": [image],
                        "user_id": user_id,
                        "photoshoot_id": photoshoot_id,
                        "photoshoot_batch_id": photoshoot_batch.id,
                        "prompt_config": prompt_config,
                        "is_ai_example": test_pics,
                        "base_output_path": base_output_path,
                        "base_output_path_s3": base_output_path_s3,
                        "filename": filename,
                    }
                    infos_backend.append(infos_process_backend)


        print("spawning process_images_backend")
        images_backend = list(process_images_backend.map(
            infos_backend  # Pass the list of dictionaries directly
        ))
        print(f"Got {len(images_backend)} images back")
        print("Processes all spawned, now adding images to user")

        user = engine.repository("User").add_to_user(
            user_firebase_id=photoshoot.user_firebase_id, 
            fields_update={"images": images_backend}
        )   
        # model_volume.commit()
        # outputs_volume.commit()
        photoshoot.update(status="to_email")
        if photoshoot.mode != enums.PhotoshootModeEnum.VIP:
            utils.send_email(photoshoot)
        photoshoot.update(status="delivered")

        print(f"Inference completed for user {user_id}, model {model_id}, photoshoot {photoshoot_id}")
    except Exception as e:
        import traceback
        print(f"Error during inference: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        photoshoot.update(status="error")



@app.function(
    timeout=7200,  # 2 hours, increase or decrease if needed
    image=image,
    secrets=[modal.Secret.from_name("roast-aws-secret"),modal.Secret.from_dict({"MODAL_LOGLEVEL": "DEBUG"})],
    volumes={MOUNT_DIR_OUTPUTS: outputs_volume,
             MOUNT_DIR_MODELS: model_volume,
            MOUNT_DIR_PHOTOSHOOT_TECH: modal.CloudBucketMount(
            bucket_name="roast-photoshoot-ai-tech",
            secret=aws_secret,
            read_only=False
        ),
            MOUNT_DIR_PHOTOSHOOT_MODELS: modal.CloudBucketMount(
            bucket_name="roast-photoshoot-ai-models",
            secret=aws_secret,
            read_only=False
        ),
            MOUNT_PHOTOSHOOT_BUCKET_DIR: modal.CloudBucketMount(
            bucket_name="roast-photoshoot-ai",
            secret=aws_secret,
            read_only=False
        ),
            MOUNT_ROAST_IMAGE_BUCKET_DIR: modal.CloudBucketMount(
            bucket_name="roast-images-ai",
            secret=aws_secret,
            read_only=False)

         },
         max_containers=10
)
def process_images_backend(kwargs):
    """
    Process images with named parameters for better control
    """
    print(f"Inside process_images_backend, received kwargs: {kwargs}")
    
    # Unpack required values with defaults if needed
    images_generated = kwargs.get('images_generated')
    image = images_generated[0]

    image = postprocess_image(image)


    user_id = kwargs.get('user_id')
    photoshoot_id = kwargs.get('photoshoot_id')
    photoshoot_batch_id = kwargs.get('photoshoot_batch_id')
    prompt_config = kwargs.get('prompt_config')
    is_ai_example = kwargs.get('is_ai_example')
    base_output_path = kwargs.get('base_output_path')
    base_output_path_s3 = kwargs.get('base_output_path_s3')
    filename = kwargs.get('filename')
    
    from rdbc.db.mongodb.mongo_engine import MongoDBEngine

    engine = MongoDBEngine(uri=os.environ.get("MONGODB_URI"))
    # Create clients inside the function
    s3_client = boto3.client('s3',
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
        region_name=os.environ.get('AWS_REGION'))
    
    sqs_client = boto3.client('sqs',
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
        region_name=os.environ.get('AWS_REGION'))
    
    photoshoot = engine.repository("Photoshoot").get_photoshoot(id=photoshoot_id)
    photoshoot_batch = engine.repository("PhotoshootBatch").get_batch(id=photoshoot_batch_id)
    user = engine.repository("User").get_user(id=user_id)
        
    
    backend_image_payload = {
        "photoshoot_id": photoshoot_id,
        "user_firebase_id": photoshoot.user_firebase_id,
        "email": photoshoot.email,
        "gender": photoshoot.gender,
        "source": "remixme",
        "filename": f"{filename}.png",
        "image_type": enums.ImageTypeEnum.PROFILE_PICTURE_AI,
        "upload_status": "not_uploaded",
        "photoshoot_id": photoshoot_id,
        "is_visible": True,
        "prompt": prompt_config['prompt'],
        "user_prompt": prompt_config['prompt'],
        "niche": prompt_config.get('niche', None),
        "pose": prompt_config.get('pose', None),
        "is_ai_example": is_ai_example,
        "is_ai": True,
        "embeddings": {},
        "photoshoot_category": prompt_config.get('category', None),
        "photoshoot_subcategory": prompt_config.get('subcategory', None),
        "photoshoot_class": prompt_config.get('class', "good"),
        "photoshoot_similarity": 0.51,
        "photoshoot_similarity_computed": False,
        "photoshoot_ai_label": "good",
    }


    image_backend, success = engine.repository("Image").post_image(**backend_image_payload)
    print(f"image_backend: {image_backend.id}")
    output_path = os.path.join(base_output_path, f"{filename}.png")
    output_path_s3 = os.path.join(base_output_path_s3, f"{image_backend.s3_key}")
    print(f"output_path_s3: {output_path_s3}")

    images_generated.append(image_backend)
    utils.upload_from_url(s3_client=s3_client,
                        bucket="roast-images-ai", 
                        pil_image=image,
                            key=f"{image_backend.s3_key}",
                            content_type= "image/png")
    image_backend.update(upload_status="uploaded")
    photoshoot_batch.update(push__images=image_backend,inc__num_images=1)
    user.update(push__images=image_backend)
    ## Tag image: 
    # utils.send_message_sqs(sqs_client=sqs, dict_value=backend_image_payload)
    message_id = utils.send_message_sqs(sqs_client=sqs_client, dict_value={"id": str(image_backend.id), "case": "image-tag"})

    print(f"Saved image to {output_path}")
    return image_backend


def postprocess_image(image_pil):

    from PIL import Image, ImageEnhance, ImageFilter
    import numpy as np

    # Apply texture: add slight noise
    np_image = np.array(image_pil).astype(np.float32)
    noise = np.random.normal(0, 5, np_image.shape).astype(np.float32)
    noisy_image = np.clip(np_image + noise, 0, 255).astype(np.uint8)
    image_with_noise = Image.fromarray(noisy_image)

    # Apply light grain and sharpness
    sharpener = ImageEnhance.Sharpness(image_with_noise)
    sharpened_image = sharpener.enhance(1.5)

    # Apply soft vignetting
    width, height = sharpened_image.size
    x_center = width / 2
    y_center = height / 2
    max_distance = np.sqrt((x_center) ** 2 + (y_center) ** 2)

    # Create vignette mask
    vignette = np.zeros((height, width), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            distance = np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)
            vignette[y, x] = 1 - 0.3 * (distance / max_distance)

    # Apply vignette to each RGB channel
    vignette_rgb = np.dstack([vignette]*3)
    sharpened_np = np.array(sharpened_image).astype(np.float32) / 255
    vignetted_np = sharpened_np * vignette_rgb
    vignetted_image = Image.fromarray(np.uint8(vignetted_np * 255))

    # Save the final image
    # output_path = f"/Users/benoitbaylin/Downloads/{image_id}_edited.png"
    # vignetted_image.save(output_path)
    return vignetted_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # require at least one config file
    parser.add_argument(
        'config_file_list',
        nargs='+',
        type=str,
        help='Name of config file (eg: person_v1 for config/person_v1.json/yaml), or full path if it is not in config folder, you can pass multiple config files and run them all sequentially'
    )
    # flag to continue if a job fails
    parser.add_argument(
        '-r', '--recover',
        action='store_true',
        help='Continue running additional jobs even if a job fails'
    )

    # optional name replacement for config file
    parser.add_argument(
        '-n', '--name',
        type=str,
        default=None,
        help='Name to replace [name] tag in config file, useful for shared config file'
    )
    args = parser.parse_args()
    print("args:", args)

    # convert list of config files to a comma-separated string for Modal compatibility
    config_file_list_str = ",".join(args.config_file_list)

    # main.call(config_file_list_str=config_file_list_str, recover=args.recover, name=args.name)


