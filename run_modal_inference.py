import os
import modal
import yaml
import argparse
from typing import Dict
import torch
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, snapshot_download
from modal import web_endpoint
from fastapi import Request
from utils import utils

# Load the .env file if it exists
load_dotenv()

# Set up Modal volume and stub
volume = modal.Volume.from_name("flux-lora-models")
# stub = modal.Stub("flux-lora-inference")


secret = modal.Secret.from_dict({"AWS_ACCESS_KEY_ID": os.environ.get('AWS_ACCESS_KEY_ID'),
                                 "AWS_SECRET_ACCESS_KEY": os.environ.get('AWS_SECRET_ACCESS_KEY'),
                                 "AWS_REGION": os.environ.get('AWS_REGION')})
MOUNT_PHOTOSHOOT_BUCKET_DIR = "/root/ai-toolkit/roast-photoshoot-ai"

# Define the image for our Modal container
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .env({"HF_TOKEN": os.environ.get("HF_TOKEN"), "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .apt_install("libgl1", "libglib2.0-0", "git")
    .run_commands(
        "pip install --upgrade pip",
        "pip install --upgrade setuptools",
        "pip install --upgrade wheel",
    )
    .pip_install(
        "python-dotenv",
        "torch", 
        "diffusers[torch]", 
        "transformers", 
        "accelerate", 
        "safetensors",
        "pyyaml",
        # "black-forest-labs-FLUX"
        "ftfy", 
        "torchvision", 
        "oyaml", 
        "opencv-python", 
        "albumentations",
        "safetensors",
        "lycoris-lora==1.8.3",
        "flatten_json",
        "pyyaml",
        "tensorboard", 
        "kornia", 
        "invisible-watermark", 
        "einops", 
        "accelerate", 
        "toml", 
        "pydantic",
        "omegaconf",
        "k-diffusion",
        "open_clip_torch",
        "timm",
        "prodigyopt",
        "controlnet_aux==0.0.7",
        "bitsandbytes",
        "hf_transfer",
        "lpips", 
        "pytorch_fid", 
        "optimum-quanto", 
        "sentencepiece", 
        "huggingface_hub", 
        "peft",
        "git+https://github.com/tencent-ailab/IP-Adapter.git",
        "fastapi",
        "pydantic"

    )
)

code_mount = modal.Mount.from_local_dir("/Users/benoitbaylin/Documents/code/ai-toolkit", remote_path="/root/ai-toolkit")
model_volume = modal.Volume.from_name("flux-lora-models", create_if_missing=True)
MOUNT_DIR_MODELS = "/root/ai-toolkit/models"  # modal_output, due to "cannot mount volume on non-empty path" requirement
output_volume = modal.Volume.from_name("inference_outputs", create_if_missing=True)
MOUNT_DIR_OUTPUTS = "/root/ai-toolkit/outputs"  # modal_output, due to "cannot mount volume on non-empty path" requirement
MOUNT_PHOTOSHOOT_BUCKET_DIR = "/root/ai-toolkit/roast-photoshoot-ai"


from toolkit.job import get_job

app = modal.App(name=f"inference-flux-lora",
                image=image,
                mounts=[code_mount],
                volumes={MOUNT_DIR_OUTPUTS: output_volume, MOUNT_DIR_MODELS: model_volume},
                secrets=[modal.Secret.from_name("roast-aws-secret"),modal.Secret.from_name("backend-keys")],
                )


def run_inference(config, user_id: str, model_id: str, photoshoot_id: str):
    from diffusers import FluxPipeline
    
    utils.put_photoshoot(photoshoot_id, {"status_flux": "shooting"})
    print(f"Loading flux pipeline, finetuned model: {config['model']['name_or_path']}")
    pipe = FluxPipeline.from_pretrained(
        config['model']['name_or_path'],
        torch_dtype=torch.bfloat16,
        device_map='balanced'
    )

    print(f"Pipe loaded")
    base_output_path = f"/root/ai-toolkit/outputs/{config['run_name']}"
    base_output_path_s3 = os.path.join(MOUNT_PHOTOSHOOT_BUCKET_DIR,f"{user_id}/models/{model_id}/flux_outputs/")
    os.makedirs(base_output_path, exist_ok=True)
    os.makedirs(base_output_path_s3, exist_ok=True)
    print(f"Pipe initiated:{pipe}")

    # Load the bntbln LoRA (always applied)
    user_lora_path = config['loras']['user']['path']
    pipe.load_lora_weights(user_lora_path, adapter_name="user")
    
    # Load other LoRAs
    for lora_name, lora_config in config['loras'].items():
        if lora_name != 'user':
            pipe.load_lora_weights(lora_config['path'], adapter_name=lora_name)
    
    # Store the original model weights

    for i, prompt_config in enumerate(config['prompts']):
        # Reset to original weights (including bntbln)
        # pipe.unet.load_state_dict(original_weights)
        # Apply bntbln LoRA
        # Apply additional LoRAs for this prompt
        lora_names = ["user"]
        lora_weights = [1.0]
        if 'loras' in prompt_config:
            for lora in prompt_config['loras']:
                lora_names.append(lora['name'])
                lora_weights.append(lora['weight'])
        print(f"Applying LoRAs: {lora_names} with weights {lora_weights}")
        pipe.set_adapters(lora_names, adapter_weights=lora_weights)
        
        generator = torch.Generator("cpu").manual_seed(prompt_config.get('seed', config.get('global_settings', {}).get('seed', 42)))
        
        image = pipe(
            prompt=prompt_config['text'],
            num_inference_steps=prompt_config.get('num_inference_steps', config['global_settings']['num_inference_steps']),
            guidance_scale=prompt_config.get('guidance_scale', config['global_settings']['guidance_scale']),
            width=config['global_settings']['width'],
            height=config['global_settings']['height'],
            generator=generator,
        ).images[0]

        # Create filename with adapter names and weights
        adapter_info = "_".join([f"{name}_{weight}" for name, weight in zip(lora_names, lora_weights)])
        filename = f"{prompt_config['filename']}_{adapter_info}"
        
        output_path = os.path.join(base_output_path, f"{filename}.png")
        output_path_s3 = os.path.join(base_output_path_s3, f"{filename}.png")
        
        image.save(output_path,format="PNG")
        image.save(output_path_s3,format="PNG")
        print(f"Saved image to {output_path}")

    model_volume.commit()
    output_volume.commit()
    utils.put_photoshoot(photoshoot_id, {"status_flux": "to_email"})

@app.function(
    gpu="A100",  # gpu="H100"
    timeout=7200,  # 2 hours, increase or decrease if needed
    image=image,
    secrets=[modal.Secret.from_name("roast-aws-secret"), modal.Secret.from_name("backend-keys")],
    volumes={MOUNT_DIR_OUTPUTS: output_volume, MOUNT_DIR_MODELS: model_volume,
            MOUNT_PHOTOSHOOT_BUCKET_DIR: modal.CloudBucketMount(
            bucket_name="roast-photoshoot-ai",
            secret=secret,
            read_only=False
        )},
)
@web_endpoint(method="POST")
async def infer_user_endpoint(request: Request):
    print("Inference request: ", request)
    data = await request.json()
    user_id = data['user_id']
    model_id = data['model_id']
    photoshoot_id = data['photoshoot_id']
    print("Starting inference")
    infer_user.spawn(user_id, model_id, photoshoot_id)
    return {"status": "success", "user_id": user_id, "model_id": model_id}


@app.function(
    gpu="A100",  # gpu="H100"
    timeout=7200,  # 2 hours, increase or decrease if needed
    image=image,
    secrets=[modal.Secret.from_name("roast-aws-secret")],
    volumes={MOUNT_DIR_OUTPUTS: output_volume, MOUNT_DIR_MODELS: model_volume,
            MOUNT_PHOTOSHOOT_BUCKET_DIR: modal.CloudBucketMount(
            bucket_name="roast-photoshoot-ai",
            secret=secret,
            read_only=False
        )},
)
def infer_user(user_id: str, model_id: str, photoshoot_id: str):
    # Load the generic_flux_inference.yaml config file
    config_path = "/root/ai-toolkit/config/generic_flux_inference.yaml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Update the config with the specific model_id
    config['run_name'] = model_id
    config['loras']['user']['path'] = config['loras']['user']['path'].format(model_id, model_id)

    # Run inference
    run_inference(config, user_id=user_id, model_id=model_id, photoshoot_id=photoshoot_id)

    # Commit the output volume after inference
    output_volume.commit()
    model_volume.commit()

    print(f"Inference completed for user {user_id}, model {model_id}")


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

    main.call(config_file_list_str=config_file_list_str, recover=args.recover, name=args.name)