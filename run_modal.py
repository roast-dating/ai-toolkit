'''

ostris/ai-toolkit on https://modal.com
Run training with the following command:
modal run run_modal.py --config-file-list-str=/root/ai-toolkit/config/whatever_you_want.yml

'''
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import sys
import modal
from dotenv import load_dotenv
from modal import web_endpoint
from fastapi import Request
import shutil
import yaml
from utils import utils
# Load the .env file if it exists
load_dotenv()

print(f"HF_TOKEN: {os.environ.get('HF_TOKEN')}")
from huggingface_hub import HfApi, HfFolder
sys.path.insert(0, "/root/ai-toolkit")
# must come before ANY torch or fastai imports
# import toolkit.cuda_malloc

# turn off diffusers telemetry until I can figure out how to make it opt-in
os.environ['DISABLE_TELEMETRY'] = 'YES'

# define the volume for storing model outputs, using "creating volumes lazily": https://modal.com/docs/guide/volumes
# you will find your model, samples and optimizer stored in: https://modal.com/storage/your-username/main/flux-lora-models
# define modal app
image = (
    # modal.Image.debian_slim(python_version="3.11")
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .env({"HF_TOKEN": os.environ.get("HF_TOKEN"),
          "HF_HUB_ENABLE_HF_TRANSFER": "1",
          "AWS_DEFAULT_REGION": "eu-west-1",
          "CUDA_HOME": "/usr/local/cuda" })
    # install required system and pip packages, more about this modal approach: https://modal.com/docs/examples/dreambooth_app
    .apt_install("libgl1", "libglib2.0-0", "git")
    .pip_install(
        "python-dotenv",
        "torch", 
        "diffusers[torch]", 
        "transformers", 
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

print("Installing IP-Adapter")
image.pip_install("git+https://github.com/tencent-ailab/IP-Adapter.git")

# mount for the entire ai-toolkit directory
# example: "/Users/username/ai-toolkit" is the local directory, "/root/ai-toolkit" is the remote directory
code_mount = modal.Mount.from_local_dir("/Users/benoitbaylin/Documents/code/ai-toolkit", remote_path="/root/ai-toolkit")
secret = modal.Secret.from_dict({"AWS_ACCESS_KEY_ID": os.environ.get('AWS_ACCESS_KEY_ID'),
                                 "AWS_SECRET_ACCESS_KEY": os.environ.get('AWS_SECRET_ACCESS_KEY'),
                                 "AWS_REGION": os.environ.get('AWS_REGION')})

model_volume = modal.Volume.from_name("flux-lora-models", create_if_missing=True)

# modal_output, due to "cannot mount volume on non-empty path" requirement
MOUNT_DIR = "/root/ai-toolkit/modal_output"  # modal_output, due to "cannot mount volume on non-empty path" requirement
MOUNT_PHOTOSHOOT_BUCKET_DIR = "/root/ai-toolkit/roast-photoshoot-ai"

# create the Modal app with the necessary mounts and volumes
app = modal.App(name="train-flux-lora",
                image=image,
                mounts=[code_mount],
                )

# Check if we have DEBUG_TOOLKIT in env
if os.environ.get("DEBUG_TOOLKIT", "0") == "1":
    # Set torch to trace mode
    import torch
    torch.autograd.set_detect_anomaly(True)

import argparse
from toolkit.job import get_job
from huggingface_hub import HfApi, HfFolder
HfFolder.save_token(os.environ.get('HF_TOKEN'))

print(f"Env variables aws: {os.environ.get('AWS_ACCESS_KEY_ID')}")
print(f"Env variables aws: {os.environ.get('AWS_SECRET_ACCESS_KEY')}")

def print_end_message(jobs_completed, jobs_failed):
    failure_string = f"{jobs_failed} failure{'' if jobs_failed == 1 else 's'}" if jobs_failed > 0 else ""
    completed_string = f"{jobs_completed} completed job{'' if jobs_completed == 1 else 's'}"
    
    print("")
    print("========================================")
    print("Result:")
    if len(completed_string) > 0:
        print(f" - {completed_string}")
    if len(failure_string) > 0:
        print(f" - {failure_string}")
    print("========================================")


@app.function(gpu="A100", image=image)
def check_nvidia_smi():
    import subprocess
    output = subprocess.check_output(["nvidia-smi"], text=True)
    assert "Driver Version: 550.54.15" in output
    assert "CUDA Version: 12.4" in output
    print(f"All good, output: {output}")
    return output

@app.function(
    # request a GPU with at least 24GB VRAM
    # more about modal GPU's: https://modal.com/docs/guide/gpu
    gpu="A100", # gpu="H100"
    # more about modal timeouts: https://modal.com/docs/guide/timeouts
    timeout=7200,  # 2 hours, increase or decrease if needed
    secrets = [modal.Secret.from_name("roast-aws-secret"),modal.Secret.from_name("backend-keys")],    # Note: providing AWS_REGION can help when automatic detection of the bucket region fails.
    volumes={MOUNT_DIR: model_volume, MOUNT_PHOTOSHOOT_BUCKET_DIR: modal.CloudBucketMount(
            bucket_name="roast-photoshoot-ai",
            secret=secret,
            read_only=True
        )},
)
@web_endpoint(method="POST")
async def train_lora(request: Request):
    print("Training Lora: ", request)
    data = await request.json()
    prompt = data.get("prompt", "A default prompt")
    # print(f"Secret name: {secret_name}")
    print(f"Secret: {secret}")
    print("PHOTOSHOOT_BUCKET_DIR: ", MOUNT_PHOTOSHOOT_BUCKET_DIR)
    # print(f"Files in bucket: {os.listdir(MOUNT_PHOTOSHOOT_BUCKET_DIR)}")
    # Your inference code here
    
    model_id = data['model_id']
    user_id = data['user_id']
    photoshoot_id = data['photoshoot_id']
    print("Starting training")
    train_user.spawn(user_id, model_id, photoshoot_id)
    return {"status": "success", "prompt": prompt}



@app.function(
    gpu="A100", # gpu="H100"
    timeout=7200,  # 2 hours, increase or decrease if needed
    image=image,
    secrets=[modal.Secret.from_name("roast-aws-secret"),modal.Secret.from_name("backend-keys")],    volumes={MOUNT_DIR: model_volume, MOUNT_PHOTOSHOOT_BUCKET_DIR: modal.CloudBucketMount(
            bucket_name="roast-photoshoot-ai",
            secret=secret,
            read_only=True
        )},
)
def train_user(user_id: str, model_id: str, photoshoot_id: str):
    
    s3_base_folder = os.path.join(MOUNT_PHOTOSHOOT_BUCKET_DIR, "{}/models/{}/preprocessed/")
    s3_base_captions_folder = os.path.join(MOUNT_PHOTOSHOOT_BUCKET_DIR,"{}/models/{}/captions/")
    # Copy data from both s3_base_folder and s3_base_captions_folder to dataset_path
    print(f"Bearer: {os.environ.get('ROAST_BEARER')}")
    username = "udaxihhe"
    s3_folder = s3_base_folder.format(user_id, model_id)
    s3_caption_folder = s3_base_captions_folder.format(user_id, model_id)
    
    dataset_path = "/root/ai-toolkit/data/{}/".format(model_id)

    # Print the number of files in s3_folder and s3_caption_folder
    s3_folder_files = len(os.listdir(s3_folder))
    s3_caption_folder_files = len(os.listdir(s3_caption_folder))
    
    print(f"Number of photos in s3_folder:{s3_folder}: {s3_folder_files}")
    print(f"Number of captions in s3_caption_folder:{s3_caption_folder}: {s3_caption_folder_files}")
    
    # model_save_path = os.path.join(MOUNT_DIR, model_id)
    os.makedirs(dataset_path, exist_ok=True)
    for folder in [s3_folder, s3_caption_folder]:
        for file in os.listdir(folder):
            shutil.copy2(os.path.join(folder, file), dataset_path)
            
    print(f"Number of photos in dataset_path:{dataset_path}: {len(os.listdir(dataset_path))}")

    # Load the generic_flux_training.yaml config file
    config_path = "/root/ai-toolkit/config/generic_flux_training.yaml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    # Update the config with the specific model_id
    config['config']['name'] = model_id
    config['config']['process'][0]['datasets'][0]['folder_path'] = dataset_path
    utils.put_photoshoot(photoshoot_id, {"status_flux": "training"})
    main_function([config_path], config, False)
    utils.put_photoshoot(photoshoot_id, {"status_flux": "to_shoot"})
    


@app.function(
    # request a GPU with at least 24GB VRAM
    # more about modal GPU's: https://modal.com/docs/guide/gpu
    gpu="A100", # gpu="H100"
    # more about modal timeouts: https://modal.com/docs/guide/timeouts
    timeout=7200,  # 2 hours, increase or decrease if needed
    image=image
)
def main(config_file_list_str: str, recover: bool = False, name: str = None):
    # convert the config file list from a string to a list
    config_file_list = config_file_list_str.split(",")

    main_function(config_file_list, recover, name)



def main_function(config_file_list: list, config: dict = None, recover: bool = False, name: str = None):

    jobs_completed = 0
    jobs_failed = 0
    print(f"Is config is none: {config is None}")
    print(f"Running {len(config_file_list)} job{'' if len(config_file_list) == 1 else 's'}")

    for config_file in config_file_list:
        try:
            job = get_job(config_file, config, name)
            
            job.config['process'][0]['training_folder'] = MOUNT_DIR
            os.makedirs(MOUNT_DIR, exist_ok=True)
            print(f"Training outputs will be saved to: {MOUNT_DIR}")
            
            # run the job
            job.run()
            
            # commit the volume after training
            model_volume.commit()
            
            job.cleanup()
            jobs_completed += 1
        except Exception as e:
            print(f"Error running job: {e}")
            jobs_failed += 1
            if not recover:
                print_end_message(jobs_completed, jobs_failed)
                raise e

    print_end_message(jobs_completed, jobs_failed)

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

    # convert list of config files to a comma-separated string for Modal compatibility
    config_file_list_str = ",".join(args.config_file_list)

    main.call(config_file_list_str=config_file_list_str, recover=args.recover, name=args.name)
