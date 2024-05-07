import os
import re
import toml
from time import time
from IPython.display import Markdown, display
from pipeline_rf import RectifiedFlowPipeline
import torch


if "dependencies_installed" not in globals():
  dependencies_installed = False
if "model_file" not in globals():
  model_file = None

if "custom_dataset" not in globals():
  custom_dataset = None
if "override_dataset_config_file" not in globals():
  override_dataset_config_file = None
if "override_config_file" not in globals():
  override_config_file = None
if "optimizer" not in globals():
  optimizer = "AdamW8bit"
if "optimizer_args" not in globals():
  optimizer_args = None
if "continue_from_lora" not in globals():
  continue_from_lora = ""
if "weighted_captions" not in globals():
  weighted_captions = False
if "adjust_tags" not in globals():
  adjust_tags = False
if "keep_tokens_weight" not in globals():
  keep_tokens_weight = 1.0

COLAB = True 
XFORMERS = True
BETTER_EPOCH_NAMES = True
LOAD_TRUNCATED_IMAGES = True


project_name = ""
folder_structure = "Organize by project (MyDrive/Loras/project_name/dataset)"
training_model = RectifiedFlowPipeline.from_pretrained("XCLIU/instaflow_0_9B_from_sd_1_5", torch_dtype=torch.float16)




resolution = 512 
flip_aug = False 
caption_extension = ".txt" 
shuffle_tags = True 
shuffle_caption = shuffle_tags
activation_tags = "1" 
keep_tokens = int(activation_tags)

num_repeats = 10 
preferred_unit = "Epochs" 
how_many = 10
max_train_epochs = how_many if preferred_unit == "Epochs" else None
max_train_steps = how_many if preferred_unit == "Steps" else None
save_every_n_epochs = 1 
keep_only_last_n_epochs = 10 
if not save_every_n_epochs:
  save_every_n_epochs = max_train_epochs
if not keep_only_last_n_epochs:
  keep_only_last_n_epochs = max_train_epochs
train_batch_size = 2

unet_lr = 5e-4 
text_encoder_lr = 1e-4 
lr_scheduler = "cosine_with_restarts" 
lr_scheduler_number = 3 
lr_scheduler_num_cycles = lr_scheduler_number if lr_scheduler == "cosine_with_restarts" else 0
lr_scheduler_power = lr_scheduler_number if lr_scheduler == "polynomial" else 0

lr_warmup_ratio = 0.05 
lr_warmup_steps = 0

min_snr_gamma = True 
min_snr_gamma_value = 5.0 if min_snr_gamma else None



lora_type = "LoRA" 









network_dim = 16 
network_alpha = 8 

conv_dim = 8 
conv_alpha = 4 

network_module = "networks.lora"
network_args = None
if lora_type.lower() == "locon":
  network_args = [f"conv_dim={conv_dim}", f"conv_alpha={conv_alpha}"]







if optimizer.lower() == "prodigy" or "dadapt" in optimizer.lower():
  if override_values_for_dadapt_and_prodigy:
    unet_lr = 0.5
    text_encoder_lr = 0.5
    lr_scheduler = "constant_with_warmup"
    lr_warmup_ratio = 0.05
    network_alpha = network_dim

  if not optimizer_args:
    optimizer_args = ["decouple=True","weight_decay=0.01","betas=[0.9,0.999]"]
    if optimizer == "Prodigy":
      optimizer_args.extend(["d_coef=2","use_bias_correction=True"])
      if lr_warmup_ratio > 0:
        optimizer_args.append("safeguard_warmup=True")
      else:
        optimizer_args.append("safeguard_warmup=False")

root_dir = "/content" if COLAB else "~/Loras"
deps_dir = os.path.join(root_dir, "deps")

if "/Loras" in folder_structure:
  main_dir      = os.path.join(root_dir, "drive/MyDrive/Loras") if COLAB else root_dir
  log_folder    = os.path.join(main_dir, "_logs")
  config_folder = os.path.join(main_dir, project_name)
  images_folder = os.path.join(main_dir, project_name, "dataset")
  output_folder = os.path.join(main_dir, project_name, "output")
else:
  main_dir      = os.path.join(root_dir, "drive/MyDrive/lora_training") if COLAB else root_dir
  images_folder = os.path.join(main_dir, "datasets", project_name)
  output_folder = os.path.join(main_dir, "output", project_name)
  config_folder = os.path.join(main_dir, "config", project_name)
  log_folder    = os.path.join(main_dir, "log")

config_file = os.path.join(config_folder, "training_config.toml")
dataset_config_file = os.path.join(config_folder, "dataset_config.toml")
accelerate_config_file = os.path.join(repo_dir, "accelerate_config/config.yaml")

def install_dependencies():
  os.chdir(root_dir)
  os.chdir(repo_dir)


  !apt -y update -qq
  !apt -y install aria2 -qq
  !pip install accelerate==0.15.0 diffusers==0.10.2 transformers==4.26.0 bitsandbytes==0.41.3.post2 opencv-python==4.8.0.76 tensorflow torchvision==0.16.0 torchtext==0.16.0 torchaudio==2.1.0 jax==0.4.23 jaxlib==0.4.23
  !pip install toml==0.10.2 ftfy==6.1.1 einops==0.6.0 timm==0.6.12 fairscale==0.4.13 albumentations==1.3.1 voluptuous==0.13.1 requests==2.31.0 pytorch-lightning==1.9.0 
  !pip install safetensors lion_pytorch==0.0.6 dadaptation==3.1 prodigyopt==1.0 pygments
  !pip install .
  if XFORMERS:
    !pip install xformers==0.0.22.post7

  if COLAB:
    !sed -i "s@cpu@cuda@" library/model_util.py 
  if LOAD_TRUNCATED_IMAGES:
    !sed -i 's/from PIL import Image/from PIL import Image, ImageFile\nImageFile.LOAD_TRUNCATED_IMAGES=True/g' library/train_util.py 
  if BETTER_EPOCH_NAMES:
    !sed -i 's/{:06d}/{:02d}/g' library/train_util.py 
    !sed -i 's/"." + args.save_model_as)/"-{:02d}.".format(num_train_epochs) + args.save_model_as)/g' train_network.py 

  from accelerate.utils import write_basic_config
  if not os.path.exists(accelerate_config_file):
    write_basic_config(save_location=accelerate_config_file)

  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
  os.environ["BITSANDBYTES_NOWELCOME"] = "1"
  os.environ["SAFETENSORS_FAST_GPU"] = "1"

def validate_dataset():
  global lr_warmup_steps, lr_warmup_ratio, caption_extension, keep_tokens, keep_tokens_weight, weighted_captions, adjust_tags
  supported_types = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

  print("\n💿 Checking dataset...")
  if not project_name.strip() or any(c in project_name for c in " .()\"'\\/"):
    print("💥 Error: Please choose a valid project name.")
    return

  if custom_dataset:
    try:
      datconf = toml.loads(custom_dataset)
      datasets = [d for d in datconf["datasets"][0]["subsets"]]
    except:
      print(f"💥 Error: Your custom dataset is invalid or contains an error! Please check the original template.")
      return
    reg = [d.get("image_dir") for d in datasets if d.get("is_reg", False)]
    datasets_dict = {d["image_dir"]: d["num_repeats"] for d in datasets}
    folders = datasets_dict.keys()
    files = [f for folder in folders for f in os.listdir(folder)]
    images_repeats = {folder: (len([f for f in os.listdir(folder) if f.lower().endswith(supported_types)]), datasets_dict[folder]) for folder in folders}
  else:
    reg = []
    folders = [images_folder]
    files = os.listdir(images_folder)
    images_repeats = {images_folder: (len([f for f in files if f.lower().endswith(supported_types)]), num_repeats)}

  for folder in folders:
    if not os.path.exists(folder):
      print(f"💥 Error: The folder {folder.replace('/content/drive/', '')} doesn't exist.")
      return
  for folder, (img, rep) in images_repeats.items():
    if not img:
      print(f"💥 Error: Your {folder.replace('/content/drive/', '')} folder is empty.")
      return
  for f in files:
    if not f.lower().endswith((".txt", ".npz")) and not f.lower().endswith(supported_types):
      print(f"💥 Error: Invalid file in dataset: \"{f}\". Aborting.")
      return

  if not [txt for txt in files if txt.lower().endswith(".txt")]:
    caption_extension = ""
  if continue_from_lora and not (continue_from_lora.endswith(".safetensors") and os.path.exists(continue_from_lora)):
    print(f"💥 Error: Invalid path to existing Lora. Example: /content/drive/MyDrive/Loras/example.safetensors")
    return

  pre_steps_per_epoch = sum(img*rep for (img, rep) in images_repeats.values())
  steps_per_epoch = pre_steps_per_epoch/train_batch_size
  total_steps = max_train_steps or int(max_train_epochs*steps_per_epoch)
  estimated_epochs = int(total_steps/steps_per_epoch)
  lr_warmup_steps = int(total_steps*lr_warmup_ratio)

  for folder, (img, rep) in images_repeats.items():
    print("📁"+folder.replace("/content/drive/", "") + (" (Regularization)" if folder in reg else ""))
    print(f"📈 Found {img} images with {rep} repeats, equaling {img*rep} steps.")
  print(f"📉 Divide {pre_steps_per_epoch} steps by {train_batch_size} batch size to get {steps_per_epoch} steps per epoch.")
  if max_train_epochs:
    print(f"🔮 There will be {max_train_epochs} epochs, for around {total_steps} total training steps.")
  else:
    print(f"🔮 There will be {total_steps} steps, divided into {estimated_epochs} epochs and then some.")

  if total_steps > 10000:
    print("💥 Error: Your total steps are too high. You probably made a mistake. Aborting...")
    return

  if adjust_tags:
    print(f"\n📎 Weighted tags: {'ON' if weighted_captions else 'OFF'}")
    if weighted_captions:
      print(f"📎 Will use {keep_tokens_weight} weight on {keep_tokens} activation tag(s)")
    print("📎 Adjusting tags...")
    adjust_weighted_tags(folders, keep_tokens, keep_tokens_weight, weighted_captions)

  return True

def adjust_weighted_tags(folders, keep_tokens: int, keep_tokens_weight: float, weighted_captions: bool):
  weighted_tag = re.compile(r"\((.+?):[.\d]+\)(,|$)")
  for folder in folders:
    for txt in [f for f in os.listdir(folder) if f.lower().endswith(".txt")]:
      with open(os.path.join(folder, txt), 'r') as f:
        content = f.read()
      
      content = content.replace('\\', '')
      content = weighted_tag.sub(r'\1\2', content)
      if weighted_captions:
        
        content = content.replace(r'(', r'\(').replace(r')', r'\)').replace(r':', r'\:')
        if keep_tokens_weight > 1:
          tags = [s.strip() for s in content.split(",")]
          for i in range(min(keep_tokens, len(tags))):
            tags[i] = f'({tags[i]}:{keep_tokens_weight})'
          content = ", ".join(tags)
      with open(os.path.join(folder, txt), 'w') as f:
        f.write(content)

def create_config():
  global dataset_config_file, config_file, model_file

  if override_config_file:
    config_file = override_config_file
    print(f"\n⭕ Using custom config file {config_file}")
  else:
    config_dict = {
      "additional_network_arguments": {
        "unet_lr": unet_lr,
        "text_encoder_lr": text_encoder_lr,
        "network_dim": network_dim,
        "network_alpha": network_alpha,
        "network_module": network_module,
        "network_args": network_args,
        "network_train_unet_only": True if text_encoder_lr == 0 else None,
        "network_weights": continue_from_lora if continue_from_lora else None
      },
      "optimizer_arguments": {
        "learning_rate": unet_lr,
        "lr_scheduler": lr_scheduler,
        "lr_scheduler_num_cycles": lr_scheduler_num_cycles if lr_scheduler == "cosine_with_restarts" else None,
        "lr_scheduler_power": lr_scheduler_power if lr_scheduler == "polynomial" else None,
        "lr_warmup_steps": lr_warmup_steps if lr_scheduler != "constant" else None,
        "optimizer_type": optimizer,
        "optimizer_args": optimizer_args if optimizer_args else None,
      },
      "training_arguments": {
        "max_train_steps": max_train_steps,
        "max_train_epochs": max_train_epochs,
        "save_every_n_epochs": save_every_n_epochs,
        "save_last_n_epochs": keep_only_last_n_epochs,
        "train_batch_size": train_batch_size,
        "noise_offset": None,
        "clip_skip": 2,
        "min_snr_gamma": min_snr_gamma_value,
        "weighted_captions": weighted_captions,
        "seed": 42,
        "max_token_length": 225,
        "xformers": XFORMERS,
        "lowram": COLAB,
        "max_data_loader_n_workers": 8,
        "persistent_data_loader_workers": True,
        "save_precision": "fp16",
        "mixed_precision": "fp16",
        "output_dir": output_folder,
        "logging_dir": log_folder,
        "output_name": project_name,
        "log_prefix": project_name,
      },
      "model_arguments": {
        "pretrained_model_name_or_path": model_file,
        "v2": custom_model_is_based_on_sd2,
        "v_parameterization": True if custom_model_is_based_on_sd2 else None,
      },
      "saving_arguments": {
        "save_model_as": "safetensors",
      },
      "dreambooth_arguments": {
        "prior_loss_weight": 1.0,
      },
      "dataset_arguments": {
        "cache_latents": True,
      },
    }

    for key in config_dict:
      if isinstance(config_dict[key], dict):
        config_dict[key] = {k: v for k, v in config_dict[key].items() if v is not None}

    with open(config_file, "w") as f:
      f.write(toml.dumps(config_dict))
    print(f"\n📄 Config saved to {config_file}")

  if override_dataset_config_file:
    dataset_config_file = override_dataset_config_file
    print(f"⭕ Using custom dataset config file {dataset_config_file}")
  else:
    dataset_config_dict = {
      "general": {
        "resolution": resolution,
        "shuffle_caption": shuffle_caption,
        "keep_tokens": keep_tokens,
        "flip_aug": flip_aug,
        "caption_extension": caption_extension,
        "enable_bucket": True,
        "bucket_reso_steps": 64,
        "bucket_no_upscale": False,
        "min_bucket_reso": 320 if resolution > 640 else 256,
        "max_bucket_reso": 1280 if resolution > 640 else 1024,
      },
      "datasets": toml.loads(custom_dataset)["datasets"] if custom_dataset else [
        {
          "subsets": [
            {
              "num_repeats": num_repeats,
              "image_dir": images_folder,
              "class_tokens": None if caption_extension else project_name
            }
          ]
        }
      ]
    }

    for key in dataset_config_dict:
      if isinstance(dataset_config_dict[key], dict):
        dataset_config_dict[key] = {k: v for k, v in dataset_config_dict[key].items() if v is not None}

    with open(dataset_config_file, "w") as f:
      f.write(toml.dumps(dataset_config_dict))
    print(f"📄 Dataset config saved to {dataset_config_file}")



def main():
  global dependencies_installed

  if COLAB and not os.path.exists('/content/drive'):
    from google.colab import drive
    print("📂 Connecting to Google Drive...")
    drive.mount('/content/drive')

  for dir in (main_dir, deps_dir, repo_dir, log_folder, images_folder, output_folder, config_folder):
    os.makedirs(dir, exist_ok=True)

  if not validate_dataset():
    return

  if not dependencies_installed:
    print("\n🏭 Installing dependencies...\n")
    t0 = time()
    install_dependencies()
    t1 = time()
    dependencies_installed = True
    print(f"\n✅ Installation finished in {int(t1-t0)} seconds.")
  else:
    print("\n✅ Dependencies already installed.")



  create_config()

  print("\n⭐ Starting trainer...\n")
  os.chdir(repo_dir)

  !accelerate launch --config_file={accelerate_config_file} --num_cpu_threads_per_process=1 train_network_wrapper.py --dataset_config={dataset_config_file} --config_file={config_file}

main()
