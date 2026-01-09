## 404-base-miner
Image to 3D miner based on Microsoft Trellis Image-To-3D Model that is fully commercially ready to use.

### Hardware Requirements

To run this generator you will need a GPU with at least 24 GB of VRAM. It can work with GPUs from NVIDIA Blackwell family,
including Geforce 5090 RTX.

### Software Requirements

- latest docker package (we provide docker file in "docker" folder) or latest conda environment (we provide "conda_env.yml");
- NVIDIA GPU with cuda 12.8 support
- python 3.11

### Installation

- Docker (building & pushing to remote register):
```console
cd /docker
docker build --build-arg GITHUB_USER="" --build-arg GITHUB_TOKEN="" -t docker_name:docker-tag .
docker tag docker_name:docker-tag docker-register-path:docker-register-name
docker push docker-register-path:docker-register-name   
```
- Conda Env. (shell script will install everything you need to run the project):
```console
bash setup_env.sh
```
### How to run:
- Docker (run locally):
```commandline
docker run --gpus all -it docker_name:docker-tag bash

# outside docker
curl -X POST "http://0.0.0.0:10006/generate" -F "prompt_image_file=@/path/to/your/image.png" > model.ply
```
- Conda Env.:
```commandline
# start pm2 process
pm2 start generation.config.js

# view logs
pm2 logs

# send prompt image
curl -X POST "http://0.0.0.0:10006/generate" -F "prompt_image_file=@/path/to/your/image.png" > model.ply
```

### Qwen image edit (hard-coded, before background removal)

This miner applies a fixed Qwen image-edit step **automatically** before background removal, using a hard-coded prompt tuned for 3D-friendly inputs.

To change the prompt/params, edit `GaussianProcessor.QWEN_EDIT_PROMPT` (and related constants) in `trellis_generator/trellis_gs_processor.py`.

This setup uses the **Lightning LoRA** by default (4 steps / cfg=1.0). You can override the LoRA weights location via:
- `QWEN_EDIT_LORA_PATH` (local path or `repo_id/filename` on Hugging Face)

### Background removal memory tips (avoid CUDA OOM)

Running multiple background-removal models + the VLM selector can use a lot of VRAM. You can reduce memory by setting:
- `BG_REMOVER_DEVICE=cpu` (run background removal on CPU; slower but stable)

### Multi-image Trellis input

For 3D generation, the miner feeds **three images** into Trellis (when Qwen edit is enabled):
- original image (background removed)
- Qwen-edited left three-quarters view (background removed)
- Qwen-edited right three-quarters view (background removed)

Note: Qwen image edit requires a `diffusers` version that includes `QwenImageEditPlusPipeline` (the provided Dockerfile installs diffusers from git).
