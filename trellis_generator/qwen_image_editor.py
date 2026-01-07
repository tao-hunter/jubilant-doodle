import os
import math
from dataclasses import dataclass

import torch
from PIL import Image
from loguru import logger


@dataclass(frozen=True)
class QwenImageEditConfig:
    model_id: str = "Qwen/Qwen-Image-Edit-2511"
    # Lightning LoRA defaults: 4 steps, cfg=1.0
    true_cfg_scale: float = 1.0
    negative_prompt: str = " "
    num_inference_steps: int = 4
    guidance_scale: float = 1.0
    num_images_per_prompt: int = 1
    # Lightning LoRA weights (can be a local path, or "repo_id/filename" on HF)
    lora_path: str | None = "lightx2v/Qwen-Image-Edit-2511-Lightning/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-fp32.safetensors"
    cpu_offload: bool = False
    unload_after_call: bool = False


class QwenImageEditor:
    """
    Thin wrapper around diffusers' QwenImageEditPlusPipeline.
    Loaded lazily to avoid extra VRAM hit unless edit is requested.
    """

    def __init__(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        config: QwenImageEditConfig | None = None,
    ) -> None:
        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = dtype or self._default_dtype(self._device)
        self._config = config or self._config_from_env()
        self._pipe = None

    @staticmethod
    def _default_dtype(device: torch.device) -> torch.dtype:
        if device.type != "cuda":
            return torch.float32
        # Prefer bf16 where available; otherwise fp16.
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16

    @staticmethod
    def _config_from_env() -> QwenImageEditConfig:
        return QwenImageEditConfig(
            model_id=os.environ.get("QWEN_EDIT_MODEL_ID", "Qwen/Qwen-Image-Edit-2511"),
            true_cfg_scale=float(os.environ.get("QWEN_EDIT_TRUE_CFG_SCALE", "1.0")),
            negative_prompt=os.environ.get("QWEN_EDIT_NEGATIVE_PROMPT", " "),
            num_inference_steps=int(os.environ.get("QWEN_EDIT_STEPS", "4")),
            guidance_scale=float(os.environ.get("QWEN_EDIT_GUIDANCE_SCALE", "1.0")),
            num_images_per_prompt=int(os.environ.get("QWEN_EDIT_NUM_IMAGES", "1")),
            lora_path=os.environ.get(
                "QWEN_EDIT_LORA_PATH",
                "lightx2v/Qwen-Image-Edit-2511-Lightning/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-fp32.safetensors",
            ),
            cpu_offload=os.environ.get("QWEN_EDIT_CPU_OFFLOAD", "0") == "1",
            unload_after_call=os.environ.get("QWEN_EDIT_UNLOAD_AFTER", "0") == "1",
        )

    @staticmethod
    def _resolve_lora_path(lora_path: str) -> str:
        """
        Supports:
        - local file path
        - "repo_id/filename" on Hugging Face Hub
        """
        if os.path.exists(lora_path):
            return lora_path

        # Try HF Hub: interpret "repo_id/filename"
        parts = lora_path.split("/")
        if len(parts) >= 2:
            repo_id = "/".join(parts[:-1])
            filename = parts[-1]
            try:
                from huggingface_hub import hf_hub_download  # type: ignore
            except Exception as e:  # pragma: no cover
                raise RuntimeError("huggingface_hub is required to download LoRA weights") from e

            return hf_hub_download(repo_id=repo_id, filename=filename)

        raise FileNotFoundError(
            f"LoRA weights not found: {lora_path}. Provide a local path or 'repo_id/filename' via QWEN_EDIT_LORA_PATH."
        )

    def load(self) -> None:
        if self._pipe is not None:
            return

        # Local import so normal runs don't pay import time / deps unless used.
        try:
            from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline  # type: ignore
            from diffusers.models import QwenImageTransformer2DModel  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Qwen image edit pipeline is not available. "
                "Install a diffusers version that includes QwenImageEditPlusPipeline "
                "(this repo's Dockerfile installs diffusers from git)."
            ) from e

        logger.info(f"Loading Qwen image editor: {self._config.model_id} (dtype={self._dtype}, device={self._device})")
        if self._config.lora_path:
            lora_resolved = self._resolve_lora_path(self._config.lora_path)
            logger.info(f"Using Qwen Lightning LoRA: {self._config.lora_path} -> {lora_resolved}")

            model = QwenImageTransformer2DModel.from_pretrained(
                self._config.model_id, subfolder="transformer", torch_dtype=self._dtype
            )

            # Matches Qwen's Lightning distillation setup (shift=3).
            scheduler_config = {
                "base_image_seq_len": 256,
                "base_shift": math.log(3),
                "invert_sigmas": False,
                "max_image_seq_len": 8192,
                "max_shift": math.log(3),
                "num_train_timesteps": 1000,
                "shift": 1.0,
                "shift_terminal": None,
                "stochastic_sampling": False,
                "time_shift_type": "exponential",
                "use_beta_sigmas": False,
                "use_dynamic_shifting": True,
                "use_exponential_sigmas": False,
                "use_karras_sigmas": False,
            }
            scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
            self._pipe = QwenImageEditPlusPipeline.from_pretrained(
                self._config.model_id,
                transformer=model,
                scheduler=scheduler,
                torch_dtype=self._dtype,
            )
            self._pipe.load_lora_weights(lora_resolved)
        else:
            self._pipe = QwenImageEditPlusPipeline.from_pretrained(self._config.model_id, torch_dtype=self._dtype)

        if self._config.cpu_offload:
            # Reduces VRAM usage; slower but safer alongside Trellis.
            self._pipe.enable_model_cpu_offload()
        else:
            self._pipe.to(self._device)

        self._pipe.set_progress_bar_config(disable=None)
        logger.info("Qwen image editor loaded.")

    def unload(self) -> None:
        if self._pipe is None:
            return
        try:
            del self._pipe
        finally:
            self._pipe = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def edit(
        self,
        image: Image.Image,
        prompt: str,
        seed: int = 0,
        *,
        true_cfg_scale: float | None = None,
        negative_prompt: str | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        num_images_per_prompt: int | None = None,
    ) -> Image.Image:
        """Returns the edited image (PIL)."""
        if not prompt or not prompt.strip():
            return image

        self.load()
        assert self._pipe is not None

        cfg = self._config
        gen_device = self._device if self._device.type == "cuda" else torch.device("cpu")
        gen = torch.Generator(device=gen_device).manual_seed(seed)

        inputs = {
            "image": [image],
            "prompt": prompt,
            "generator": gen,
            "true_cfg_scale": cfg.true_cfg_scale if true_cfg_scale is None else float(true_cfg_scale),
            "negative_prompt": cfg.negative_prompt if negative_prompt is None else str(negative_prompt),
            "num_inference_steps": cfg.num_inference_steps if num_inference_steps is None else int(num_inference_steps),
            "guidance_scale": cfg.guidance_scale if guidance_scale is None else float(guidance_scale),
            "num_images_per_prompt": cfg.num_images_per_prompt if num_images_per_prompt is None else int(num_images_per_prompt),
        }

        with torch.inference_mode():
            out = self._pipe(**inputs)
            edited = out.images[0]

        if cfg.unload_after_call:
            self.unload()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

        return edited


