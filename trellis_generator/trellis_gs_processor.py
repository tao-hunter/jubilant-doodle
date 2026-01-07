import gc
import random
import os
from io import BytesIO
from PIL import Image

import ray
import torch
import torch.distributed as dist

from loguru import logger
from trellis_generator.pipelines import TrellisImageTo3DPipeline
from trellis_generator.qwen_image_editor import QwenImageEditor
from background_remover.ray_bg_remover import RayBGRemoverProcessor
from background_remover.bg_removers.birefnet_bg_remover import BiRefNetBGRemover
from background_remover.utils.rand_utils import secure_randint, set_random_seed


class GaussianProcessor:
    """Generates 3d models and videos"""

    # Hard-coded Qwen edit prompt and parameters for consistent 3D-friendly inputs.
    QWEN_EDIT_PROMPT: str = (
        "Show this object in three-quarters view and make sure it is fully visible. "
        "Turn background neutral solid color contrasting with an object. "
        "Delete background details. Delete watermarks. Keep object colors. "
        "Sharpen image details"
    )
    QWEN_EDIT_SEED: int = 0
    QWEN_EDIT_TRUE_CFG_SCALE: float = 1.0
    QWEN_EDIT_NEGATIVE_PROMPT: str = " "
    QWEN_EDIT_NUM_INFERENCE_STEPS: int = 4
    QWEN_EDIT_GUIDANCE_SCALE: float = 1.0
    QWEN_EDIT_NUM_IMAGES_PER_PROMPT: int = 1

    def __init__(self, image_shape: tuple[int, int, int], vllm_flash_attn_backend: str = "FLASHINFER") -> None:
        logger.info(f"VLLM FLASH ATTENTION backend: {vllm_flash_attn_backend}")
        logger.info(f"TRELLIS ATTENTION backend: {os.environ['ATTN_BACKEND']}")

        self._bg_removers_workers: list[RayBGRemoverProcessor] = []
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._image_to_3d_pipeline: TrellisImageTo3DPipeline | None = None
        self._qwen_editor: QwenImageEditor | None = None
        self.gaussians: torch.Tensor | None = None

    def load_models(self, model_name: str = "microsoft/TRELLIS-image-large") -> None:
        """ Function for preloading all essential models for image -> 3D pipeline """

        self._image_to_3d_pipeline = TrellisImageTo3DPipeline.from_pretrained(model_name)
        self._image_to_3d_pipeline.to(self._device)

        # BG removal can be VRAM-heavy. This miner uses BiRefNet only.
        # - BG_REMOVER_DEVICE=cuda|cpu (default: cuda)
        bg_device = os.environ.get("BG_REMOVER_DEVICE", "cuda").lower()
        use_gpu = (bg_device != "cpu") and torch.cuda.is_available()

        if use_gpu:
            self._bg_removers_workers = [RayBGRemoverProcessor.remote(BiRefNetBGRemover)]
        else:
            self._bg_removers_workers = [RayBGRemoverProcessor.options(num_gpus=0).remote(BiRefNetBGRemover)]
        torch.cuda.empty_cache()

        # Preload Qwen image-edit so first request doesn't pay cold-start latency.
        # Disable via env if needed: QWEN_EDIT_PRELOAD=0
        if os.environ.get("QWEN_EDIT_PRELOAD", "1") == "1":
            if self._qwen_editor is None:
                self._qwen_editor = QwenImageEditor(device=self._device)
            self._qwen_editor.load()

    def unload_models(self) -> None:
        """  Function for unloading all models for image -> 3D pipeline """

        for worker in self._bg_removers_workers:
            worker.unload_model.remote()
        dist.destroy_process_group()

        if self._qwen_editor is not None:
            self._qwen_editor.unload()
            self._qwen_editor = None

        del self._image_to_3d_pipeline
        del self.gaussians

        self._image_to_3d_pipeline = None
        self.gaussians = None

        gc.collect()
        torch.cuda.empty_cache()

    def warmup_generator(self):
        """ Function for warming up the generator. """

        # Warmup should not load extra models (like Qwen edit).
        # Also: Trellis preprocess expects a non-empty alpha mask; use a safe RGBA dummy.
        dummy = Image.new("RGBA", (64, 64), color=(128, 128, 128, 255))
        self.get_model_from_image_as_ply_obj(image=dummy, seed=0, apply_qwen_edit=False)

    @staticmethod
    def _get_random_index_cycler(list_size: int):
        """
        Creates a generator that yields random indices without repetition.
        When all indices are exhausted, it reshuffles and continues.
        """

        while True:
            indices = list(range(list_size))
            random.shuffle(indices)
            for idx in indices:
                yield idx

    def _remove_background(self, image: Image.Image, seed: int) -> Image.Image:
        """ Function for removing background from the image. """

        if not self._bg_removers_workers:
            return image

        # Single BG worker (BiRefNet) -> single output image; no selector.
        return ray.get(self._bg_removers_workers[0].run.remote(image))

    def _edit_image_for_3d_style(
        self,
        image: Image.Image,
        edit_prompt: str,
        edit_seed: int,
        *,
        true_cfg_scale: float | None = None,
        negative_prompt: str | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        num_images_per_prompt: int | None = None,
    ) -> Image.Image:
        """Optional style/edit step that runs BEFORE background removal."""
        if self._qwen_editor is None:
            self._qwen_editor = QwenImageEditor(device=self._device)
        return self._qwen_editor.edit(
            image=image,
            prompt=edit_prompt,
            seed=edit_seed,
            true_cfg_scale=true_cfg_scale,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
        )

    def _generate_3d_object(self, images_no_bg: list[Image.Image], seed: int) -> BytesIO:
        """Generate a 3D object from one or more input images without background."""

        if seed < 0:
            set_seed = secure_randint(0, 10000)
            set_random_seed(set_seed)
        else:
            set_random_seed(seed)

        if len(images_no_bg) == 1:
            outputs = self._image_to_3d_pipeline.run(images_no_bg[0])
        else:
            outputs = self._image_to_3d_pipeline.run_multi_image(images_no_bg)
        self.gaussians = outputs["gaussian"][0]

        buffer = BytesIO()
        self.gaussians.save_ply(buffer)
        buffer.seek(0)

        return buffer

    def get_model_from_image_as_ply_obj(
        self,
        image: Image.Image,
        seed: int = -1,
        *,
        apply_qwen_edit: bool = True,
    ) -> tuple[BytesIO, Image.Image]:
        """Generate 3D model from image(s) (Qwen edit -> BG removal -> Trellis multi-image)."""

        # 1) Remove background from the original image
        original_has_alpha = image.mode in ("LA", "RGBA", "PA")
        original_no_bg = image if original_has_alpha else self._remove_background(image, seed)

        # 2) Optionally edit, then remove background from edited image
        if apply_qwen_edit:
            logger.info("Applying Qwen image edit (pre background-removal) ...")
            edited = self._edit_image_for_3d_style(
                image,
                edit_prompt=self.QWEN_EDIT_PROMPT,
                edit_seed=self.QWEN_EDIT_SEED,
                true_cfg_scale=self.QWEN_EDIT_TRUE_CFG_SCALE,
                negative_prompt=self.QWEN_EDIT_NEGATIVE_PROMPT,
                num_inference_steps=self.QWEN_EDIT_NUM_INFERENCE_STEPS,
                guidance_scale=self.QWEN_EDIT_GUIDANCE_SCALE,
                num_images_per_prompt=self.QWEN_EDIT_NUM_IMAGES_PER_PROMPT,
            )
            edited_has_alpha = edited.mode in ("LA", "RGBA", "PA")
            edited_no_bg = edited if edited_has_alpha else self._remove_background(edited, seed)
            images_for_3d = [original_no_bg, edited_no_bg]
        else:
            edited_no_bg = original_no_bg
            images_for_3d = [original_no_bg]

        buffer = self._generate_3d_object(images_for_3d, seed)
        return buffer, edited_no_bg
