import gc
import os
import random
from io import BytesIO
from PIL import Image

import ray
import torch
import torch.distributed as dist
import numpy as np

from trellis_generator.pipelines import TrellisImageTo3DPipeline
from background_remover.ray_bg_remover import RayBGRemoverProcessor
from background_remover.bg_removers.ben2_bg_remover import Ben2BGRemover
from background_remover.bg_removers.birefnet_bg_remover import BiRefNetBGRemover
from background_remover.image_selector import ImageSelector


def secure_randint(low: int, high: int) -> int:
    """Return a random integer in [low, high] using os.urandom."""
    range_size = high - low + 1
    num_bytes = 4
    max_int = 2**(8 * num_bytes) - 1

    while True:
        rand_bytes = os.urandom(num_bytes)
        rand_int = int.from_bytes(rand_bytes, 'big')
        if rand_int <= max_int - (max_int % range_size):
            return low + (rand_int % range_size)


class GaussianProcessor:
    """Generates 3d models and videos"""

    def __init__(self, image_shape: tuple[int, int, int], vllm_flash_attn_backend: str = "FLASHINFER") -> None:
        self._bg_removers_workers: list[RayBGRemoverProcessor] = []
        self._vlm_image_selector = ImageSelector(3, image_shape, vllm_flash_attn_backend)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._image_to_3d_pipeline: TrellisImageTo3DPipeline | None = None
        self.gaussians: torch.Tensor | None = None

    def load_models(self, model_name: str = "microsoft/TRELLIS-image-large") -> None:
        """ Function for preloading all essential models for image -> 3D pipeline """

        self._image_to_3d_pipeline = TrellisImageTo3DPipeline.from_pretrained(model_name)
        self._image_to_3d_pipeline.to(self._device)

        self._bg_removers_workers: list[RayBGRemoverProcessor] = [
            RayBGRemoverProcessor.remote(Ben2BGRemover),
            RayBGRemoverProcessor.remote(BiRefNetBGRemover),
        ]
        torch.cuda.empty_cache()
        self._vlm_image_selector.load_model()

    def unload_models(self) -> None:
        """  Function for unloading all models for image -> 3D pipeline """

        for worker in self._bg_removers_workers:
            worker.unload_model.remote()
        dist.destroy_process_group()

        del self._image_to_3d_pipeline
        del self.gaussians

        self._image_to_3d_pipeline = None
        self.gaussians = None

        gc.collect()
        torch.cuda.empty_cache()

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

    def _remove_background(self, image: Image.Image) -> Image.Image:
        """ Function for removing background from the image. """

        futurs = [worker.run.remote(image) for worker in self._bg_removers_workers]
        results = ray.get(futurs)
        image1 = results[0][0]
        image2 = results[1][0]
        output_image = self._vlm_image_selector.select_with_image_selector(image1, image2, image)
        return output_image

    def _generate_3d_object(self, image_no_bg: Image.Image, seed: int) -> BytesIO:
        """ Function for generating a 3D object using an input image without background. """

        if seed < 0:
            set_seed = secure_randint(0, 10000)
        else:
            set_seed = seed

        outputs = self._image_to_3d_pipeline.run(
            image_no_bg,
            seed=set_seed
        )
        self.gaussians = outputs["gaussian"][0]

        T = np.array([0, 0, 0])
        R = self.gaussians.rotate_by_euler_angles(90.0, 0.0, 0.0)
        self.gaussians.transform_data(T, R)

        buffer = BytesIO()
        self.gaussians.save_ply(buffer)
        buffer.seek(0)

        return buffer

    def get_model_from_image_as_ply_obj(self, image: Image.Image, seed: int = -1) -> tuple[BytesIO, Image.Image]:
        """ Generate 3D model using image as ab input """

        has_alpha = image.mode in ("LA", "RGBA", "PA")
        if not has_alpha:
            output_image = self._remove_background(image)
        else:
            output_image = image
        buffer = self._generate_3d_object(output_image, seed)

        return buffer, output_image
