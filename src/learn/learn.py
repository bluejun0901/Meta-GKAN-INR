import argparse
import logging
import random
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from skimage import io as skio
from skimage.metrics import peak_signal_noise_ratio
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from src.logging.train_logger import TrainLogger


class BaseLearner(ABC):
    @abstractmethod
    def train(self, seed: int = 1):
        pass


class Learner(BaseLearner):
    def __init__(
        self,
        image_path: str,
        run_dir: str | Path,
        model: nn.Module,
        model_name: str,
        steps: int = 1000,
        learning_rate: float = 1e-3,
        batch_size: int = 100000,
        image_save_steps: int = 50,
        meta_learn: bool = False,
        meta_path: str | None = None,
    ):
        # Basic hyperparameters
        self.model = model
        self.model_name = model_name
        self.image_path: str = image_path
        # Accept str in config, store as Path
        self.run_dir: Path = Path(run_dir)
        self.artifact_path: Path = self.run_dir / "learn" / "artifacts"
        self.model_path: Path = self.run_dir / "learn" / "models"
        self.log_path: Path = self.run_dir / "learn" / "logs"
        self.artifact_path.mkdir(parents=True, exist_ok=True)
        self.model_path.mkdir(parents=True, exist_ok=True)
        self.log_path.mkdir(parents=True, exist_ok=True)

        self.steps: int = steps
        self.learning_rate: float = learning_rate
        self.batch_size: int = batch_size
        self.image_save_steps: int = image_save_steps
        self.meta_learn: bool = meta_learn
        self.meta_path: str | None = meta_path

        self.logger = logging.getLogger(__name__)

        self.metrics_logger: TrainLogger = TrainLogger(
            run_dir=self.run_dir,
            name="learn",
            auto_draw=True,
            draw_freq=100,
            draw_kwargs={"y_axis": ["psnr", "psnr_best", "loss"], "x_axis": "step"},
        )

    def _set_seed(self, seed: int = 1) -> None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def _device(self) -> torch.device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return device

    def _load_image_tensor(self, path: str, device: torch.device) -> torch.Tensor:
        img_np = skio.imread(path)
        if img_np.ndim == 2:
            img_np = img_np[:, :, None]
        if img_np.shape[-1] == 4:
            img_np = img_np[:, :, :3]
        if np.issubdtype(img_np.dtype, np.integer):
            img_np = img_np.astype(np.float32) / np.iinfo(img_np.dtype).max
        else:
            img_np = img_np.astype(np.float32)
            if img_np.max() > 1.0:
                img_np = img_np / img_np.max()
        img = torch.from_numpy(img_np).to(device=device, dtype=torch.float32)
        return img

    def _make_coords(self, h: int, w: int, c: int) -> torch.Tensor:
        x_in = torch.arange(1, h + 1, dtype=torch.float32)
        y_in = torch.arange(1, w + 1, dtype=torch.float32)
        z_in = torch.arange(1, c + 1, dtype=torch.float32)
        x_in, y_in, z_in = torch.meshgrid(x_in, y_in, z_in, indexing="ij")
        coords = torch.stack((x_in.reshape(-1), y_in.reshape(-1), z_in.reshape(-1)), dim=1)
        return coords

    def _prepare_loader(self, img: torch.Tensor, batch_size: int) -> tuple[DataLoader, torch.Tensor, torch.Tensor]:
        h, w, c = img.shape
        coords = self._make_coords(h, w, c)
        pixels = img.reshape(-1, 1).detach().cpu().to(dtype=torch.float32)
        dataset = TensorDataset(coords, pixels)
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=torch.cuda.is_available(),
        )
        return data_loader, coords, img

    def _reconstruct_full(
        self,
        model: torch.nn.Module,
        coords: torch.Tensor,
        img: torch.Tensor,
        device: torch.device,
    ) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            outs = []
            for start in range(0, coords.shape[0], self.batch_size):
                chunk = coords[start : start + self.batch_size].to(device)
                out = model(chunk).detach().cpu()
                outs.append(out)
            recon = torch.cat(outs, dim=0).reshape(img.shape).cpu().numpy()
        model.train()
        return recon

    def train(self, seed: int = 1) -> None:
        self._set_seed(seed)
        device = self._device()

        gt = self._load_image_tensor(self.image_path, device)
        data_loader, coords, gt = self._prepare_loader(gt, self.batch_size)
        data_iter = iter(data_loader)

        model = self.model.to(device)

        if self.meta_learn and self.meta_path:
            checkpoint = torch.load(self.meta_path, map_location="cpu")
            state_dict = (
                checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
            )
            model.load_state_dict(state_dict)

        params = list(model.parameters())
        n_params = sum(int(np.prod(list(p.size()))) for p in params)
        self.logger.info(f"Number of params: {n_params}")

        optimizer = optim.Adam(params, lr=self.learning_rate, weight_decay=0)
        ps_best = 0.0

        for i in range(self.steps):
            self.logger.info(f"Step {i + 1}/{self.steps} - Best PSNR: {ps_best:.4f}")
            try:
                batch_coords, batch_pixels = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                batch_coords, batch_pixels = next(data_iter)

            batch_coords = batch_coords.to(device)
            batch_pixels = batch_pixels.to(device)

            preds = model(batch_coords)
            loss = F.mse_loss(preds, batch_pixels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % self.image_save_steps == 0:
                recon = self._reconstruct_full(model, coords, gt, device)
                ps_here = peak_signal_noise_ratio(gt.detach().cpu().numpy(), np.clip(recon, 0, 1))
                ps_best = max(ps_best, ps_here)
                self.metrics_logger.log(step=i, psnr=ps_here, psnr_best=ps_best, loss=loss.item())

                arr = np.clip(recon, 0, 1)
                img = Image.fromarray((arr * 255).astype(np.uint8))
                img.save(self.artifact_path / f"{i}.png")

        # Save final model
        model_save_path = self.model_path / self.model_name
        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict()}, model_save_path)


def main():
    import hydra

    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="path to configuration file")
    args = parser.parse_args()

    file_config = OmegaConf.load(args.config_path)

    learner: BaseLearner = hydra.utils.instantiate(file_config)
    learner.train()


if __name__ == "__main__":
    main()
