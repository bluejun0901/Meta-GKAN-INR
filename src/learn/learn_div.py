import logging
import random
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from skimage import io as skio
from skimage.metrics import peak_signal_noise_ratio
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from src.logging.train_logger import TrainLogger
from src.utils import pad_image_to_block_size, separate_frequencies


class BaseLearner(ABC):
    @abstractmethod
    def train(self, seed: int = 1):
        pass


class DivLearner(BaseLearner):
    def __init__(
        self,
        image_path: str,
        run_dir: str | Path,
        model_big: nn.Module,
        model_small: nn.Module,
        block_size: int = 64,
        cutoff: int = 16,
        model_name: str = "div_model",
        steps: int = 1000,
        learning_rate: float = 1e-3,
        batch_size: int = 100000,
        image_save_steps: int = 50,
    ):
        # Basic hyperparameters
        self.model_big: nn.Module = model_big
        self.model_small: nn.Module = model_small
        self.model_name = model_name
        self.block_size: int = block_size
        self.cutofff: int = cutoff
        self.image_path: str = image_path

        # Accept str in config, store as Path
        self.run_dir: Path = Path(run_dir)
        self.artifact_path: Path = self.run_dir / "learn" / "artifacts"
        self.model_path: Path = self.run_dir / "learn" / "models"
        self.artifact_path.mkdir(parents=True, exist_ok=True)
        self.model_path.mkdir(parents=True, exist_ok=True)

        self.steps: int = steps
        self.learning_rate: float = learning_rate
        self.batch_size: int = batch_size
        self.image_save_steps: int = image_save_steps

        self.logger = logging.getLogger(__name__)

        self.metrics_logger_big: TrainLogger = TrainLogger(
            run_dir=self.run_dir,
            name="learn_big",
            auto_draw=True,
            draw_freq=100,
            draw_kwargs={"y_axis": ["psnr", "psnr_best", "loss"], "x_axis": "step"},
        )

        self.metrics_logger_small: TrainLogger = TrainLogger(
            run_dir=self.run_dir,
            name="learn_small",
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

    def _normalize_image(self, img_np: np.ndarray) -> np.ndarray:
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
        return img_np

    def _load_image_tensor(self, path: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        img_np = skio.imread(path)
        img_np = pad_image_to_block_size(img_np, block_size=self.block_size)

        original_pad_save = Image.fromarray(img_np)
        original_pad_save.save(self.artifact_path / "original_padded.png")

        low_freq_img_np, high_freq_img_np = separate_frequencies(img_np, block_size=self.block_size, cutoff=self.cutofff)

        low_freq_img_np = self._normalize_image(low_freq_img_np)
        high_freq_img_np = self._normalize_image(high_freq_img_np)
        low_freq_img = torch.from_numpy(low_freq_img_np).to(device=device, dtype=torch.float32)
        high_freq_img = torch.from_numpy(high_freq_img_np).to(device=device, dtype=torch.float32)

        low_save = Image.fromarray((np.clip(low_freq_img_np, 0, 1) * 255).astype(np.uint8))
        low_save.save(self.artifact_path / "low_frequency.png")
        high_save = Image.fromarray((np.clip(high_freq_img_np, 0, 1) * 255).astype(np.uint8))
        high_save.save(self.artifact_path / "high_frequency.png")
        return low_freq_img, high_freq_img

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
    
    def train_high(self, high_img: torch.Tensor) -> None:
        device = self._device()

        data_loader, coords, gt = self._prepare_loader(high_img, self.batch_size)
        data_iter = iter(data_loader)

        model = self.model_big.to(device)

        params = list(model.parameters())
        n_params = sum(int(np.prod(list(p.size()))) for p in params)
        self.logger.info(f"Number of params: {n_params}")

        optimizer = optim.Adam(params, lr=self.learning_rate, weight_decay=0)
        ps_best = 0.0

        for i in range(self.steps):
            self.logger.info(f"high freq Step {i + 1}/{self.steps} - Best PSNR: {ps_best:.4f}")
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
                self.metrics_logger_big.log(step=i, psnr=ps_here, psnr_best=ps_best, loss=loss.item())

                arr = np.clip(recon, 0, 1)
                img = Image.fromarray((arr * 255).astype(np.uint8))
                img.save(self.artifact_path / f"high{i}.png")

        # Save final model
        model_save_path = self.model_path / self.model_name / "high_freq.pth"
        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict()}, model_save_path)

    def train_low(self, low_img: torch.Tensor) -> None:
        device = self._device()

        data_loader, coords, gt = self._prepare_loader(low_img, self.batch_size)
        data_iter = iter(data_loader)

        model = self.model_small.to(device)

        params = list(model.parameters())
        n_params = sum(int(np.prod(list(p.size()))) for p in params)
        self.logger.info(f"Number of params: {n_params}")

        optimizer = optim.Adam(params, lr=self.learning_rate, weight_decay=0)
        ps_best = 0.0

        for i in range(self.steps):
            self.logger.info(f"low freq Step {i + 1}/{self.steps} - Best PSNR: {ps_best:.4f}")
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
                self.metrics_logger_small.log(step=i, psnr=ps_here, psnr_best=ps_best, loss=loss.item())

                arr = np.clip(recon, 0, 1)
                img = Image.fromarray((arr * 255).astype(np.uint8))
                img.save(self.artifact_path / f"low{i}.png")

        # Save final model
        model_save_path = self.model_path / self.model_name / "low_freq.pth"
        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict()}, model_save_path)

    def train(self, seed: int = 1) -> None:
        self._set_seed(seed)
        device = self._device()

        low_img, high_img = self._load_image_tensor(self.image_path, device)
        
        self.train_low(low_img)
        self.train_high(high_img)
        