import os
import argparse
import json
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf, ListConfig, DictConfig
from PIL import Image
from skimage import io as skio
from skimage.metrics import peak_signal_noise_ratio
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from models.model import INR


@dataclass
class LearnerConfig:
    image_path: str
    save_path: str
    method: str = "GKAN"
    steps: int = 1000
    learning_rate: float = 1e-3
    batch_size: int = 100000
    image_save_steps: int = 50
    mid: int | None = None
    meta_learn: bool = False
    meta_path: str | None = None


class Learner:
    def __init__(self, config: LearnerConfig | ListConfig | DictConfig):
        # Basic hyperparameters
        self.method: str = config.method
        self.image_path: str = config.image_path
        # Accept str in config, store as Path
        self.save_path: Path = (
            Path(config.save_path)
            if not isinstance(config.save_path, Path)
            else config.save_path
        )
        self.steps: int = config.steps
        self.learning_rate: float = config.learning_rate
        self.batch_size: int = config.batch_size
        self.image_save_steps: int = config.image_save_steps
        self.mid: int | None = config.mid
        self.meta_learn: bool = config.meta_learn
        self.meta_path: str | None = config.meta_path

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
        x_in, y_in, z_in = torch.meshgrid(x_in, y_in, z_in)
        coords = torch.stack(
            (x_in.reshape(-1), y_in.reshape(-1), z_in.reshape(-1)), dim=1
        )
        return coords

    def _prepare_loader(
        self, img: torch.Tensor, batch_size: int
    ) -> tuple[DataLoader, torch.Tensor, torch.Tensor]:
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

        model = INR(self.method, self.mid).to(device)

        if self.meta_learn and self.meta_path:
            checkpoint = torch.load(self.meta_path, map_location="cpu")
            state_dict = (
                checkpoint["state_dict"]
                if isinstance(checkpoint, dict) and "state_dict" in checkpoint
                else checkpoint
            )
            model.load_state_dict(state_dict)

        params = list(model.parameters())
        n_params = sum(int(np.prod(list(p.size()))) for p in params)
        print(f"Number of params: {n_params}")

        optimizer = optim.Adam(params, lr=self.learning_rate, weight_decay=0)
        ps_best = 0.0
        self.save_path.mkdir(parents=True, exist_ok=True)
        psnr_record: list[dict] = []

        for i in range(self.steps):
            print("\r", i, ps_best, end="\r\r")
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
                ps_here = peak_signal_noise_ratio(
                    gt.detach().cpu().numpy(), np.clip(recon, 0, 1)
                )
                psnr_record.append(
                    {
                        "name": str(self.save_path).replace("/", "_").replace("_", " "),
                        "step": int(i),
                        "params": int(n_params),
                        "psnr": float(ps_here),
                        "psnr_best": float(ps_best),
                        "loss": float(loss.item()),
                    }
                )
                ps_best = max(ps_best, ps_here)

                arr = np.clip(recon, 0, 1)
                img = Image.fromarray((arr * 255).astype(np.uint8))
                meta_str = "_meta" if self.meta_learn else ""
                img.save(self.save_path / f"{self.method}{i}{meta_str}.png")

        # Persist logs atomically
        log_save_path = self.save_path / "log.json"
        log_save_path.parent.mkdir(parents=True, exist_ok=True)
        if log_save_path.exists():
            try:
                with open(log_save_path, "r", encoding="utf-8") as f:
                    existing_records = json.load(f)
                if not isinstance(existing_records, list):
                    existing_records = []
            except json.JSONDecodeError:
                existing_records = []
        else:
            existing_records = []

        combined = existing_records + psnr_record
        with tempfile.NamedTemporaryFile(
            mode="w", dir=log_save_path.parent, delete=False, encoding="utf-8"
        ) as tmp_file:
            json.dump(combined, tmp_file)
            temp_name = tmp_file.name
        os.replace(temp_name, log_save_path)

        # Save final model
        model_save_path = self.save_path / "model.pth"
        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": model.state_dict(), "method": self.method}, model_save_path
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="path to configuration file")
    args = parser.parse_args()

    file_config = OmegaConf.load(args.config_path)
    base_config = OmegaConf.structured(LearnerConfig)
    config = OmegaConf.merge(base_config, file_config)

    learner = Learner(config)
    learner.train()


if __name__ == "__main__":
    main()
