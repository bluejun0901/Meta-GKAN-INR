import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from pathlib import Path
from dotenv import load_dotenv
import argparse
load_dotenv()

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT")).resolve() # type: ignore
MODEL_ROOT = Path(os.getenv("MODEL_ROOT")).resolve() # type: ignore
IMAGE_ROOT = Path(os.getenv("IMAGE_ROOT")).resolve() # type: ignore
CONFIG_ROOT = Path(os.getenv("CONFIG_ROOT")).resolve() # type: ignore
LOG_ROOT = Path(os.getenv("LOG_ROOT")).resolve() # type: ignore

import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio
from skimage import io as skio
import random
from omegaconf import OmegaConf
from model import INR
import json
import tempfile

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_path", 
        type=str,
        help="reletive path to configuration file"
    )
    args = parser.parse_args()
    config = OmegaConf.load(CONFIG_ROOT / args.config_path)

    method = config.method
    seed=1
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    image_path = IMAGE_ROOT / config.image_path
    max_iter = config.steps
    lr_real = config.learning_rate

    # Load image and normalize to [0,1], handle grayscale/RGBA
    img_np = skio.imread(image_path)
    if img_np.ndim == 2:
        img_np = img_np[:, :, None]
    # Drop alpha channel if present
    if img_np.shape[-1] == 4:
        img_np = img_np[:, :, :3]
    if np.issubdtype(img_np.dtype, np.integer):
        img_np = img_np.astype(np.float32) / np.iinfo(img_np.dtype).max
    else:
        img_np = img_np.astype(np.float32)
        if img_np.max() > 1.0:
            img_np = img_np / img_np.max()

    gt_np = img_np
    gt = torch.from_numpy(gt_np).to(device=device, dtype=torch.float32)

    n_1, n_2, n_3 = gt.shape

    x_in = torch.arange(1, n_1 + 1, dtype=torch.float32)
    y_in = torch.arange(1, n_2 + 1, dtype=torch.float32)
    z_in = torch.arange(1, n_3 + 1, dtype=torch.float32)
    x_in, y_in, z_in = torch.meshgrid(x_in, y_in, z_in)
    coords = torch.stack(
        (x_in.reshape(-1), y_in.reshape(-1), z_in.reshape(-1)), dim=1
    )

    pixels = torch.from_numpy(gt_np.reshape(-1, 1)).to(dtype=torch.float32)
    dataset = TensorDataset(coords, pixels)
    batch_size = config.batch_size if "batch_size" in config else 100000
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )
    data_iter = iter(data_loader)

    model = INR(method, config.mid).to(device)
    if config.meta_learn:
        checkpoint_path = MODEL_ROOT / config.meta_path
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)

    params = list(model.parameters())

    s = sum([np.prod(list(p.size())) for p in params]); 
    print('Number of params: %d' % s)

    optimizier = optim.Adam(params, lr=lr_real, weight_decay=0)

    ps_best = 0

    save_path = IMAGE_ROOT / config.save_path
    save_path.mkdir(parents=True, exist_ok=True)
    psnr_record = []
    for i in range(max_iter):
        print('\r', i, ps_best, end='\r\r')
        try:
            batch_coords, batch_pixels = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            batch_coords, batch_pixels = next(data_iter)

        batch_coords = batch_coords.to(device)
        batch_pixels = batch_pixels.to(device)

        preds = model(batch_coords)
        loss = F.mse_loss(preds, batch_pixels)
        optimizier.zero_grad()
        loss.backward()
        optimizier.step()

        if i % config.image_save_steps == 0:
            with torch.no_grad():
                recon_chunks = []
                for start in range(0, coords.shape[0], batch_size):
                    chunk = coords[start:start + batch_size].to(device)
                    out = model(chunk).detach().cpu()
                    recon_chunks.append(out)
                full_recon = torch.cat(recon_chunks, dim=0).reshape(gt.shape).cpu().numpy()

            ps_here = peak_signal_noise_ratio(
                gt_np,
                np.clip(full_recon, 0, 1)
            )
            cur_log = {
                "name": str(config.save_path.replace("/", "_").replace("_", " ")),
                "step": int(i),
                "params": int(s),
                "psnr": float(ps_here),
                "psnr_best": float(ps_best),
                "loss": float(loss.item())
            }
            psnr_record.append(cur_log)
            if ps_here > ps_best:
                ps_best = ps_here
            
            arr = np.clip(full_recon, 0, 1)
            img = Image.fromarray((arr * 255).astype(np.uint8))
            meta_str = "_meta" if config.meta_learn else ""
            img.save(save_path / f"{method}{i}{meta_str}.png")

    log_save_path = LOG_ROOT / config.save_path / "log.json"
    log_save_path.parent.mkdir(parents=True, exist_ok=True)
    if log_save_path.exists():
        try:
            with open(log_save_path, "r", encoding="utf-8") as existing_file:
                existing_records = json.load(existing_file)
            if not isinstance(existing_records, list):
                existing_records = []
        except json.JSONDecodeError:
            existing_records = []
    else:
        existing_records = []

    combined_records = existing_records + psnr_record

    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=log_save_path.parent,
        delete=False,
        encoding="utf-8"
    ) as tmp_file:
        json.dump(combined_records, tmp_file)
        temp_name = tmp_file.name

    os.replace(temp_name, log_save_path)

    model_save_path = MODEL_ROOT / config.save_path / "model.pth"
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "method": method}, model_save_path)

if __name__ == "__main__":
    main()
