import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

import glob
import random
from typing import List, Tuple

import numpy as np
import torch
from torch import nn, optim
from skimage import io as skio
from skimage.metrics import peak_signal_noise_ratio

from model import INR


def list_images(root: str) -> List[str]:
    exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.ppm", "*.pgm"]
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(root, "**", ext), recursive=True))
    return sorted(paths)


def load_image_tensor(path: str, device: torch.device) -> torch.Tensor:
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
    img = torch.from_numpy(img_np).to(device)
    return img


def make_coords(h: int, w: int, c: int, device: torch.device) -> torch.Tensor:
    x = torch.arange(1, h + 1, device=device, dtype=torch.float32)
    y = torch.arange(1, w + 1, device=device, dtype=torch.float32)
    z = torch.arange(1, c + 1, device=device, dtype=torch.float32)
    X, Y, Z = torch.meshgrid(x, y, z)
    coords = torch.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], dim=1)
    return coords.to(device=device, dtype=torch.float32)


def split_support_query(
    coords: torch.Tensor,
    pixels: torch.Tensor,
    n_support: int,
    n_query: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    N = coords.shape[0]
    idx = torch.randperm(N, device=coords.device)
    n_support = min(n_support, N // 2)
    n_query = min(n_query, N - n_support)
    is_idx = idx[:n_support]
    iq_idx = idx[n_support : n_support + n_query]
    return coords[is_idx], pixels[is_idx], coords[iq_idx], pixels[iq_idx]


def clone_model(model: nn.Module) -> nn.Module:
    fast = INR(model.method).to(next(model.parameters()).device)
    fast.load_state_dict(model.state_dict())
    return fast


def maml_train(
    method: str = "GKAN",
    data_root: str = "dataset/STI",
    meta_iters: int = 500,
    meta_batch_size: int = 4,
    inner_steps: int = 5,
    inner_lr: float = 1e-2,
    meta_lr: float = 1e-3,
    n_support: int = 4096,
    n_query: int = 4096,
    seed: int = 1,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Collect tasks (images)
    img_paths = list_images(data_root)
    if len(img_paths) == 0:
        raise RuntimeError(f"No images found under {data_root}")

    # Initialize base model and meta-optimizer
    base_model = INR(method).to(device)
    base_model.method = method  # type: ignore
    meta_opt = optim.Adam(base_model.parameters(), lr=meta_lr)
    mse = nn.MSELoss()

    print(f"Tasks: {len(img_paths)} images. Device: {device}.")
    total_params = sum(p.numel() for p in base_model.parameters())
    print(f"Base model params: {total_params}")

    for it in range(1, meta_iters + 1):
        meta_opt.zero_grad()

        batch_paths = random.sample(img_paths, k=min(meta_batch_size, len(img_paths)))
        task_psnr = []

        for pth in batch_paths:
            # Prepare task data
            img = load_image_tensor(pth, device)
            h, w, c = img.shape
            coords = make_coords(h, w, c, device)
            pixels = img.view(-1, 1)  # target per (x,y,channel)

            xs, ys, xq, yq = split_support_query(coords, pixels, n_support, n_query)

            # Inner-loop adaptation on a cloned model (FO-MAML)
            fast_model = clone_model(base_model)
            fast_opt = optim.SGD(fast_model.parameters(), lr=inner_lr)

            for _ in range(inner_steps):
                pred_s = fast_model(xs)
                loss_s = mse(pred_s, ys)
                fast_opt.zero_grad()
                loss_s.backward()
                fast_opt.step()

            # Evaluate on query and accumulate grads w.r.t. fast weights
            pred_q = fast_model(xq)
            loss_q = mse(pred_q, yq)

            # Compute PSNR for monitoring on query subset
            with torch.no_grad():
                pq = pred_q.clamp(0, 1).detach().cpu().numpy()
                yq_np = yq.detach().cpu().numpy()
                try:
                    ps = peak_signal_noise_ratio(yq_np, pq, data_range=1.0)
                except Exception:
                    ps = 0.0
                task_psnr.append(ps)

            # First-order MAML gradient accumulation: copy grads from fast to base
            grads = torch.autograd.grad(loss_q, tuple(fast_model.parameters()))
            for g, p in zip(grads, base_model.parameters()):
                if p.grad is None:
                    p.grad = g.detach().clone()
                else:
                    p.grad = p.grad + g.detach()

        # Average gradients across tasks and take a meta step
        for p in base_model.parameters():
            if p.grad is not None:
                p.grad /= len(batch_paths)
        meta_opt.step()

        if it % 10 == 0:
            avg_psnr = float(np.mean(task_psnr)) if task_psnr else 0.0
            print(f"[Meta {it:04d}/{meta_iters}] Avg Query PSNR: {avg_psnr:.2f} dB")

    # Save meta-learned weights
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = os.path.join("checkpoints", f"maml_{method.lower()}.pth")
    torch.save({"state_dict": base_model.state_dict(), "method": method}, ckpt_path)
    print(f"Saved meta-learned weights to {ckpt_path}")


def main():
    maml_train()

if __name__ == "__main__":
    main()
