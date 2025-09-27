import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import glob
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch import nn, optim
from skimage import io as skio
from skimage.metrics import peak_signal_noise_ratio

from model import INR

mid = 100

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


def indices_to_coords(idx: torch.Tensor, h: int, w: int, c: int) -> torch.Tensor:
    idx = idx.to(torch.int64)
    z = (idx % c) + 1
    hw_idx = idx // c
    y = (hw_idx % w) + 1
    x = (hw_idx // w) + 1
    return torch.stack([x, y, z], dim=1).to(dtype=torch.float32)


def split_support_query(
    img: torch.Tensor,
    n_support: int,
    n_query: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    h, w, c = img.shape
    total = h * w * c
    n_support = min(n_support, total // 2)
    n_query = min(n_query, total - n_support)
    perm = torch.randperm(total, device=img.device)
    support_idx = perm[:n_support]
    query_idx = perm[n_support : n_support + n_query]

    coords_support = indices_to_coords(support_idx, h, w, c).to(img.device)
    coords_query = indices_to_coords(query_idx, h, w, c).to(img.device)

    flat_pixels = img.view(-1, 1)
    ys = flat_pixels[support_idx]
    yq = flat_pixels[query_idx]
    return coords_support, ys, coords_query, yq


def clone_model(model: nn.Module) -> nn.Module:
    fast = INR(model.method, mid).to(next(model.parameters()).device)
    fast.load_state_dict(model.state_dict())
    return fast


def maml_train(
    method: str = "GKAN",
    data_root: str = "../../dataset/STI",
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
    base_model = INR(method, mid=mid).to(device)
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

            xs, ys, xq, yq = split_support_query(img, n_support, n_query)

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

    # Save meta-learned weights in repo-level checkpoints directory
    ckpt_dir = Path(__file__).resolve().parents[2] / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"maml_{method.lower()}_{mid}.pth"
    torch.save({"state_dict": base_model.state_dict(), "method": method}, ckpt_path)
    print(f"Saved meta-learned weights to {ckpt_path}")


def main():
    maml_train()

if __name__ == "__main__":
    main()
