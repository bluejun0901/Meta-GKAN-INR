import os
os.environ["CUDA_VISIBLE_DEVICES"] = "9"
from pathlib import Path
from dotenv import load_dotenv
import argparse
load_dotenv()

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT")).resolve() # type: ignore
MODEL_ROOT = Path(os.getenv("MODEL_ROOT")).resolve() # type: ignore
IMAGE_ROOT = Path(os.getenv("IMAGE_ROOT")).resolve() # type: ignore
CONFIG_ROOT = Path(os.getenv("CONFIG_ROOT")).resolve() # type: ignore

import torch
from torch import optim 
import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio
from skimage import io as skio
import random
from omegaconf import OmegaConf
dtype = torch.cuda.FloatTensor

from model import INR

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
    gt = torch.from_numpy(gt_np).type(dtype)

    [n_1,n_2,n_3] = gt.shape

    x_in = torch.arange(1,n_1+1)
    y_in = torch.arange(1,n_2+1)
    z_in = torch.arange(1,n_3+1)
    x_in,y_in,z_in = torch.meshgrid(
        x_in, y_in, z_in)
    x_in = torch.flatten(x_in).unsqueeze(1)
    y_in = torch.flatten(y_in).unsqueeze(1)
    z_in = torch.flatten(z_in).unsqueeze(1)
    in_crood = torch.cat((x_in,y_in,z_in),dim=1).type(dtype)

    model = INR(method).type(dtype)
    if config.meta_learn:
        checkpoint_path = MODEL_ROOT / config.meta_path
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)

    params = []
    params += [x for x in model.parameters()]

    s = sum([np.prod(list(p.size())) for p in params]); 
    print('Number of params: %d' % s)

    optimizier = optim.Adam(params, lr=lr_real, weight_decay=0) 

    ps_best = 0

    save_path = IMAGE_ROOT / config.save_path
    save_path.mkdir(parents=True, exist_ok=True)
    for iter in range(max_iter):
        print('\r', iter, ps_best, end='\r\r')
        X_Out_real = model(in_crood).reshape(gt.shape)
        loss = torch.norm(X_Out_real-gt,2)
        optimizier.zero_grad()
        loss.backward()
        optimizier.step()

        if iter % config.image_save_steps == 0:
            ps_here = peak_signal_noise_ratio(gt_np, 
                                                np.clip(X_Out_real.cpu(
                                                    ).clone().detach().numpy(),0,1))
            if ps_here > ps_best:
                ps_best = ps_here
            
            arr = X_Out_real.cpu().clone().detach().numpy()
            arr = np.clip(arr, 0, 1)
            img = Image.fromarray((arr * 255).astype(np.uint8))
            meta_str = "_meta" if config.meta_learn else ""
            img.save(save_path / f"{method}{iter}{meta_str}.png")

    model_save_path = MODEL_ROOT / config.save_path / "model.pth"
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "method": method}, MODEL_ROOT / config.save_path / "model.pth")

if __name__ == "__main__":
    main()
