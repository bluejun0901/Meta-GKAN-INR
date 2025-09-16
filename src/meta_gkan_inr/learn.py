import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
from torch import optim 
import numpy as np 
import matplotlib.pyplot as plt 
from skimage.metrics import peak_signal_noise_ratio
from skimage import io as skio
import random
dtype = torch.cuda.FloatTensor

from model import INR

def main():
    for method in ['GKAN']:
        seed=1
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        torch.cuda.empty_cache()

        image_path = "../../dataset/STI/Classic/goldhill.bmp"
        max_iter = 100
        lr_real = 0.001

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
        checkpoint_path = f'../../checkpoints/maml_gkan.pth'
        model.load_state_dict(torch.load(checkpoint_path))

        params = []
        params += [x for x in model.parameters()]

        s = sum([np.prod(list(p.size())) for p in params]); 
        print('Number of params: %d' % s)

        optimizier = optim.Adam(params, lr=lr_real, weight_decay=0) 

        ps_best = 0

        for iter in range(max_iter):
            print('\r', iter, ps_best, end='\r\r')
            X_Out_real = model(in_crood).reshape(gt.shape)
            loss = torch.norm(X_Out_real-gt,2)
            optimizier.zero_grad()
            loss.backward()
            optimizier.step()

            if iter % 10 == 0 and iter >= 10:
                ps_here = peak_signal_noise_ratio(gt_np, 
                                                  np.clip(X_Out_real.cpu(
                                                      ).clone().detach().numpy(),0,1))
                if ps_here > ps_best:
                    ps_best = ps_here
                    
                plt.imshow(X_Out_real.cpu().clone().detach().numpy())
                plt.savefig(f"{method}{iter}_meta.png")
                print(ps_here)  

if __name__ == "__main__":
    main()
