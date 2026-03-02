import logging
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from src.learn.learn import BaseLearner
from src.logging.train_logger import TrainLogger
from src.mesh import load_mesh_data, symmetric_point_mesh_error


def _create_icosphere(level: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    t = (1.0 + math.sqrt(5.0)) / 2.0
    vertices = [
        [-1, t, 0],
        [1, t, 0],
        [-1, -t, 0],
        [1, -t, 0],
        [0, -1, t],
        [0, 1, t],
        [0, -1, -t],
        [0, 1, -t],
        [t, 0, -1],
        [t, 0, 1],
        [-t, 0, -1],
        [-t, 0, 1],
    ]
    faces = [
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ]

    def _normalized(v: list[float]) -> list[float]:
        n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
        return [v[0] / n, v[1] / n, v[2] / n]

    vertices = [_normalized(v) for v in vertices]

    for _ in range(level):
        midpoint_cache: dict[tuple[int, int], int] = {}

        def midpoint_idx(i: int, j: int) -> int:
            key = (min(i, j), max(i, j))
            if key in midpoint_cache:
                return midpoint_cache[key]

            vi = vertices[i]
            vj = vertices[j]
            mid = _normalized([(vi[0] + vj[0]) * 0.5, (vi[1] + vj[1]) * 0.5, (vi[2] + vj[2]) * 0.5])
            vertices.append(mid)
            idx = len(vertices) - 1
            midpoint_cache[key] = idx
            return idx

        new_faces: list[list[int]] = []
        for tri in faces:
            v1, v2, v3 = tri
            a = midpoint_idx(v1, v2)
            b = midpoint_idx(v2, v3)
            c = midpoint_idx(v3, v1)
            new_faces.extend([[v1, a, c], [v2, b, a], [v3, c, b], [a, b, c]])
        faces = new_faces

    verts_t = torch.tensor(vertices, dtype=torch.float32, device=device)
    faces_t = torch.tensor(faces, dtype=torch.int64, device=device)
    return verts_t, faces_t


def _save_obj(path: Path, verts: torch.Tensor, faces: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for v in verts.detach().cpu():
            f.write(f"v {float(v[0]):.8f} {float(v[1]):.8f} {float(v[2]):.8f}\n")
        for tri in faces.detach().cpu():
            f.write(f"f {int(tri[0]) + 1} {int(tri[1]) + 1} {int(tri[2]) + 1}\n")


class MeshLearner(BaseLearner):
    def __init__(
        self,
        mesh_path: str,
        run_dir: str | Path,
        model_x: nn.Module,
        model_y: nn.Module,
        model_z: nn.Module,
        model_name: str = "mesh_inr",
        steps: int = 2000,
        learning_rate: float = 1e-3,
        num_surface_samples: int = 15000,
        eval_samples: int = 25000,
        icosphere_level: int = 4,
        normalize_mesh: bool = True,
        log_steps: int = 10,
        mesh_save_steps: int = 100,
        distance_scale: float = 1e4,
    ):
        self.mesh_path = mesh_path
        self.model_x = model_x
        self.model_y = model_y
        self.model_z = model_z
        self.model_name = model_name
        self.steps = steps
        self.learning_rate = learning_rate
        self.num_surface_samples = num_surface_samples
        self.eval_samples = eval_samples
        self.icosphere_level = icosphere_level
        self.normalize_mesh = normalize_mesh
        self.log_steps = log_steps
        self.mesh_save_steps = mesh_save_steps
        self.distance_scale = distance_scale

        self.run_dir = Path(run_dir)
        self.artifact_path = self.run_dir / "learn_3d" / "artifacts"
        self.model_path = self.run_dir / "learn_3d" / "models"
        self.log_path = self.run_dir / "learn_3d" / "logs"
        self.artifact_path.mkdir(parents=True, exist_ok=True)
        self.model_path.mkdir(parents=True, exist_ok=True)
        self.log_path.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(__name__)
        self.metrics_logger = TrainLogger(
            run_dir=self.run_dir,
            name="learn_3d",
            auto_draw=True,
            draw_freq=100,
            draw_kwargs={"y_axis": ["loss", "r2t", "t2r", "best_loss"], "x_axis": "step"},
        )

    def _set_seed(self, seed: int = 1) -> None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
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

    def _predict_vertices(
        self,
        base_verts: torch.Tensor,
        model_x: nn.Module,
        model_y: nn.Module,
        model_z: nn.Module,
    ) -> torch.Tensor:
        pred_x = model_x(base_verts)
        pred_y = model_y(base_verts)
        pred_z = model_z(base_verts)
        return torch.cat([pred_x, pred_y, pred_z], dim=1)

    def _save_mesh(self, path: Path, verts: torch.Tensor, faces: torch.Tensor) -> None:
        _save_obj(path, verts, faces)

    def train(self, seed: int = 1) -> None:
        self._set_seed(seed)
        device = self._device()

        target_mesh = load_mesh_data(self.mesh_path, device=device, normalize=self.normalize_mesh)
        base_verts, base_faces = _create_icosphere(self.icosphere_level, device=device)

        self.logger.info(
            f"Loaded target mesh verts={target_mesh.verts.shape[0]} faces={target_mesh.faces.shape[0]} on {device}."
        )
        self._save_mesh(self.artifact_path / "target_normalized.obj", target_mesh.verts, target_mesh.faces)
        self._save_mesh(self.artifact_path / "base_icosphere.obj", base_verts, base_faces)

        model_x = self.model_x.to(device)
        model_y = self.model_y.to(device)
        model_z = self.model_z.to(device)

        parameters = list(model_x.parameters()) + list(model_y.parameters()) + list(model_z.parameters())
        optimizer = optim.Adam(parameters, lr=self.learning_rate, weight_decay=0.0)

        total_params = sum(p.numel() for p in parameters)
        self.logger.info(f"Number of trainable parameters across xyz models: {total_params}")

        best_loss = torch.tensor(float("inf"), device=device)
        for step in range(1, self.steps + 1):
            pred_verts = self._predict_vertices(base_verts, model_x, model_y, model_z)
            loss, r2t, t2r = symmetric_point_mesh_error(
                pred_verts=pred_verts,
                pred_faces=base_faces,
                target_verts=target_mesh.verts,
                target_faces=target_mesh.faces,
                num_points=self.num_surface_samples,
                scale=self.distance_scale,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if loss < best_loss:
                best_loss = loss.detach()

            if step % self.log_steps == 0 or step == 1:
                self.metrics_logger.log(
                    step=step,
                    loss=float(loss.detach().cpu()),
                    r2t=float(r2t.detach().cpu()),
                    t2r=float(t2r.detach().cpu()),
                    best_loss=float(best_loss.detach().cpu()),
                )
                self.logger.info(
                    f"[3D {step}/{self.steps}] loss={loss.item():.6f}, r2t={r2t.item():.6f}, t2r={t2r.item():.6f}"
                )

            if step % self.mesh_save_steps == 0 or step == self.steps:
                with torch.no_grad():
                    eval_pred = self._predict_vertices(base_verts, model_x, model_y, model_z)
                    eval_loss, eval_r2t, eval_t2r = symmetric_point_mesh_error(
                        pred_verts=eval_pred,
                        pred_faces=base_faces,
                        target_verts=target_mesh.verts,
                        target_faces=target_mesh.faces,
                        num_points=self.eval_samples,
                        scale=self.distance_scale,
                    )
                self.logger.info(
                    f"[3D eval {step}] loss={eval_loss.item():.6f}, r2t={eval_r2t.item():.6f}, "
                    f"t2r={eval_t2r.item():.6f}"
                )
                self._save_mesh(self.artifact_path / f"recon_{step}.obj", eval_pred, base_faces)

        torch.save(
            {
                "state_dict_x": model_x.state_dict(),
                "state_dict_y": model_y.state_dict(),
                "state_dict_z": model_z.state_dict(),
                "icosphere_level": self.icosphere_level,
            },
            self.model_path / self.model_name,
        )
