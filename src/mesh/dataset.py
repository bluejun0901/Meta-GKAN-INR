from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class MeshData:
    verts: torch.Tensor
    faces: torch.Tensor


def _parse_face_vertex_index(token: str, num_verts: int) -> int:
    head = token.split("/")[0]
    idx = int(head)
    if idx > 0:
        return idx - 1
    return num_verts + idx


def _load_obj(mesh_path: str | Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    vertices: list[list[float]] = []
    triangles: list[list[int]] = []

    with Path(mesh_path).open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("v "):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                continue

            if line.startswith("f "):
                parts = line.split()[1:]
                face_indices = [_parse_face_vertex_index(token, len(vertices)) for token in parts]
                if len(face_indices) < 3:
                    continue
                if len(face_indices) == 3:
                    triangles.append(face_indices)
                else:
                    base = face_indices[0]
                    for i in range(1, len(face_indices) - 1):
                        triangles.append([base, face_indices[i], face_indices[i + 1]])

    if not vertices:
        raise ValueError(f"OBJ has no vertices: {mesh_path}")
    if not triangles:
        raise ValueError(f"OBJ has no faces: {mesh_path}")

    verts = torch.tensor(vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(triangles, dtype=torch.int64, device=device)
    return verts, faces


def _normalize_vertices(verts: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    centered = verts - verts.mean(dim=0, keepdim=True)
    radius = torch.linalg.vector_norm(centered, dim=1).amax().clamp_min(eps)
    return centered / radius


def load_mesh_data(mesh_path: str | Path, device: torch.device, normalize: bool = True) -> MeshData:
    verts, faces = _load_obj(mesh_path, device=device)
    if normalize:
        verts = _normalize_vertices(verts)
    return MeshData(verts=verts, faces=faces)
