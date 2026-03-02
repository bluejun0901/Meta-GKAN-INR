import torch
import torch.nn.functional as F


def face_areas(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    edge_1 = verts[faces[:, 0]] - verts[faces[:, 1]]
    edge_2 = verts[faces[:, 0]] - verts[faces[:, 2]]
    normals = torch.cross(edge_1, edge_2, dim=1)
    return torch.sqrt(torch.sum(normals**2, dim=1).clamp_min(1e-12)) * 0.5


def sample_points_on_mesh(
    verts: torch.Tensor,
    faces: torch.Tensor,
    num_points: int,
    prior: torch.Tensor | None = None,
) -> torch.Tensor:
    if prior is None:
        probs = F.normalize(face_areas(verts, faces), dim=0, p=1)
    else:
        probs = F.normalize(prior, dim=0, p=1)

    sampled_face_indices = torch.multinomial(probs, int(num_points), replacement=True)
    sampled_faces = faces[sampled_face_indices]
    alpha = torch.rand(int(num_points), 1, device=verts.device, dtype=verts.dtype)
    beta = torch.rand(int(num_points), 1, device=verts.device, dtype=verts.dtype)
    k = beta.sqrt()
    bary_a = 1 - k
    bary_b = (1 - alpha) * k
    bary_c = alpha * k
    return (
        verts[sampled_faces[:, 0]] * bary_a + verts[sampled_faces[:, 1]] * bary_b + verts[sampled_faces[:, 2]] * bary_c
    )


def _chunked_one_way_distance(
    src_points: torch.Tensor,
    dst_points: torch.Tensor,
    scale: float,
    chunk_size: int = 2048,
) -> torch.Tensor:
    src_scaled = src_points * scale
    dst_scaled = dst_points * scale

    min_dists: list[torch.Tensor] = []
    for start in range(0, src_scaled.shape[0], chunk_size):
        src_chunk = src_scaled[start : start + chunk_size]
        dists = torch.cdist(src_chunk, dst_scaled)
        min_dists.append(dists.min(dim=1).values)
    return torch.cat(min_dists, dim=0).mean() / scale


def point_to_mesh_error(
    points: torch.Tensor,
    mesh_verts: torch.Tensor,
    mesh_faces: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    if mesh_verts.device != points.device:
        mesh_verts = mesh_verts.to(points.device)
    if mesh_faces.device != points.device:
        mesh_faces = mesh_faces.to(points.device)

    # Approximate point-to-surface distance by comparing with points sampled from target mesh.
    target_samples = sample_points_on_mesh(mesh_verts, mesh_faces, num_points=points.shape[0])
    return _chunked_one_way_distance(points, target_samples, scale=scale)


def symmetric_point_mesh_error(
    pred_verts: torch.Tensor,
    pred_faces: torch.Tensor,
    target_verts: torch.Tensor,
    target_faces: torch.Tensor,
    num_points: int,
    scale: float = 1e4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred_samples = sample_points_on_mesh(pred_verts, pred_faces, num_points=num_points)
    target_samples = sample_points_on_mesh(target_verts, target_faces, num_points=num_points)

    recon_to_target = _chunked_one_way_distance(pred_samples, target_samples, scale=scale)
    target_to_recon = _chunked_one_way_distance(target_samples, pred_samples, scale=scale)
    return recon_to_target + target_to_recon, recon_to_target, target_to_recon
