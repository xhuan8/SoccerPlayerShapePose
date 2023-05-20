# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
from torch.nn import functional as F


def squared_euclidean_distance_matrix(pts1: torch.Tensor, pts2: torch.Tensor) -> torch.Tensor:
    """
    Get squared Euclidean Distance Matrix
    Computes pairwise squared Euclidean distances between points

    Args:
        pts1: Tensor [M x D], M is the number of points, D is feature dimensionality
        pts2: Tensor [N x D], N is the number of points, D is feature dimensionality

    Return:
        Tensor [M, N]: matrix of squared Euclidean distances; at index (m, n)
            it contains || pts1[m] - pts2[n] ||^2
    """
    edm = torch.mm(-2 * pts1, pts2.t())
    edm += (pts1 * pts1).sum(1, keepdim=True) + (pts2 * pts2).sum(1, keepdim=True).t()
    return edm.contiguous()


def normalize_embeddings(embeddings: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """
    Normalize N D-dimensional embedding vectors arranged in a tensor [N, D]

    Args:
        embeddings (tensor [N, D]): N D-dimensional embedding vectors
        epsilon (float): minimum value for a vector norm
    Return:
        Normalized embeddings (tensor [N, D]), such that L2 vector norms are all equal to 1.
    """
    return embeddings / torch.clamp(
        embeddings.norm(p=None, dim=1, keepdim=True), min=epsilon  # pyre-ignore[6]
    )


def get_closest_vertices_mask_from_ES(
    E: torch.Tensor,
    S: torch.Tensor,
    h: int,
    w: int,
    mesh_vertex_embeddings: torch.Tensor,
    device: torch.device,
):
    """
    Interpolate Embeddings and Segmentations to the size of a given bounding box,
    and compute closest vertices and the segmentation mask

    Args:
        E (tensor [1, D, H, W]): D-dimensional embedding vectors for every point of the
            default-sized box
        S (tensor [1, 2, H, W]): 2-dimensional segmentation mask for every point of the
            default-sized box
        h (int): height of the target bounding box
        w (int): width of the target bounding box
        mesh_vertex_embeddings (tensor [N, D]): vertex embeddings for a chosen mesh
            N is the number of vertices in the mesh, D is feature dimensionality
        device (torch.device): device to move the tensors to
    Return:
        Closest Vertices (tensor [h, w]), int, for every point of the resulting box
        Segmentation mask (tensor [h, w]), boolean, for every point of the resulting box
    """
    embedding_resized = F.interpolate(E, size=(h, w), mode="bilinear")[0].to(device)
    coarse_segm_resized = F.interpolate(S, size=(h, w), mode="bilinear")[0].to(device)
    mask = coarse_segm_resized.argmax(0) > 0
    closest_vertices = torch.zeros(mask.shape, dtype=torch.long, device=device)
    all_embeddings = embedding_resized[:, mask].t()
    size_chunk = 10_000  # Chunking to avoid possible OOM
    edm = []
    if len(all_embeddings) == 0:
        return closest_vertices, mask
    for chunk in range((len(all_embeddings) - 1) // size_chunk + 1):
        chunk_embeddings = all_embeddings[size_chunk * chunk : size_chunk * (chunk + 1)]
        edm.append(
            torch.argmin(
                squared_euclidean_distance_matrix(chunk_embeddings, mesh_vertex_embeddings), dim=1
            )
        )
    closest_vertices[mask] = torch.cat(edm)
    return closest_vertices, mask

def get_all_vertices_mask_from_ES(
    E: torch.Tensor,
    S: torch.Tensor,
    h: int,
    w: int,
    mesh_vertex_embeddings: torch.Tensor,
    device: torch.device,
):
    """
    Interpolate Embeddings and Segmentations to the size of a given bounding box,
    and compute closest vertices and the segmentation mask

    Args:
        E (tensor [1, D, H, W]): D-dimensional embedding vectors for every point of the
            default-sized box
        S (tensor [1, 2, H, W]): 2-dimensional segmentation mask for every point of the
            default-sized box
        h (int): height of the target bounding box
        w (int): width of the target bounding box
        mesh_vertex_embeddings (tensor [N, D]): vertex embeddings for a chosen mesh
            N is the number of vertices in the mesh, D is feature dimensionality
        device (torch.device): device to move the tensors to
    Return:
        Closest Vertices (tensor [h, w]), int, for every point of the resulting box
        Segmentation mask (tensor [h, w]), boolean, for every point of the resulting box
    """
    embedding_resized = F.interpolate(E, size=(h, w), mode="bilinear")[0].to(device)
    coarse_segm_resized = F.interpolate(S, size=(h, w), mode="bilinear")[0].to(device)
    mask = coarse_segm_resized.argmax(0) > 0
    closest_vertices = torch.zeros(mask.shape, dtype=torch.long, device=device)
    all_embeddings = embedding_resized[:, mask].t()
    size_chunk = 10_000  # Chunking to avoid possible OOM
    edm = []
    if len(all_embeddings) == 0:
        return closest_vertices, mask
    ans = []
    pixel_index = []
    pixel_val = []
    vertex_index = None
    vertex_val = None
    #print(len(all_embeddings))
    for chunk in range((len(all_embeddings) - 1) // size_chunk + 1):
        chunk_embeddings = all_embeddings[size_chunk * chunk : size_chunk * (chunk + 1)]
        edm_chunk = squared_euclidean_distance_matrix(chunk_embeddings, mesh_vertex_embeddings)
        edm_vertex = squared_euclidean_distance_matrix(mesh_vertex_embeddings, chunk_embeddings)
        edm_vertex_min = torch.min(edm_vertex, dim = 1)
        edm_chunk_min = torch.min(edm_chunk, dim = 1)
        #print('edm_chunk ' + str(edm_chunk.shape))
        if vertex_val is None:
            #print('edm_min ' + str(edm_min[0].shape))
            vertex_val = edm_vertex_min[0]
            vertex_index = edm_vertex_min[1]
        else:
            #print('edm_min ' + str(edm_min[0].shape))
            vertex_val_temp = edm_vertex_min[0]
            vertex_index_temp = edm_vertex_min[1]
            vertex_mask = torch.gt(vertex_val, vertex_val_temp)
            vertex_val[vertex_mask] = vertex_val_temp[vertex_mask]
            vertex_index[vertex_mask] = torch.add(vertex_index_temp[vertex_mask], size_chunk * chunk)
        pixel_index.append(edm_chunk_min[1])
        pixel_val.append(edm_chunk_min[0])
    
    pixel_index = torch.cat(pixel_index)
    #print(pixel_index)
    pixel_val = torch.cat(pixel_val)
    vertex_min = [[float('inf'), -1] for i in range(len(mesh_vertex_embeddings))] 
    
    for i in range(len(all_embeddings)):
    #for i in range(10):
        v_index = pixel_index[i].item()
        #print('v_index ' + str(v_index))
        #print('vertex min ' + str(vertex_min[v_index]))
        if (vertex_min[v_index][1] == -1):
            vertex_min[v_index][0] = pixel_val[i]
            vertex_min[v_index][1] = i
            ans.append(torch.tensor(v_index))
        else:
            if (vertex_min[v_index][0] > pixel_val[i]):
                ans[vertex_min[v_index][1]] = torch.tensor(-1)
                vertex_min[v_index][0] = pixel_val[i]
                vertex_min[v_index][1] = i
                ans.append(torch.tensor(v_index))
            else:
                ans.append(torch.tensor(-1))
        #print('vertex min after ' + str(vertex_min[v_index]))
        #print('vertex min after ' + str(vertex_min[1]))
        #print(ans)
    #print(vertex_min)
    #print(closest_vertices[mask].shape)
    #print(torch.stack(ans).shape)
    closest_vertices[mask] = torch.stack(ans)
    #print(closest_vertices[0])
    #print(closest_vertices[mask][0])
    #print(closest_vertices[mask][1])
    return closest_vertices, mask
