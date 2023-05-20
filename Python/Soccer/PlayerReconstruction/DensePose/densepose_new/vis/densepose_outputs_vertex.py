# Copyright (c) Facebook, Inc.  and its affiliates.  All Rights Reserved
import json
import numpy as np
from functools import lru_cache
from typing import Dict, List, Optional, Tuple
import cv2
import torch

from detectron2.utils.file_io import PathManager
from scipy import interpolate

from ..modeling import build_densepose_embedder
from ..modeling.cse.utils import get_closest_vertices_mask_from_ES, get_all_vertices_mask_from_ES

from ..data.utils import get_class_to_mesh_name_mapping
from ..structures import DensePoseEmbeddingPredictorOutput
from ..structures.mesh import create_mesh
from .base import Boxes, Image, MatrixVisualizer
from .densepose_results_textures import get_texture_atlas


@lru_cache()
def get_xyz_vertex_embedding(mesh_name: str, device: torch.device):
    if mesh_name == "smpl_27554":
        embed_path = PathManager.get_local_path("Data/Densepose/mds_d=256.npy")
        embed_map, _ = np.load(embed_path, allow_pickle=True)
        embed_map = torch.tensor(embed_map).float()[:, 0]
        embed_map -= embed_map.min()
        embed_map /= embed_map.max()
    else:
        mesh = create_mesh(mesh_name, device)
        embed_map = mesh.vertices.sum(dim=1)
        embed_map -= embed_map.min()
        embed_map /= embed_map.max()
        embed_map = embed_map ** 2
    return embed_map


class DensePoseOutputsVertexVisualizer(object):
    def __init__(self,
        cfg,
        inplace=True,
        cmap=cv2.COLORMAP_JET,
        alpha=0.7,
        device="cuda",
        default_class=0,
        **kwargs,):
        self.mask_visualizer = MatrixVisualizer(inplace=inplace, cmap=cmap, val_scale=1.0, alpha=alpha)
        self.class_to_mesh_name = get_class_to_mesh_name_mapping(cfg)
        self.embedder = build_densepose_embedder(cfg)
        self.device = torch.device(device)
        self.default_class = default_class

        self.mesh_vertex_embeddings = {
            mesh_name: self.embedder(mesh_name).to(self.device)
            for mesh_name in self.class_to_mesh_name.values()
            if self.embedder.has_embeddings(mesh_name)
        }

    def visualize(self,
        image_bgr: Image,
        outputs_boxes_xywh_classes: Tuple[Optional[DensePoseEmbeddingPredictorOutput], Optional[Boxes], Optional[List[int]]],) -> Image:
        if outputs_boxes_xywh_classes[0] is None:
            return image_bgr

        S, E, N, bboxes_xywh, pred_classes = self.extract_and_check_outputs_and_boxes(outputs_boxes_xywh_classes)

        for n in range(N):
            x, y, w, h = bboxes_xywh[n].int().tolist()
            mesh_name = self.class_to_mesh_name[pred_classes[n]]
            closest_vertices, mask = get_closest_vertices_mask_from_ES(E[[n]],
                S[[n]],
                h,
                w,
                self.mesh_vertex_embeddings[mesh_name],
                self.device,)
            embed_map = get_xyz_vertex_embedding(mesh_name, self.device)
            vis = (embed_map[closest_vertices].clip(0, 1) * 255.0).cpu().numpy()
            mask_numpy = mask.cpu().numpy().astype(dtype=np.uint8)
            image_bgr = self.mask_visualizer.visualize(image_bgr, mask_numpy, vis, [x, y, w, h])

        return image_bgr

    def extract_and_check_outputs_and_boxes(self, outputs_boxes_xywh_classes):

        densepose_output, bboxes_xywh, pred_classes = outputs_boxes_xywh_classes

        if pred_classes is None:
            pred_classes = [self.default_class] * len(bboxes_xywh)

        assert isinstance(densepose_output, DensePoseEmbeddingPredictorOutput), "DensePoseEmbeddingPredictorOutput expected, {} encountered".format(type(densepose_output))

        S = densepose_output.coarse_segm
        E = densepose_output.embedding
        N = S.size(0)
        assert N == E.size(0), "CSE coarse_segm {} and embeddings {}" " should have equal first dim size".format(S.size(), E.size())
        assert N == len(bboxes_xywh), "number of bounding boxes {}" " should be equal to first dim size of outputs {}".format(len(bboxes_xywh), N)
        assert N == len(pred_classes), ("number of predicted classes {}"
            " should be equal to first dim size of outputs {}".format(len(bboxes_xywh), N))

        return S, E, N, bboxes_xywh, pred_classes


def get_texture_atlases(json_str: Optional[str]) -> Optional[Dict[str, Optional[np.ndarray]]]:
    """
    json_str is a JSON string representing a mesh_name -> texture_atlas_path dictionary
    """
    if json_str is None:
        return None

    paths = json.loads(json_str)
    return {mesh_name: get_texture_atlas(path) for mesh_name, path in paths.items()}


class DensePoseOutputsTextureVisualizer(DensePoseOutputsVertexVisualizer):
    def __init__(self,
        cfg,
        texture_atlases_dict,
        device="cuda",
        default_class=0,
        **kwargs,):
        self.embedder = build_densepose_embedder(cfg)

        self.texture_image_dict = {}
        self.alpha_dict = {}

        for mesh_name in texture_atlases_dict.keys():
            if texture_atlases_dict[mesh_name].shape[-1] == 4:  # Image with alpha channel
                self.alpha_dict[mesh_name] = texture_atlases_dict[mesh_name][:, :, -1] / 255.0
                self.texture_image_dict[mesh_name] = texture_atlases_dict[mesh_name][:, :, :3]
            else:
                self.alpha_dict[mesh_name] = texture_atlases_dict[mesh_name].sum(axis=-1) > 0
                self.texture_image_dict[mesh_name] = texture_atlases_dict[mesh_name]

        self.device = torch.device(device)
        self.class_to_mesh_name = get_class_to_mesh_name_mapping(cfg)
        self.default_class = default_class

        self.mesh_vertex_embeddings = {
            mesh_name: self.embedder(mesh_name).to(self.device)
            for mesh_name in self.class_to_mesh_name.values()
        }

    def visualize(self,
        image_bgr: Image,
        outputs_boxes_xywh_classes: Tuple[Optional[DensePoseEmbeddingPredictorOutput], Optional[Boxes], Optional[List[int]]],) -> Image:
        image_target_bgr = image_bgr.copy()
        if outputs_boxes_xywh_classes[0] is None:
            return image_target_bgr

        S, E, N, bboxes_xywh, pred_classes = self.extract_and_check_outputs_and_boxes(outputs_boxes_xywh_classes)

        meshes = {
            p: create_mesh(self.class_to_mesh_name[p], self.device) for p in np.unique(pred_classes)
        }

        #print("N: {}".format(N))
        for n in range(N):
            x, y, w, h = bboxes_xywh[n].int().cpu().numpy()
            mesh_name = self.class_to_mesh_name[pred_classes[n]]
            closest_vertices, mask = get_closest_vertices_mask_from_ES(E[[n]],
                S[[n]],
                h,
                w,
                self.mesh_vertex_embeddings[mesh_name],
                self.device,)
            uv_array = meshes[pred_classes[n]].texcoords[closest_vertices].permute((2, 0, 1))
            uv_array = uv_array.cpu().numpy().clip(0, 1)
            textured_image = self.generate_image_with_texture(image_target_bgr[y : y + h, x : x + w],
                uv_array,
                mask.cpu().numpy(),
                self.class_to_mesh_name[pred_classes[n]],)
            if textured_image is None:
                continue
            image_target_bgr[y : y + h, x : x + w] = textured_image

        return image_target_bgr

    def generate_image_with_texture(self, bbox_image_bgr, uv_array, mask, mesh_name):
        alpha = self.alpha_dict.get(mesh_name)
        texture_image = self.texture_image_dict.get(mesh_name)
        if alpha is None or texture_image is None:
            return None
        texture_image = np.ones_like(texture_image) * 255
        U, V = uv_array
        x_index = (U * texture_image.shape[1]).astype(int)
        y_index = (V * texture_image.shape[0]).astype(int)
        local_texture = texture_image[y_index, x_index][mask]
        local_alpha = np.expand_dims(alpha[y_index, x_index][mask], -1)
        output_image = bbox_image_bgr.copy()
        #output_image[mask] = output_image[mask] * (1 - local_alpha) +
        #local_texture * local_alpha
        output_image[mask] = local_texture
        return output_image.astype(np.uint8)

class DensePoseOutputsTextureGenerizer(DensePoseOutputsVertexVisualizer):
    def __init__(self,
        cfg,
        texture_atlases_dict,
        device="cuda",
        default_class=0,
        **kwargs,):
        self.embedder = build_densepose_embedder(cfg)

        self.texture_image_dict = {}
        self.alpha_dict = {}

        for mesh_name in texture_atlases_dict.keys():
            if texture_atlases_dict[mesh_name].shape[-1] == 4:  # Image with alpha channel
                self.alpha_dict[mesh_name] = texture_atlases_dict[mesh_name][:, :, -1] / 255.0
                self.texture_image_dict[mesh_name] = texture_atlases_dict[mesh_name][:, :, :3]
            else:
                self.alpha_dict[mesh_name] = texture_atlases_dict[mesh_name].sum(axis=-1) > 0
                self.texture_image_dict[mesh_name] = texture_atlases_dict[mesh_name]

        self.device = torch.device(device)
        self.class_to_mesh_name = get_class_to_mesh_name_mapping(cfg)
        self.default_class = default_class

        self.mesh_vertex_embeddings = {
            mesh_name: self.embedder(mesh_name).to(self.device)
            for mesh_name in self.class_to_mesh_name.values()
        }

    def visualize(self,
        image_bgr: Image,
        outputs_boxes_xywh_classes: Tuple[Optional[DensePoseEmbeddingPredictorOutput], Optional[Boxes], Optional[List[int]]],) -> Image:
        image_target_bgr = image_bgr.copy()
        if outputs_boxes_xywh_classes[0] is None:
            return image_target_bgr

        S, E, N, bboxes_xywh, pred_classes = self.extract_and_check_outputs_and_boxes(outputs_boxes_xywh_classes)

        meshes = {
            p: create_mesh(self.class_to_mesh_name[p], self.device) for p in np.unique(pred_classes)
        }

        #print("N: {}".format(N))
        n = 0
        x, y, w, h = bboxes_xywh[n].int().cpu().numpy()
        mesh_name = self.class_to_mesh_name[pred_classes[n]]
        #closest_vertices, mask = get_all_vertices_mask_from_ES(E[[n]],
        #    S[[n]],
        #    h,
        #    w,
        #    self.mesh_vertex_embeddings[mesh_name],
        #    self.device,)

        #coords = meshes[pred_classes[n]].texcoords
        #vertex_x = []
        #vertex_y = []
        #vertex_u = []
        #vertex_v = []
        #u = np.zeros((h, w))
        #v = np.zeros((h, w))
        #for i in range(h):
        #    for j in range(w):
        #        if mask[i][j]:
        #            vertex_index = closest_vertices[i][j].item()
        #            if vertex_index != -1:
        #                vertex_x.append(j)
        #                vertex_y.append(i)
        #                #print(coords[closest_vertices[i][j]])
        #                vertex_u.append(coords[vertex_index][0])
        #                vertex_v.append(coords[vertex_index][1])
        #                u[i][j] = coords[vertex_index][0]
        #                v[i][j] = coords[vertex_index][1]
        ##print(len(vertex_x))
        #cv2.imwrite('Data/u_b.png', (u * 255).astype(np.uint8))
        #cv2.imwrite('Data/v_b.png', (v * 255).astype(np.uint8))

        #f_u = interpolate.interp2d(vertex_x, vertex_y, vertex_u)
        #f_v = interpolate.interp2d(vertex_x, vertex_y, vertex_v)
        #u = f_u(np.arange(w), np.arange(h))
        #v = f_v(np.arange(w), np.arange(h))

        #cv2.imwrite('Data/u.png', (u * 255).astype(np.uint8))
        #cv2.imwrite('Data/v.png', (v * 255).astype(np.uint8))

        #u = u[mask].clip(0, 1)
        #v = v[mask].clip(0, 1)

        closest_vertices, mask = get_closest_vertices_mask_from_ES(E[[n]],
                S[[n]],
                h,
                w,
                self.mesh_vertex_embeddings[mesh_name],
                self.device,)
        uv_array = meshes[pred_classes[n]].texcoords[closest_vertices].permute((2, 0, 1))
        uv_array = uv_array.cpu().numpy().clip(0, 1)

        u, v = uv_array
        #resize_factor = 2
        #cv2.imwrite('Data/u_before.png', (u * 255).astype(np.uint8))
        #cv2.imwrite('Data/v_before.png', (v * 255).astype(np.uint8))
        #u = cv2.resize(u, (w//resize_factor, h//resize_factor), cv2.INTER_NEAREST)
        #v = cv2.resize(v, (w//resize_factor, h//resize_factor), cv2.INTER_NEAREST)
        #u = cv2.resize(u, (w, h), cv2.INTER_CUBIC)
        #v = cv2.resize(v, (w, h), cv2.INTER_CUBIC)
        #cv2.imwrite('Data/u_after.png', (u * 255).astype(np.uint8))
        #cv2.imwrite('Data/v_after.png', (v * 255).astype(np.uint8))
        mask = mask.cpu().numpy()
        u = u[mask]
        v = v[mask]

        textured_image, texture_mask = self.generate_image_with_texture(image_target_bgr[y : y + h, x : x + w],
            u, v,
            mask,
            self.class_to_mesh_name[pred_classes[n]],)
        return textured_image, texture_mask

    def generate_image_with_texture(self, bbox_image_bgr, U, V, mask, mesh_name):
        alpha = self.alpha_dict.get(mesh_name)
        texture_image = self.texture_image_dict.get(mesh_name)
        if alpha is None or texture_image is None:
            return None

        texture_image = np.zeros((1024, 1024, 3))
        texture_mask =  np.zeros((1024, 1024))
        x_index = (U * (texture_image.shape[1]-1)).astype(int)
        y_index = (V * (texture_image.shape[0]-1)).astype(int)
        texture_image[y_index, x_index] = bbox_image_bgr[mask]
        texture_mask[y_index, x_index] = 1
        return texture_image.astype(np.uint8), texture_mask
