import torch
import torch.nn as nn
import numpy as np

import neural_renderer as nr
import config
import pickle
import torch.nn.functional as F

class NMRRenderer(nn.Module):
    """
    Neural mesh renderer module - renders 6 body-part segmentations or RGB images.
    Code adapted from https://github.com/nkolot/SPIN/blob/master/utils/part_utils.py
    6 body-part convention:
    0 - background
    1 - left arm
    2 - right arm
    3 - head
    4 - left leg
    5 - right leg
    6 - torso
    """
    def __init__(self,
                 batch_size,
                 cam_K,
                 cam_R,
                 img_wh=256,
                 rend_parts_seg=False,
                 device='cuda'):
        """
        :param batch_size
        :param cam_K: (bs, 3, 3) camera intrinsics matrix
        :param cam_R: (bs, 3, 3) camera rotation matrix (usually identity).
        :param img_wh: output render width/height
        :param rend_parts_seg: if True, render 6 part segmentation, else render RGB.
        """
        super(NMRRenderer, self).__init__()

        self.device = device
        with open(config.ra_body_path, 'rb') as f:
            this_dict = pickle.load(f)
        self.faces_uv = this_dict['faces_uv'].to(device)   # 13776x3

        verts_uv = this_dict['verts_uv']   # 7576x2
        verts_uv[:, 1] = 1-verts_uv[:, 1]

        self.verts_uv_t = verts_uv[None, None].to(device) * 2 - 1
        self.st = 8   # resolution of texture image

        faces = np.load(config.SMPL_FACES_PATH)
        faces = torch.from_numpy(faces.astype(np.int32))
        faces = faces[None, :].expand(batch_size, -1, -1)
        self.register_buffer('faces', faces)

        if rend_parts_seg:
            textures = np.load(config.VERTEX_TEXTURE_PATH)
            textures = torch.from_numpy(textures).float()
            textures = textures.expand(batch_size, -1, -1, -1, -1, -1)
            self.register_buffer('textures', textures)

            cube_parts = np.load(config.CUBE_PARTS_PATH)
            cube_parts = torch.from_numpy(cube_parts).float()
            self.register_buffer('cube_parts', cube_parts)
        else:
            texture_size = 2
            textures = torch.ones(batch_size, self.faces.shape[1], texture_size, texture_size,
                                  texture_size, 3, dtype=torch.float32)
            self.register_buffer('textures', textures)

        # Setup renderer
        if cam_K.ndim != 3:
            print("Expanding cam_K and cam_R by batch size.")
            cam_K = cam_K[None, :, :].expand(batch_size, -1, -1)
            cam_R = cam_R[None, :, :].expand(batch_size, -1, -1)
        renderer = nr.Renderer(camera_mode='projection',
                               K=cam_K,
                               R=cam_R,
                               image_size=img_wh,
                               orig_size=img_wh,
                               light_direction=[0, 0, 1])
        if rend_parts_seg:
            renderer.light_intensity_ambient = 1
            renderer.anti_aliasing = False
            renderer.light_intensity_directional = 0
        self.renderer = renderer

        self.rend_parts_seg = rend_parts_seg

    def forward(self, vertices, cam_ts):
        """
        :param vertices: (B, N, 3)
        :param cam_ts: (B, 1, 3)
        """
        if cam_ts.ndim == 2:
            cam_ts = cam_ts.unsqueeze(1)
        if self.rend_parts_seg:
            parts, _, mask = self.renderer(vertices, self.faces, self.textures,
                                           t=cam_ts)
            parts = self.get_parts(parts, mask)
            return parts
        else:
            rend_image, depth, _ = self.renderer(vertices, self.faces, self.textures,
                                                 t=cam_ts)
            return rend_image, depth

    def triangle_to_cube(self, tex_tensor):
        # input: Bx3x13776x3
        if self.st == 1:
            batch_size = tex_tensor.shape[0]
            tex_tensor = tex_tensor.mean(dim=-1)  # Bx3x13776
            tex_tensor = tex_tensor.permute(0, 2, 1).view(batch_size, -1, 1, 1, 1, 3)  # Bx13776x1x1x1x3
        else:
            tmp = torch.linspace(0, 1, self.st).to(tex_tensor.device)
            x, y, z = torch.meshgrid(tmp, tmp, tmp)
            x = x[None, None, None]
            y = y[None, None, None]
            z = z[None, None, None]
            # B x 3 x 13776 x st x st x st
            tex_tensor = x * tex_tensor[:, :, :, 0:1, None, None] + y * tex_tensor[:, :, :, 1:2, None, None] + z * tex_tensor[:, :, :, 2:, None, None]
            # B x 13776 x st x st x st x 3
            tex_tensor = tex_tensor.permute(0, 2, 3, 4, 5, 1)
        return tex_tensor

    # from Texformmer
    def get_tex_tensor(self, uv_map_t):
        batch_size = uv_map_t.shape[0]
        uv_map_t = torch.from_numpy(uv_map_t).float().to(self.device) / 255
        verts_uv_t = self.verts_uv_t.expand(batch_size, -1, -1, -1)
        sampled_uv = F.grid_sample(uv_map_t, verts_uv_t)   # Bx3x1x7576
        #print('sampled_uv ' + str(sampled_uv.shape))

        # generate texture tensor
        tex_tensor = sampled_uv.squeeze(2)[:, :, self.faces_uv.flatten()]   # Bx3x(13776*3)
        tex_tensor = tex_tensor.view(batch_size, 3, -1, 3)   # Bx3x13776x3
        tex_tensor = self.triangle_to_cube(tex_tensor)

        return tex_tensor

    def render_texture(self, vertices, cam_ts, uv_image=None):
        if cam_ts.ndim == 2:
            cam_ts = cam_ts.unsqueeze(1)

        if (uv_image is None):
            tex = self.textures
        else:
            #tex = uv_image
            tex = self.get_tex_tensor(uv_image)
        #print('tex ' + str(tex.shape))
        #print('self.texture ' + str(self.textures.shape))
        rend_image, depth, _ = self.renderer(vertices, self.faces, tex,
                                                 t=cam_ts)
        return rend_image, depth

    def get_parts(self, parts, mask):
        """Process renderer part image to get body part indices."""
        bn,c,h,w = parts.shape
        mask = mask.view(-1,1)
        parts_index = torch.floor(100*parts.permute(0,2,3,1).contiguous().view(-1,3)).long()
        parts = self.cube_parts[parts_index[:,0], parts_index[:,1], parts_index[:,2], None]
        parts *= mask
        parts = parts.view(bn,h,w).long()
        return parts