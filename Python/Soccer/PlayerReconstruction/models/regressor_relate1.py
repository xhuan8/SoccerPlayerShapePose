import torch.nn as nn
import torch
import numpy as np

class PoseRelationModule(nn.Module):
    """
    Combined encoder + regressor model that takes proxy representation input (e.g.
    silhouettes + 2D joints) and outputs SMPL body model parameters + weak-perspective
    camera.
    """
    def __init__(self, pose_dim = 216, feature_dim = 1024):
        """
        :param resnet_in_channels: 1 if input silhouette/segmentation, 1 + num_joints if
        input silhouette/segmentation + joints.
        :param resnet_layers: number of layers in ResNet backbone (18 or 50)
        :param ief_iters: number of IEF iterations.
        """
        super(PoseRelationModule, self).__init__()
        self.fc1 = nn.Linear(pose_dim, feature_dim, bias=True)
        self.fc2 = nn.Linear(feature_dim, pose_dim, bias=True)

        self.relation_module = RelationModule()

    def forward(self, input):
        pose_params, bboxes = input
        position_embedding = self.__PositionalEmbedding(bboxes)
        pose_params = pose_params.view(pose_params.shape[0], -1)
        pose_params = self.fc1(pose_params)

        pose_params = self.relation_module([pose_params, position_embedding])
        pose_params = self.fc2(pose_params)
        return pose_params.view(-1, 24, 3, 3)

    def __PositionalEmbedding(self, f_g, dim_g=64, wave_len=1000):
        x_min, y_min, x_max, y_max = torch.chunk(f_g, 4, dim=1)

        cx = (x_min + x_max) * 0.5
        cy = (y_min + y_max) * 0.5
        w = (x_max - x_min) + 1.
        h = (y_max - y_min) + 1.

        delta_x = cx - cx.view(1, -1)
        delta_x = torch.clamp(torch.abs(delta_x / w), min=1e-3)
        delta_x = torch.log(delta_x)

        delta_y = cy - cy.view(1, -1)
        delta_y = torch.clamp(torch.abs(delta_y / h), min=1e-3)
        delta_y = torch.log(delta_y)

        delta_w = torch.log(w / w.view(1, -1))
        delta_h = torch.log(h / h.view(1, -1))
        size = delta_h.size()

        delta_x = delta_x.view(size[0], size[1], 1)
        delta_y = delta_y.view(size[0], size[1], 1)
        delta_w = delta_w.view(size[0], size[1], 1)
        delta_h = delta_h.view(size[0], size[1], 1)

        position_mat = torch.cat((delta_x, delta_y, delta_w, delta_h), -1)

        feat_range = torch.arange(dim_g / 8).cuda()
        dim_mat = feat_range / (dim_g / 8)
        dim_mat = 1. / (torch.pow(wave_len, dim_mat))

        dim_mat = dim_mat.view(1, 1, 1, -1)
        position_mat = position_mat.view(size[0], size[1], 4, -1)
        position_mat = 100. * position_mat

        mul_mat = position_mat * dim_mat
        mul_mat = mul_mat.view(size[0], size[1], -1)
        sin_mat = torch.sin(mul_mat)
        cos_mat = torch.cos(mul_mat)
        embedding = torch.cat((sin_mat, cos_mat), -1)

        return embedding

class RelationModule(nn.Module):
    def __init__(self,n_relations = 16, appearance_feature_dim=1024,key_feature_dim = 64, geo_feature_dim = 64):
        super(RelationModule, self).__init__()
        self.Nr = n_relations
        self.dim_g = geo_feature_dim
        self.relation = nn.ModuleList()
        for N in range(self.Nr):
            self.relation.append(RelationUnit(appearance_feature_dim, key_feature_dim, geo_feature_dim))

    def forward(self, input_data ):
        f_a, position_embedding = input_data
        
        isFirst=True
        for N in range(self.Nr):
            if(isFirst):
                concat = self.relation[N](f_a,position_embedding)
                isFirst=False
            else:
                concat = torch.cat((concat, self.relation[N](f_a, position_embedding)), -1)
        return (concat+f_a)

class RelationUnit(nn.Module):
    def __init__(self, appearance_feature_dim=1024,key_feature_dim = 64, geo_feature_dim = 64):
        super(RelationUnit, self).__init__()
        self.dim_g = geo_feature_dim
        self.dim_k = key_feature_dim
        self.WG = nn.Linear(geo_feature_dim, 1, bias=True)
        self.WK = nn.Linear(appearance_feature_dim, key_feature_dim, bias=True)
        self.WQ = nn.Linear(appearance_feature_dim, key_feature_dim, bias=True)
        self.WV = nn.Linear(appearance_feature_dim, key_feature_dim, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, f_a, position_embedding):
        N,_ = f_a.size()

        position_embedding = position_embedding.view(-1,self.dim_g)

        w_g = self.relu(self.WG(position_embedding))
        w_k = self.WK(f_a)
        w_k = w_k.view(N,1,self.dim_k)

        w_q = self.WQ(f_a)
        w_q = w_q.view(1,N,self.dim_k)

        scaled_dot = torch.sum((w_k*w_q),-1 )
        scaled_dot = scaled_dot / np.sqrt(self.dim_k)

        w_g = w_g.view(N,N)
        w_a = scaled_dot.view(N,N)

        w_mn = torch.log(torch.clamp(w_g, min = 1e-6)) + w_a
        w_mn = torch.nn.Softmax(dim=1)(w_mn)

        w_v = self.WV(f_a)

        w_mn = w_mn.view(N,N,1)
        w_v = w_v.view(N,1,-1)

        output = w_mn*w_v

        output = torch.sum(output,-2)
        return output
