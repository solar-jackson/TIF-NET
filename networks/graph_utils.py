import torch
import scipy
import math
import scipy.sparse as sp
import numpy as np
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from utils.mano import MANO

def graph_conv_cheby(x, cl, bn, L, Fout, K):

    B, V, Fin = x.size()
    B, V, Fin = int(B), int(V), int(Fin)

    x0 = x.permute(1, 2, 0).contiguous()
    x0 = x0.view([V, Fin * B])
    x = x0.unsqueeze(0)

    def concat(x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat((x, x_), 0)

    if K > 1:
        x1 = torch.sparse.mm(L, x0)
        x = torch.cat((x, x1.unsqueeze(0)), 0)
    for k in range(2, K):
        x2 = 2 * torch.sparse.mm(L, x1) - x0
        x = torch.cat((x, x2.unsqueeze(0)), 0)
        x0, x1 = x1, x2

    x = x.view([K, V, Fin, B])
    x = x.permute(3, 1, 2, 0).contiguous()
    x = x.view([B * V, Fin * K])

    x = cl(x)
    if bn is not None:
        x = bn(x)
    x = x.view([B, V, Fout])

    return x

def laplacian(W, normalized=True):

    d = W.sum(axis=0)

    if not normalized:
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        L = D - W
    else:
        d += np.spacing(np.array(0, W.dtype))
        d = 1 / np.sqrt(d)
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        I = scipy.sparse.identity(d.size, dtype=W.dtype)
        L = I - D * W * D

    assert np.abs(L - L.T).mean() < 1e-9
    assert type(L) is scipy.sparse.csr.csr_matrix
    return L

def build_adj(joint_num, skeleton, flip_pairs):
    adj_matrix = np.zeros((joint_num, joint_num))
    for line in skeleton:
        adj_matrix[line] = 1
        adj_matrix[line[1], line[0]] = 1
    for lr in flip_pairs:
        adj_matrix[lr] = 1
        adj_matrix[lr[1], lr[0]] = 1

    return adj_matrix + np.eye(joint_num)

def build_graphs(joint_num, skeleton, flip_pairs, levels=9):
    joint_adj = build_adj(joint_num, skeleton, flip_pairs)
    input_Adj = sp.csr_matrix(joint_adj)
    input_Adj.eliminate_zeros()
    input_L = laplacian(input_Adj, normalized=True)

    return input_Adj, input_L

def sparse_python_to_torch(sp_python):
    L = sp_python.tocoo()
    indices = np.column_stack((L.row, L.col)).T
    indices = indices.astype(np.int64)
    indices = torch.from_numpy(indices)
    indices = indices.type(torch.LongTensor)
    L_data = L.data.astype(np.float32)
    L_data = torch.from_numpy(L_data)
    L_data = L_data.type(torch.FloatTensor)
    L = torch.sparse.FloatTensor(indices, L_data, torch.Size(L.shape))

    return L

if __name__ == '__main__':
    mano = MANO()
    joint_hori_conn = (
    (1, 5), (5, 9), (9, 13), (13, 17), (2, 6), (6, 10), (10, 14), (14, 18), (3, 7), (7, 11), (11, 15), (15, 19), (4, 8),
    (8, 12), (12, 16), (16, 20))
    graph_Adj, graph_L = build_graphs(joint_num=21, skeleton=mano.skeleton, flip_pairs=joint_hori_conn)
