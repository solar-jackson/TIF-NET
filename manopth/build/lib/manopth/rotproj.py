import torch

def batch_rotprojs(batches_rotmats):
    proj_rotmats = []
    for batch_idx, batch_rotmats in enumerate(batches_rotmats):
        proj_batch_rotmats = []
        for rot_idx, rotmat in enumerate(batch_rotmats):

            U, S, V = rotmat.cpu().svd()
            rotmat = torch.matmul(U, V.transpose(0, 1))
            orth_det = rotmat.det()

            if orth_det < 0:
                rotmat[:, 2] = -1 * rotmat[:, 2]

            rotmat = rotmat.cuda()
            proj_batch_rotmats.append(rotmat)
        proj_rotmats.append(torch.stack(proj_batch_rotmats))
    return torch.stack(proj_rotmats)

