import torch
from manopth.manolayer import ManoLayer
from manopth import demo

batch_size = 10

ncomps = 6

mano_layer = ManoLayer(
    mano_root='mano/models', use_pca=True, ncomps=ncomps, flat_hand_mean=False)

random_shape = torch.rand(batch_size, 10)

random_pose = torch.rand(batch_size, ncomps + 3)

hand_verts, hand_joints = mano_layer(random_pose, random_shape)
demo.display_hand({
    'verts': hand_verts,
    'joints': hand_joints
},
                  mano_faces=mano_layer.th_faces)

