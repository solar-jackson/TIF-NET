import unittest
import torch
from torch import nn

from model import HONet, Lightweight2DPoseBranch, heatmap_to_2d_joints
from networks.HTPE import KnowledgeGuidedModule
from networks.BVTF import CrossAttention

class FakeMano(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("th_hands_mean", torch.zeros(45))

    def forward(self, th_pose_coeffs, th_betas):
        pivot = th_pose_coeffs[:, :3].reshape(-1, 1, 3)
        return pivot.repeat(1, 778, 1), pivot.repeat(1, 21, 1)

class AlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(37)

    def test_soft_argmax_is_differentiable(self):
        logits = torch.randn(2, 21, 8, 8, requires_grad=True)
        xy, prob = heatmap_to_2d_joints(logits)
        self.assertEqual(tuple(xy.shape), (2,21,2))
        self.assertTrue(torch.allclose(prob.flatten(2).sum(-1),
                                       torch.ones(2,21), atol=1e-6))
        xy.square().mean().backward()
        self.assertGreater(logits.grad.abs().sum().item(), 0)

    def test_graph_propagates_across_neighbors(self):
        module = KnowledgeGuidedModule(2, 16, order=2, widths=(16,))
        self.assertNotEqual(module.laplacian[0, 1].item(), 0)
        self.assertEqual(module.laplacian[0, 8].item(), 0)
        xy = torch.randn(2, 21, 2, requires_grad=True)
        out = module(xy)
        self.assertEqual(tuple(out.shape), (2,21,16))
        out.sum().backward()
        self.assertGreater(xy.grad.abs().sum().item(), 0)

    def test_projection_and_bidirectional_visual_topology(self):
        jse = Lightweight2DPoseBranch(in_channels=8, joint_nb=21, hidden_dim=16)
        logits, _ = jse(torch.randn(2,8,8,8), return_logits=True)
        joint_xy, prob = heatmap_to_2d_joints(logits)
        node = KnowledgeGuidedModule(2, 8, widths=(16,))(joint_xy)
        fpose = torch.einsum("bjhw,bjc->bchw", prob, node)
        fusion = CrossAttention(channel_a=8, channel_b=8,
                                out_channel=8, depth=1, num_heads=2)
        fused = fusion(torch.randn(2,8,8,8), fpose)
        self.assertEqual(tuple(fused.shape), (2,8,8,8))
        fused.sum().backward()
        self.assertIsNotNone(jse.conv3.weight.grad)

    def test_full_forward_modes_have_expected_shapes(self):
        img = torch.randn(1,3,64,64)
        hand = torch.tensor([[4.,4.,52.,52.]])
        obj = torch.tensor([[18.,18.,60.,60.]])
        for mode in ("none","hand_to_object","bidirectional"):
            model = HONet(roi_res=16, mano_layer=FakeMano(), pretrained=False,
                          interaction_mode=mode, transformer_depth=1,
                          transformer_head=4).eval()
            with torch.no_grad():
                joints, mano, _, obj_pose, heatmap, xy = model.net_forward(img,hand,obj)
            self.assertEqual(tuple(heatmap.shape),(1,21,16,16))
            self.assertEqual(tuple(xy.shape),(1,21,2))
            self.assertEqual(tuple(mano["verts3d"].shape),(1,778,3))
            self.assertEqual(len(obj_pose),3)

    def test_source_gradient_detach(self):
        from networks.ODF import Transformer
        module = Transformer(inp_res=4, dim=8, depth=1, num_heads=2)
        object_feat = torch.randn(1,8,4,4, requires_grad=True)
        hand_feat = torch.randn(1,8,4,4, requires_grad=True)
        output = module(object_feat, hand_feat.detach())
        output.sum().backward()
        self.assertIsNotNone(object_feat.grad)
        self.assertIsNone(hand_feat.grad)

if __name__ == "__main__":
    unittest.main()
