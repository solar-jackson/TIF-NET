import torch
from torch import nn
import torch.nn.functional as F
from torchvision import ops
from networks.backbone_share import FPN
from networks.hand_head import hand_Encoder, hand_regHead
from networks.object_head import obj_regHead, Pose2DLayer
from networks.mano_head import mano_regHead
from networks.ODF import Transformer
from networks.loss import Joint2DLoss, ManoLoss, ObjectLoss
from networks.HTPE import KnowledgeGuidedModule
from networks.BVTF import CrossAttention

class Lightweight2DPoseBranch(nn.Module):
    def __init__(self, in_channels=512, joint_nb=21, hidden_dim=256):
        super().__init__()
        self.joint_nb = joint_nb

        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, joint_nb, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.conv1.apply(weights_init_kaiming)
        self.conv2.apply(weights_init_kaiming)
        self.conv3.apply(init_weights)

    def forward(self, x, return_logits=False):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        logits = self.conv3(x)
        heatmap = self.sigmoid(logits)
        return (logits, heatmap) if return_logits else heatmap

def heatmap_to_2d_joints(logits):
    b, j, height, width = logits.shape
    probs = torch.softmax(logits.flatten(2), dim=-1).reshape(b, j, height, width)
    yy, xx = torch.meshgrid(
        torch.arange(height, device=logits.device, dtype=logits.dtype),
        torch.arange(width, device=logits.device, dtype=logits.dtype),
        indexing="ij",
    )
    coords = torch.stack([
        (probs * xx).sum(dim=(-2, -1)),
        (probs * yy).sum(dim=(-2, -1))
    ], dim=-1)
    return coords, probs

def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight, std=0.001)
    elif type(m) == nn.Conv2d:
        nn.init.normal_(m.weight, std=0.001)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
        nn.init.constant_(m.bias, 0)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)

class HONet(nn.Module):
    def __init__(self, roi_res=32, joint_nb=21, stacks=1, channels=256, blocks=1,
                 transformer_depth=1, transformer_head=8,
                 mano_layer=None, mano_neurons=[1024, 512], coord_change_mat=None,
                 reg_object=True, pretrained=True,
                 interaction_mode="hand_to_object",
                 detach_interaction_source=True):

        super(HONet, self).__init__()

        self.out_res = roi_res
        self.joint_nb = joint_nb
        self.inp_channels = channels
        allowed = {"none", "hand_to_object", "bidirectional"}
        if interaction_mode not in allowed:
            raise ValueError("interaction_mode must be one of: {}".format(sorted(allowed)))
        self.interaction_mode = interaction_mode
        self.detach_interaction_source = detach_interaction_source

        self.base_net = FPN(pretrained=pretrained)

        self.HTPE = KnowledgeGuidedModule(2, 1024, order=2)
        self.topology_projection = nn.Linear(1024, channels)

        self.BVTF = CrossAttention(channel_a=256, channel_b=256, out_channel=256)

        self.pose2d_branch = Lightweight2DPoseBranch(
            in_channels=self.inp_channels,
            joint_nb=self.joint_nb,
            hidden_dim=256
        )

        self.hand_head = hand_regHead(roi_res=roi_res, joint_nb=joint_nb,
                                      stacks=stacks, channels=channels, blocks=blocks)

        self.hand_encoder = hand_Encoder(num_heatmap_chan=joint_nb, num_feat_chan=channels,
                                         size_input_feature=(roi_res, roi_res))

        self.mano_branch = mano_regHead(mano_layer, feature_size=self.hand_encoder.num_feat_out,
                                        mano_neurons=mano_neurons, coord_change_mat=coord_change_mat)

        self.reg_object = reg_object
        self.obj_head = obj_regHead(channels=channels, inter_channels=channels // 2, joint_nb=joint_nb)
        self.obj_reorgLayer = Pose2DLayer(joint_nb=joint_nb)

        self.transformer_obj = Transformer(inp_res=roi_res, dim=channels, depth=transformer_depth,
                                           num_heads=transformer_head)
        self.transformer_obj_to_hand = (
            Transformer(inp_res=roi_res, dim=channels, depth=transformer_depth,
                        num_heads=transformer_head)
            if interaction_mode == "bidirectional" else None)

        self.hand_head.apply(init_weights)
        self.hand_encoder.apply(init_weights)
        self.mano_branch.apply(init_weights)
        self.transformer_obj.apply(init_weights)
        if self.transformer_obj_to_hand is not None:
            self.transformer_obj_to_hand.apply(init_weights)
        self.obj_head.apply(init_weights)
        self.HTPE.apply(init_weights)
        self.pose2d_branch.apply(init_weights)

    def net_forward(self, imgs, bbox_hand, bbox_obj, mano_params=None, roots3d=None):
        batch = imgs.shape[0]
        idx_tensor = torch.arange(batch, device=imgs.device, dtype=bbox_hand.dtype).view(-1, 1)
        roi_boxes_hand = torch.cat((idx_tensor, bbox_hand), dim=1)
        p2_hand, p2_object = self.base_net(imgs)

        x_hand = ops.roi_align(
            p2_hand, roi_boxes_hand, output_size=(self.out_res, self.out_res),
            spatial_scale=1.0 / 4.0, sampling_ratio=-1)
        pose_logits, pose2d_heatmap = self.pose2d_branch(x_hand, return_logits=True)
        pred_joints2d, normalized_heatmaps = heatmap_to_2d_joints(pose_logits)

        node_features = self.topology_projection(self.HTPE(pred_joints2d))
        topology_map = torch.einsum("bjhw,bjc->bchw", normalized_heatmaps, node_features)
        fused_hand_feat = self.BVTF(FI=x_hand, FP=topology_map)

        preds_obj = None
        if self.reg_object:
            roi_boxes_obj = torch.cat((idx_tensor, bbox_obj), dim=1)
            y_object = ops.roi_align(
                p2_object, roi_boxes_obj, output_size=(self.out_res, self.out_res),
                spatial_scale=1.0 / 4.0, sampling_ratio=-1)

            inter_topleft = torch.maximum(bbox_hand[:, :2], bbox_obj[:, :2])
            inter_bottomright = torch.minimum(bbox_hand[:, 2:], bbox_obj[:, 2:])
            valid_inter = ((inter_bottomright - inter_topleft) > 0).all(dim=1)
            bbox_inter_safe = torch.where(
                valid_inter[:, None],
                torch.cat((inter_topleft, inter_bottomright), dim=-1),
                bbox_hand)
            roi_boxes_inter = torch.cat((idx_tensor, bbox_inter_safe), dim=1)
            x_overlap = ops.roi_align(
                p2_hand, roi_boxes_inter, output_size=(self.out_res, self.out_res),
                spatial_scale=1.0 / 4.0, sampling_ratio=-1)
            x_overlap = valid_inter[:, None, None, None] * x_overlap

            if self.interaction_mode == "none":
                enhanced_object = y_object
            else:
                hand_key_value = (
                    x_overlap.detach() if self.detach_interaction_source else x_overlap)
                candidate_object = self.transformer_obj(y_object, hand_key_value)
                enhanced_object = torch.where(
                    valid_inter[:, None, None, None], candidate_object, y_object)

            if self.interaction_mode == "bidirectional":
                object_key_value = (
                    y_object.detach() if self.detach_interaction_source else y_object)
                candidate_hand = self.transformer_obj_to_hand(
                    fused_hand_feat, object_key_value)
                fused_hand_feat = torch.where(
                    valid_inter[:, None, None, None], candidate_hand, fused_hand_feat)

            out_fm = self.obj_head(enhanced_object)
            preds_obj = self.obj_reorgLayer(out_fm)

        out_hm, encoding, preds_joints = self.hand_head(fused_hand_feat)
        mano_encoding = self.hand_encoder(out_hm, encoding)
        pred_mano_results, gt_mano_results = self.mano_branch(
            mano_encoding, mano_params=mano_params, roots3d=roots3d)

        return (preds_joints, pred_mano_results, gt_mano_results,
                preds_obj, pose2d_heatmap, pred_joints2d)

    def new_method(self, imgs):
        batch = imgs.shape[0]
        return batch

    def forward(self, imgs, bbox_hand, bbox_obj, mano_params=None, roots3d=None):
        if self.training:
            preds_joints, pred_mano_results, gt_mano_results, preds_obj, pose2d_heatmap, pred_joints2d = self.net_forward(
                imgs, bbox_hand, bbox_obj, mano_params=mano_params
            )
            return preds_joints, pred_mano_results, gt_mano_results, preds_obj, pose2d_heatmap, pred_joints2d
        else:
            preds_joints, pred_mano_results, _, preds_obj, _, pred_joints2d = self.net_forward(
                imgs, bbox_hand, bbox_obj, roots3d=roots3d
            )
            return preds_joints, pred_mano_results, preds_obj, pred_joints2d

class HOModel(nn.Module):

    def __init__(self, honet, mano_lambda_verts3d=None,
                 mano_lambda_joints3d=None,
                 mano_lambda_manopose=None,
                 mano_lambda_manoshape=None,
                 mano_lambda_regulshape=None,
                 mano_lambda_regulpose=None,
                 lambda_joints2d=None,
                 lambda_objects=None,
                 lambda_pose2d_heatmap=1.0):

        super(HOModel, self).__init__()
        self.honet = honet

        self.mano_loss = ManoLoss(lambda_verts3d=mano_lambda_verts3d,
                                  lambda_joints3d=mano_lambda_joints3d,
                                  lambda_manopose=mano_lambda_manopose,
                                  lambda_manoshape=mano_lambda_manoshape)
        self.joint2d_loss = Joint2DLoss(lambda_joints2d=lambda_joints2d)

        self.mano_joint_loss = ManoLoss(lambda_joints3d=mano_lambda_joints3d,
                                        lambda_regulshape=mano_lambda_regulshape,
                                        lambda_regulpose=mano_lambda_regulpose)

        self.object_loss = ObjectLoss(obj_reg_loss_weight=lambda_objects)

        self.lambda_pose2d_heatmap = lambda_pose2d_heatmap
        self.pose2d_heatmap_loss = nn.MSELoss()

    def forward(self, imgs, bbox_hand, bbox_obj,
                joints_uv=None, joints_xyz=None, mano_params=None, roots3d=None,
                obj_p2d_gt=None, obj_mask=None, obj_lossmask=None,
                pose2d_heatmap_gt=None):

        if self.training:
            losses = {}
            total_loss = 0
            preds_joints2d, pred_mano_results, gt_mano_results, preds_obj, pose2d_heatmap, _ = self.honet(
                imgs, bbox_hand, bbox_obj,mano_params=mano_params
            )
            if mano_params is not None:
                mano_total_loss, mano_losses = self.mano_loss.compute_loss(pred_mano_results, gt_mano_results)
                total_loss += mano_total_loss
                for key, val in mano_losses.items():
                    losses[key] = val
            if joints_uv is not None:
                joint2d_loss, joint2d_losses = self.joint2d_loss.compute_loss(preds_joints2d, joints_uv)
                for key, val in joint2d_losses.items():
                    losses[key] = val
                total_loss += joint2d_loss
            if preds_obj is not None:
                obj_total_loss, obj_losses = self.object_loss.compute_loss(obj_p2d_gt, obj_mask, preds_obj,
                                                                           obj_lossmask=obj_lossmask)
                for key, val in obj_losses.items():
                    losses[key] = val
                total_loss += obj_total_loss
            if total_loss is not None:
                losses["total_loss"] = total_loss.detach().cpu()
            else:
                losses["total_loss"] = 0

            if self.lambda_pose2d_heatmap > 0 and pose2d_heatmap_gt is None:
                raise ValueError(
                    "JSE supervision is enabled but pose2d_heatmap_gt is missing. "
                    "The dataset loader must generate Gaussian heatmaps in the hand RoI.")
            if pose2d_heatmap_gt is not None:
                if pose2d_heatmap.shape != pose2d_heatmap_gt.shape:
                    raise ValueError(
                        "JSE heatmap GT must have shape {}, got {}. "
                        "Generate Gaussian targets in hand RoI coordinates."
                        .format(tuple(pose2d_heatmap.shape),
                                tuple(pose2d_heatmap_gt.shape)))
                pose2d_loss = self.pose2d_heatmap_loss(pose2d_heatmap, pose2d_heatmap_gt)
                pose2d_loss = pose2d_loss * self.lambda_pose2d_heatmap
                losses["pose2d_heatmap_loss"] = pose2d_loss.detach().cpu()
                total_loss += pose2d_loss

            losses["total_loss"] = total_loss.detach().cpu() if total_loss != 0 else 0
            return total_loss, losses

        else:
            preds_joints, pred_mano_results, _, preds_obj, _, _ = self.honet.module.net_forward(imgs, bbox_hand, bbox_obj,
                                                                                          roots3d=roots3d)
            return preds_joints, pred_mano_results, preds_obj

