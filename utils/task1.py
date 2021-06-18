import numpy as np
import torch
import torchvision.ops as tv


def label2corners(label):
    """
    Task 1
    input
        label (N,7) 3D bounding box with (x,y,z,h,w,l,ry)
    output
        corners (N,8,3) corner coordinates in the rectified reference frame
        Outputs as x,y,z
    """
    # x: right, y: down, z: forward
    # 2D bounding
    label = torch.as_tensor(label)
    N = label.size()[0]
    x = label[..., 0]
    y = label[..., 1]
    z = label[..., 2]
    h = label[..., 3]
    w = label[..., 4]
    l = label[..., 5]
    alpha = label[..., 6, None]  # (N, 1)

    # 2D
    x4 = torch.FloatTensor([0.5, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5]).unsqueeze(0).to(label.device)  # (1,8)
    x4 = x4 * w.unsqueeze(-1)  # (N, 8)
    z4 = torch.FloatTensor([0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5]).unsqueeze(0).to(label.device)  # (1,8)
    z4 = z4 * l.unsqueeze(-1)  # (N, 8)
    y4 = torch.FloatTensor([0, 0, 0, 0, -1, -1, -1, -1]).unsqueeze(0).to(label.device)  # (1,8)
    y4 = y4 * h.unsqueeze(-1)  # (N, 8)
    corners = torch.stack([x4, y4, z4], dim=-1)  # (N, 8, 2)

    sin = torch.sin(alpha)  # (N, 1)
    cos = torch.cos(alpha)
    ones = torch.ones(alpha.shape)
    zeros = torch.zeros(alpha.shape)
    row1 = torch.cat([cos, zeros, sin], dim=-1)
    row2 = torch.cat([zeros, ones, zeros], dim=-1)
    row3 = torch.cat([-sin, zeros, cos], dim=-1)  # (N, 3)
    rot_T = torch.stack([row1, row2, row3], dim=-2)  # (N, 2, 2)

    corners3d = torch.bmm(corners, rot_T)  # (N, 8, 2)

    corners3d[..., 0] += x.unsqueeze(-1)
    corners3d[..., 1] += y.unsqueeze(-1)
    corners3d[..., 2] += z.unsqueeze(-1)
    return corners3d


def get_iou(pred, target):
    """
    Task 1
    input
        pred (N,7) 3D bounding box corners
        target (M,7) 3D bounding box corners
    output
        iou (N,M) pairwise 3D intersection-over-union
    """
    pred = label2corners(pred)
    target = label2corners(target)
    # 2D
    pred_2d = torch.cat([pred[:, 0, [0, 2]], pred[:, 2, [0, 2]]], dim=1)  # (N, 2, 2) (Upper left and lower right
    # of square with only x, z)
    target_2d = torch.cat([target[:, 0, [0, 2]], target[:, 2, [0, 2]]], dim=1)
    iou_2d = tv.box_iou(pred_2d, target_2d)

    zmax1 = pred[:, 0, 1].unsqueeze(-1)  # (N, 1)
    zmax2 = target[:, 0, 1].unsqueeze(0)  # (1, M)
    zmin1 = pred[:, 5, 1].unsqueeze(-1)
    zmin2 = target[:, 5, 1].unsqueeze(0)

    zmax1 = torch.broadcast_to(zmax1, [pred.shape[0], target.shape[0]])  # (N, M)
    zmax2 = torch.broadcast_to(zmax2, [pred.shape[0], target.shape[0]])  # (N, M)
    zmin1 = torch.broadcast_to(zmin1, [pred.shape[0], target.shape[0]])
    zmin2 = torch.broadcast_to(zmin2, [pred.shape[0], target.shape[0]])

    z_overlap = (torch.min(zmax1, zmax2) - torch.max(zmin1, zmin2)).clamp_min(0.)
    z_range = (torch.max(zmax1, zmax2) - torch.min(zmin1, zmin2)).clamp_min(0.)

    return iou_2d * z_overlap / z_range


def compute_recall(pred, target, threshold):
    """
    Task 1
    input
        pred (N,7) proposed 3D bounding box labels
        target (M,7) ground truth 3D bounding box labels
        threshold (float) threshold for positive samples
    output
        recall (float) recall for the scene
    """
    return get_iou(pred, target)[get_iou(pred, target) > threshold]
