# ------------------------------------------------------------------------
# Group-Free
# Copyright (c) 2021 Ze Liu. All Rights Reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------

"""Helper functions for calculating 2D and 3D bounding box IoU.

Collected and written by Charles R. Qi
Last modified: Jul 2019
"""
from __future__ import print_function

import numpy as np
from scipy.spatial import ConvexHull
import cv2


def rotx(t):
    """3D Rotation about the x-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def draw_projected_box3d(image, qs, color=(0, 255, 0), thickness=5):
    """Draw 3d bounding box in image
    qs: (8,3) array of vertices for the 3d box in following order:
        1 -------- 0
       /|         /|
      2 -------- 3 .
      | |        | |
      . 5 -------- 4
      |/         |/
      6 -------- 7
    """
    # Handle PIL image input
    if hasattr(image, "convert"):  # Check if it's a PIL image
        # Convert to RGB mode and then to numpy array
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = np.array(image)
        # PIL images are RGB; convert to BGR for OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    qs = qs.astype(np.int32)
    for k in range(0, 4):
        # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
        i, j = k, (k + 1) % 4
        # use LINE_AA for opencv3
        # cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.CV_AA)
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)

        i, j = k, k + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
    return image


def conver_box2d(boxes_3d, image_shape, item_info):
    """
    boxes_3d: (N, 7) or (7,) for single box
    image_shape: (H, W)
    item_info: dict

    return:
        corners_2d: (N, 8, 2)
    """
    # Handle single box input (1D array)
    if boxes_3d.ndim == 1:
        boxes_3d = boxes_3d[None, :]  # Add batch dimension

    extristric = np.array(item_info["image_extrinsic"])
    assert extristric.shape == (4, 4)
    # try:
    #     extristric = np.array(item_info["image_extrinsic"])
    #     extristric = np.linalg.inv(extristric)
    #     axis_tf = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
    #     extristric = np.matmul(axis_tf, extristric)
    # except:
    #     extristric = np.array(item_info["extristric"])
    K = np.array(item_info["image_intrinsic"])[:3, :3]
    if "camera_distortion" in item_info:
        D = np.array(item_info["camera_distortion"])
    else:
        # Use default distortion coefficients if not provided
        D = np.zeros(([4]), dtype=np.float32)
    num_box = boxes_3d.shape[0]
    boxes_vec_points = np.zeros([num_box, 3, 9])
    l, w, h = boxes_3d[:, 3], boxes_3d[:, 4], boxes_3d[:, 5]
    c_xyz = boxes_3d[:, :3][:, :, None]

    x_corners = [
        l / 2,
        l / 2,
        -l / 2,
        -l / 2,
        l / 2,
        l / 2,
        -l / 2,
        -l / 2,
        np.zeros([num_box]),
    ]
    z_corners = [
        h / 2,
        h / 2,
        h / 2,
        h / 2,
        -h / 2,
        -h / 2,
        -h / 2,
        -h / 2,
        np.zeros([num_box]),
    ]
    y_corners = [
        w / 2,
        -w / 2,
        -w / 2,
        w / 2,
        w / 2,
        -w / 2,
        -w / 2,
        w / 2,
        np.zeros([num_box]),
    ]

    boxes_vec_points[:, 0, :] = np.transpose(np.stack(x_corners))
    boxes_vec_points[:, 1, :] = np.transpose(np.stack(y_corners))
    boxes_vec_points[:, 2, :] = np.transpose(np.stack(z_corners))

    rotzs = []
    for box in boxes_3d:
        rotzs.append(rotz(box[6]))
    rotzs = np.stack(rotzs)

    corners_3d = rotzs @ boxes_vec_points  # N, 3, 9
    corners_3d += c_xyz
    corners_3d = np.transpose(corners_3d, (0, 2, 1)).reshape(-1, 3)
    extend_points = cart_to_hom(corners_3d[:, :3])

    points_cam = extend_points @ extristric.T
    rvecs = np.zeros((3, 1))
    tvecs = np.zeros((3, 1))
    depth = points_cam[:, 2]

    pts_img, _ = cv2.projectPoints(
        points_cam[:, :3].astype(np.float32), rvecs, tvecs, K, D
    )

    corners_2d = pts_img[:, 0, :]
    corners_2d = corners_2d.reshape(num_box, 9, 2)
    depth = depth.reshape(num_box, 9)
    centers_2d = corners_2d[:, -1, :]
    centers_depth = depth[:, -1]

    kept1 = (
        (centers_2d[:, 1] >= 0)
        & (centers_2d[:, 1] < image_shape[1])
        & (centers_2d[:, 0] >= 0)
        & (centers_2d[:, 0] < image_shape[0])
        & (centers_depth > 0.1)
    )

    return corners_2d[:, :8, :], None


def cart_to_hom(pts):
    """
    :param pts: (N, 3 or 2)
    :return pts_hom: (N, 4 or 3)
    """
    pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
    return pts_hom

def draw_points_on_image(
    image,
    points_2d,
    color=(0, 255, 0),
    radius=2,
    save_path=None,
    create_mask=False,
    valid_mask=None,
):
    # Ensure image is BGR np.uint8
    if hasattr(image, "convert"):
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    result_image = image.copy()
    H, W = result_image.shape[:2]

    P = np.asarray(points_2d, dtype=np.float32)
    if valid_mask is not None:
        P = P[np.asarray(valid_mask, dtype=bool)]
    if P.shape[0] == 0:
        if create_mask:
            mask = np.zeros((H, W), np.uint8)
            return result_image, mask
        return result_image

    # Round to int, clip to image boundary
    u = np.rint(P[:, 0]).astype(np.int32)
    v = np.rint(P[:, 1]).astype(np.int32)
    inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v = u[inb], v[inb]

    # Draw points (optional)
    for x, y in zip(u, v):
        cv2.circle(result_image, (x, y), radius, color, -1)

    if not create_mask:
        # if save_path is not None:
        #     cv2.imwrite(save_path, result_image)
        #     print(f"    Point cloud image saved: {save_path}")
        return result_image

    # Single channel uint8 mask: set points to 255, then morphological connectivity
    mask = np.zeros((H, W), np.uint8)
    mask[v, u] = 255

    # Make sparse "multi-row points" connect: adjust kernel/iterations by point density
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # if save_path is not None:
    #     cv2.imwrite(save_path, result_image)
    #     print(f"    Point cloud image saved: {save_path}")
    return result_image, mask


def extract_points_in_bbox_3d(lidar_points, bbox_3d):
    """
    Extract points inside a 3D bounding box from point cloud

    Args:
        lidar_points: point cloud data (N, 4) - [x, y, z, intensity]
        bbox_3d: 3D bounding box [x, y, z, l, w, h, yaw] or [x, y, z, l, w, h, yaw, pitch, roll]
        
        7-dim bbox (waymo): [x, y, z, l, w, h, yaw]
        Only yaw rotation (about z-axis)
        9-dim bbox (quad, drone): [x, y, z, l, w, h, yaw, pitch, roll]
        Three rotation angles: yaw (about z), pitch (about y), roll (about x)

    Returns:
        np.ndarray: points inside bounding box
    """
    # Handle different bbox_3d lengths
    if len(bbox_3d) == 7:
        x, y, z, l, w, h, yaw = bbox_3d
        pitch, roll = 0.0, 0.0  # default value
    elif len(bbox_3d) == 9:
        x, y, z, l, w, h, yaw, pitch, roll = bbox_3d
    else:
        raise ValueError(f"bbox_3d must have length 7 or 9, got {len(bbox_3d)}")

    # Create rotation matrix
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    cos_pitch = np.cos(pitch)
    sin_pitch = np.sin(pitch)
    cos_roll = np.cos(roll)
    sin_roll = np.sin(roll)

    # Rotation matrix (simplified version, only yaw considered)
    R = np.array([[cos_yaw, -sin_yaw, 0], [sin_yaw, cos_yaw, 0], [0, 0, 1]])

    # Transform points to box coordinate system
    points_centered = lidar_points[:, :3] - np.array([x, y, z])
    points_rotated = (R.T @ points_centered.T).T

    # Check if points are inside the box
    half_l, half_w, half_h = l / 2, w / 2, h / 2
    mask = (
        (points_rotated[:, 0] >= -half_l)
        & (points_rotated[:, 0] <= half_l)
        & (points_rotated[:, 1] >= -half_w)
        & (points_rotated[:, 1] <= half_w)
        & (points_rotated[:, 2] >= -half_h)
        & (points_rotated[:, 2] <= half_h)
    )

    return lidar_points[mask], mask


def project_points_to_2d(points_3d, image_shape, item_info):
    """
    Project 3D points onto a 2D image plane

    Args:
        points_3d: (N, 3) 3D point cloud coordinates
        image_shape: (H, W) image shape
        item_info: dict, containing camera parameters

    Returns:
        points_2d: (N, 2) projected 2D coordinates
        depth: (N,) depth values
        valid_mask: (N,) valid points mask
    """
    # Get camera extrinsic
    # try:
    extrinsic = np.array(item_info["image_extrinsic"])
    # extrinsic = np.linalg.inv(extrinsic)
    # # Only apply axis_tf if shape is not 4x4
    assert extrinsic.shape == (
        4,
        4,
    ), f"extrinsic shape must be (4, 4), got {extrinsic.shape}"
    #     axis_tf = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
    #     extrinsic = np.matmul(axis_tf, extrinsic)
    # except:
    #     extrinsic = np.array(item_info["extristric"])

    # Get camera intrinsic
    K = np.array(item_info["image_intrinsic"])[:3, :3]

    # Get distortion parameters
    if "camera_distortion" in item_info:
        D = np.array(item_info["camera_distortion"])
    else:
        # If distortion not provided, use default
        D = np.zeros(4, dtype=np.float32)

    # Convert 3D points to homogeneous coordinates
    points_hom = cart_to_hom(points_3d)

    # Transform points from world to camera coordinate system
    points_cam = points_hom @ extrinsic.T

    # Get depth values
    depth = points_cam[:, 2]

    # Set rotation and translation vectors (use zero because transformation is in extrinsic)
    rvecs = np.zeros((3, 1))
    tvecs = np.zeros((3, 1))

    # Project points to image plane
    pts_img, _ = cv2.projectPoints(
        points_cam[:, :3].astype(np.float32), rvecs, tvecs, K, D
    )

    # Extract 2D coordinates
    points_2d = pts_img[:, 0, :]

    # Create valid mask: False if projection is invalid (out of image bounds or unreasonable depth)
    # 1. Image boundary check:
    # points_2d[:, 1] >= 0 and points_2d[:, 1] < image_shape[0]: check projected y in image height
    # points_2d[:, 0] >= 0 and points_2d[:, 0] < image_shape[1]: check projected x in image width
    # 2. Depth reasonableness check:
    # depth > 0.1: keep points with depth larger than 0.1 meters, filter out close/invalid depth
    valid_mask = (
        (points_2d[:, 1] >= 0)
        & (points_2d[:, 1] < image_shape[0])
        & (points_2d[:, 0] >= 0)
        & (points_2d[:, 0] < image_shape[1])
        & (depth > 0.1)
    )

    return points_2d, depth, valid_mask


def polygon_clip(subjectPolygon, clipPolygon):
    """Clip a polygon with another polygon.

    Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python

    Args:
      subjectPolygon: a list of (x,y) 2d points, any polygon.
      clipPolygon: a list of (x,y) 2d points, has to be *convex*
    Note:
      **points have to be counter-clockwise ordered**

    Return:
      a list of (x,y) vertex point for the intersection polygon.
    """

    def inside(p):
        return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) > (cp2[1] - cp1[1]) * (p[0] - cp1[0])

    def computeIntersection():
        dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
        dp = [s[0] - e[0], s[1] - e[1]]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
        return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]

    outputList = subjectPolygon
    cp1 = clipPolygon[-1]

    for clipVertex in clipPolygon:
        cp2 = clipVertex
        inputList = outputList
        outputList = []
        s = inputList[-1]

        for subjectVertex in inputList:
            e = subjectVertex
            if inside(e):
                if not inside(s):
                    outputList.append(computeIntersection())
                outputList.append(e)
            elif inside(s):
                outputList.append(computeIntersection())
            s = e
        cp1 = cp2
        if len(outputList) == 0:
            return None
    return outputList


def poly_area(x, y):
    """Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates"""
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def convex_hull_intersection(p1, p2):
    """Compute area of two convex hull's intersection area.
    p1,p2 are a list of (x,y) tuples of hull vertices.
    return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1, p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0


def box3d_vol(corners):
    """corners: (8,3) no assumption on axis direction"""
    a = np.sqrt(np.sum((corners[0, :] - corners[1, :]) ** 2))
    b = np.sqrt(np.sum((corners[1, :] - corners[2, :]) ** 2))
    c = np.sqrt(np.sum((corners[0, :] - corners[4, :]) ** 2))
    return a * b * c


def is_clockwise(p):
    x = p[:, 0]
    y = p[:, 1]
    return np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)) > 0


def box3d_iou(corners1, corners2):
    """Compute 3D bounding box IoU.

    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU

    todo (rqi): add more description on corner points' orders.
    """
    # corner points are in counter clockwise order
    rect1 = [(corners1[i, 0], corners1[i, 2]) for i in range(3, -1, -1)]
    rect2 = [(corners2[i, 0], corners2[i, 2]) for i in range(3, -1, -1)]
    area1 = poly_area(np.array(rect1)[:, 0], np.array(rect1)[:, 1])
    area2 = poly_area(np.array(rect2)[:, 0], np.array(rect2)[:, 1])
    inter, inter_area = convex_hull_intersection(rect1, rect2)
    iou_2d = inter_area / (area1 + area2 - inter_area)
    ymax = min(corners1[0, 1], corners2[0, 1])
    ymin = max(corners1[4, 1], corners2[4, 1])
    inter_vol = inter_area * max(0.0, ymax - ymin)
    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    return iou, iou_2d


def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two 2D bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1["x1"] < bb1["x2"]
    assert bb1["y1"] < bb1["y2"]
    assert bb2["x1"] < bb2["x2"]
    assert bb2["y1"] < bb2["y2"]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1["x1"], bb2["x1"])
    y_top = max(bb1["y1"], bb2["y1"])
    x_right = min(bb1["x2"], bb2["x2"])
    y_bottom = min(bb1["y2"], bb2["y2"])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1["x2"] - bb1["x1"]) * (bb1["y2"] - bb1["y1"])
    bb2_area = (bb2["x2"] - bb2["x1"]) * (bb2["y2"] - bb2["y1"])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def box2d_iou(box1, box2):
    """Compute 2D bounding box IoU.

    Input:
        box1: tuple of (xmin,ymin,xmax,ymax)
        box2: tuple of (xmin,ymin,xmax,ymax)
    Output:
        iou: 2D IoU scalar
    """
    return get_iou(
        {"x1": box1[0], "y1": box1[1], "x2": box1[2], "y2": box1[3]},
        {"x1": box2[0], "y1": box2[1], "x2": box2[2], "y2": box2[3]},
    )


# -----------------------------------------------------------
# Convert from box parameters to
# -----------------------------------------------------------
def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def roty_batch(t):
    """Rotation about the y-axis.
    t: (x1,x2,...xn)
    return: (x1,x2,...,xn,3,3)
    """
    input_shape = t.shape
    output = np.zeros(tuple(list(input_shape) + [3, 3]))
    c = np.cos(t)
    s = np.sin(t)
    output[..., 0, 0] = c
    output[..., 0, 2] = s
    output[..., 1, 1] = 1
    output[..., 2, 0] = -s
    output[..., 2, 2] = c
    return output


def get_3d_box(box_size, heading_angle, center):
    """box_size is array(l,w,h), heading_angle is radius clockwise from pos x axis, center is xyz of box center
    output (8,3) array for 3D box cornders
    Similar to utils/compute_orientation_3d
    """
    R = roty(heading_angle)
    l, w, h = box_size
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] = corners_3d[0, :] + center[0]
    corners_3d[1, :] = corners_3d[1, :] + center[1]
    corners_3d[2, :] = corners_3d[2, :] + center[2]
    corners_3d = np.transpose(corners_3d)
    return corners_3d


def get_3d_box_batch(box_size, heading_angle, center):
    """box_size: [x1,x2,...,xn,3]
        heading_angle: [x1,x2,...,xn]
        center: [x1,x2,...,xn,3]
    Return:
        [x1,x3,...,xn,8,3]
    """
    input_shape = heading_angle.shape
    R = roty_batch(heading_angle)
    l = np.expand_dims(box_size[..., 0], -1)  # [x1,...,xn,1]
    w = np.expand_dims(box_size[..., 1], -1)
    h = np.expand_dims(box_size[..., 2], -1)
    corners_3d = np.zeros(tuple(list(input_shape) + [8, 3]))
    corners_3d[..., :, 0] = np.concatenate(
        (l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2), -1
    )
    corners_3d[..., :, 1] = np.concatenate(
        (h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2), -1
    )
    corners_3d[..., :, 2] = np.concatenate(
        (w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2), -1
    )
    tlist = [i for i in range(len(input_shape))]
    tlist += [len(input_shape) + 1, len(input_shape)]
    corners_3d = np.matmul(corners_3d, np.transpose(R, tuple(tlist)))
    corners_3d += np.expand_dims(center, -2)
    return corners_3d

