import argparse
import os
from dataclasses import dataclass
from datetime import datetime

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# set matplotlib backend to webagg for p36 compatibility


import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.qhull import QhullError # import here for p36 compatibility

CLASS_LABELS = ["No Thrust", "Chemical", "Electric", "Impulsive"]

def parse_args():
    parser = argparse.ArgumentParser(description="Plot combined point cloud for reachability ensemble.")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the combined point cloud.")
    parser.add_argument("--webagg", action="store_true", help="Use WebAgg backend for matplotlib (for p36 compatibility).")
    return parser.parse_args()

def alpha_shape_faces_and_volume(points, edge_quantile=0.95):
    n = points.shape[0]
    if n < 5:
        hull = ConvexHull(points)
        return points[hull.simplices], hull.volume

    try:
        tet = Delaunay(points)
    except QhullError:
        hull = ConvexHull(points)
        return points[hull.simplices], hull.volume

    simplices = tet.simplices  # (m, 4)
    p = points[simplices]      # (m, 4, 3)

    e01 = np.linalg.norm(p[:, 1] - p[:, 0], axis=1)
    e02 = np.linalg.norm(p[:, 2] - p[:, 0], axis=1)
    e03 = np.linalg.norm(p[:, 3] - p[:, 0], axis=1)
    e12 = np.linalg.norm(p[:, 2] - p[:, 1], axis=1)
    e13 = np.linalg.norm(p[:, 3] - p[:, 1], axis=1)
    e23 = np.linalg.norm(p[:, 3] - p[:, 2], axis=1)
    max_edge = np.maximum.reduce([e01, e02, e03, e12, e13, e23])

    cross = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    tet_vol = np.abs(np.einsum("ij,ij->i", cross, p[:, 3] - p[:, 0])) / 6.0
    valid = tet_vol > 1e-14
    if not np.any(valid):
        hull = ConvexHull(points)
        return points[hull.simplices], hull.volume

    thresh = np.quantile(max_edge[valid], edge_quantile)
    keep = valid & (max_edge <= thresh)
    if not np.any(keep):
        hull = ConvexHull(points)
        return points[hull.simplices], hull.volume

    kept = simplices[keep]
    faces = np.concatenate(
        [
            kept[:, [0, 1, 2]],
            kept[:, [0, 1, 3]],
            kept[:, [0, 2, 3]],
            kept[:, [1, 2, 3]],
        ],
        axis=0,
    )
    faces_sorted = np.sort(faces, axis=1)
    uniq_faces, counts = np.unique(faces_sorted, axis=0, return_counts=True)
    boundary_faces = uniq_faces[counts == 1]
    return points[boundary_faces], tet_vol[keep].sum()


def main(args):

    # Load the predicted point clouds and class labels
    data_path = os.path.join(args.save_dir, "predicted_reachability_point_clouds.npz")
    data = np.load(data_path, allow_pickle=True)
    true_rollouts = data["true_rollouts"]
    pred_rollouts = data["pred_rollouts"]
    pred_classes = data["pred_classes"]
    true_classes = data["true_classes"]
    class_clouds = [data[f"class{i}"] for i in range(4)]
    class_labels = data["class_labels"]

    print("Loaded predicted point clouds and class labels.")
    print(f"True rollouts shape: {true_rollouts.shape}")
    print(f"Pred rollouts shape: {pred_rollouts.shape}")
    print(f"Pred classes shape: {pred_classes.shape}")
    print(f"True classes shape: {true_classes.shape}")
    print(f"Class clouds shapes: {[cloud.shape for cloud in class_clouds]}")

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 8))
    colors = ["red", "blue", "green", "orange"]
    for i in range(4):
        cloud = true_rollouts[true_classes == i]
        if cloud.shape[0] > 0:
            ax.scatter(cloud[:, -1, 0], cloud[:, -1, 1],cloud[:, -1, 2], color=colors[i], label=f"{CLASS_LABELS[i]}", alpha=0.5)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_zlabel("z (km)")
    ax.set_title("True Reachability Point Cloud")
    ax.legend()
    ax.grid()

    # using final true state rollouts for alpha shape
    fig,ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 8))
    for i in range(4):
        cloud = true_rollouts[true_classes == i]
        print(cloud.shape)
        if cloud.shape[0] > 0:
            points = cloud[:, -1, :3]  # (N, 3)
            faces, volume = alpha_shape_faces_and_volume(points)
            ax.add_collection3d(Poly3DCollection(faces, alpha=0.3, facecolor=colors[i], edgecolor='k', label=f"{CLASS_LABELS[i]} (Volume: {volume:.2f} km^3)"))

    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)") 
    ax.set_zlabel("z (km)")
    ax.set_title("Alpha Shape of True Reachability Point Cloud")
    ax.legend()
    ax.grid()

    # same plots but with predicted classes
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 8))
    for i in range(4):
        cloud = pred_rollouts[pred_classes == i]
        if cloud.shape[0] > 0:
            ax.scatter(cloud[:, -1, 0], cloud[:, -1, 1],cloud[:, -1, 2], color=colors[i], label=f"{CLASS_LABELS[i]} (Pred)", alpha=0.5)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_zlabel("z (km)")
    ax.set_title("Combined Predicted Reachability Point Cloud")
    ax.legend()
    ax.grid()   

    fig,ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 8))
    for i in range(4):
        cloud = pred_rollouts[pred_classes == i]
        print(cloud.shape)
        if cloud.shape[0] > 0:
            points = cloud[:, -1, :3]  # (N, 3)
            faces, volume = alpha_shape_faces_and_volume(points)
            ax.add_collection3d(Poly3DCollection(faces, alpha=0.3, facecolor=colors[i], edgecolor='k', label=f"{CLASS_LABELS[i]} (Pred Volume: {volume:.2f} km^3)"))
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_zlabel("z (km)")
    ax.set_title("Alpha Shape of Predicted Reachability Point Cloud")
    ax.legend()
    ax.grid()

    # # plot clouds for each timestep for each class 
    # for i in range(4):
    #     fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 8))
    #     for t in range(pred_rollouts.shape[1]):
    #         cloud = pred_rollouts[pred_classes == i][:, t]
    #         if cloud.shape[0] > 0:
    #             ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], color=colors[i], label=f"{CLASS_LABELS[i]} (t={t})", alpha=0.5)
    #     ax.set_xlabel("x (km)")
    #     ax.set_ylabel("y (km)")
    #     ax.set_zlabel("z (km)")
    #     ax.set_title(f"Predicted Reachability Point Cloud for {CLASS_LABELS[i]} Over Time")
    #     ax.legend()
    #     ax.grid()

    # plot class clouds for each class predicted vs true
    fig = plt.figure(figsize=(14, 10))
    axes = [fig.add_subplot(2, 2, i + 1, projection="3d") for i in range(4)]

    for i, ax in enumerate(axes):
        true_cloud = true_rollouts[true_classes == i][:, -1, :3]
        pred_cloud = pred_rollouts[pred_classes == i][:, -1, :3]
        if true_cloud.shape[0] > 0:
            ax.scatter(true_cloud[:, 0], true_cloud[:, 1], true_cloud[:, 2], color=colors[i], label=f"{CLASS_LABELS[i]} (True)", alpha=0.5)
        if pred_cloud.shape[0] > 0:
            ax.scatter(pred_cloud[:, 0], pred_cloud[:, 1], pred_cloud[:, 2], color=colors[i], label=f"{CLASS_LABELS[i]} (Pred)", alpha=0.5,marker='x')
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")
        ax.set_zlabel("z (km)")
        ax.set_title(f"Class Cloud for {CLASS_LABELS[i]}: Predicted vs True")
        ax.legend()
        ax.grid()

    # plot class alpha shapes for each class predicted vs true
    fig = plt.figure(figsize=(14, 10))
    axes = [fig.add_subplot(2, 2, i + 1, projection="3d") for i in range(4)]

    for i, ax in enumerate(axes):
        # plot in a single image, 2x2 grid
        true_cloud = true_rollouts[true_classes == i][:, -1, :3]
        pred_cloud = pred_rollouts[pred_classes == i][:, -1, :3]
        if true_cloud.shape[0] > 0:
            faces, volume = alpha_shape_faces_and_volume(true_cloud)
            # make true color slighly transparent and pred color more opaque
            ax.add_collection3d(Poly3DCollection(faces, alpha=0.7, facecolor=colors[i], edgecolor='k', label=f"{CLASS_LABELS[i]} (True Volume: {volume:.2f} km^3)"))
        if pred_cloud.shape[0] > 0:
            faces, volume = alpha_shape_faces_and_volume(pred_cloud)
            ax.add_collection3d(Poly3DCollection(faces, alpha=0.3, facecolor=colors[i], edgecolor='k', label=f"{CLASS_LABELS[i]} (Pred Volume: {volume:.2f} km^3)"))
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")
        ax.set_zlabel("z (km)")
        ax.set_title(f"Alpha Shape for Class {CLASS_LABELS[i]}: Predicted vs True")
        ax.legend()
        ax.grid()
    plt.tight_layout()


if __name__ == "__main__":
    args = parse_args()
    if args.webagg:
        plt.switch_backend('webagg')
    main(args)

    plt.show()
