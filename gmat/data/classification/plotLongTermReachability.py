import argparse
import os
from dataclasses import dataclass
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# set matplotlib backend to webagg for p36 compatibility


import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.qhull import QhullError # import here for p36 compatibility

CLASS_LABELS = ["No Thrust", "Electric"]

def parse_args():
    parser = argparse.ArgumentParser(description="Plot combined point cloud for reachability ensemble.")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the combined point cloud.")
    parser.add_argument("--webagg", action="store_true", help="Use WebAgg backend for matplotlib (for p36 compatibility).")
    parser.add_argument("--animate", action="store_true", help="Animate the change in shape of the point cloud over time.")
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
    colors = ["red", "blue"]

    NoThrust_cloud = np.load(os.path.join(args.save_dir, "statesArrayNoThrust.npz"))["statesArrayNoThrust"]
    Electric_cloud = np.load(os.path.join(args.save_dir, "statesArrayElectric.npz"))["statesArrayElectric"]

    print(f"No Thrust cloud shape: {NoThrust_cloud.shape}") #(80 trajs, 4230 time steps, 6 dimensions)
    print(f"Electric cloud shape: {Electric_cloud.shape}") #(80 trajs, 4230 time steps, 6 dimensions)

    fig,ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 8))
    for i in range(2):
        cloud = NoThrust_cloud if i == 0 else Electric_cloud
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

    fig,ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 8))
    for i in range(2):
        cloud = NoThrust_cloud if i == 0 else Electric_cloud
        print(cloud.shape)
        if cloud.shape[0] > 0:
            points = cloud[:, 4000, :3]  # (N, 3)
            faces, volume = alpha_shape_faces_and_volume(points)
            ax.add_collection3d(Poly3DCollection(faces, alpha=0.3, facecolor=colors[i], edgecolor='k', label=f"{CLASS_LABELS[i]} (Pred Volume: {volume:.2f} km^3)"))
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_zlabel("z (km)")
    ax.set_title("Alpha Shape of Predicted Reachability Point Cloud")
    ax.legend()
    ax.grid()


    # find difference between the two clouds
    NoThrust_points = NoThrust_cloud[:, -1, :3]
    Electric_points = Electric_cloud[:, -1, :3]
    from scipy.spatial import cKDTree
    tree = cKDTree(NoThrust_points)
    dists, _ = tree.query(Electric_points, k=1)
    threshold = 10.0  # km
    far_points = Electric_points[dists > threshold]
    print(f"Number of points in Electric cloud that are more than {threshold} km from any point in No Thrust cloud: {far_points.shape[0]}")
    ax.scatter(far_points[:, 0], far_points[:, 1], far_points[:, 2], color='magenta', label=f"Electric points > {threshold} km from No Thrust", alpha=0.5)
    ax.legend()

    # # plot entire trajectory alpha shape for electric cloud
    # fig,ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 8))
    # points = Electric_cloud[:, :, :3].reshape(-1, 3)  # (N*4230, 3)
    # faces, volume = alpha_shape_faces_and_volume(points)
    # ax.add_collection3d(Poly3DCollection(faces, alpha=0.3, facecolor='orange', label=f"Electric (Pred Volume: {volume:.2f} km^3)"))
    # ax.set_xlabel("x (km)")
    # ax.set_ylabel("y (km)")
    # ax.set_zlabel("z (km)")
    # ax.set_title("Alpha Shape of Entire Predicted Electric Reachability Point Cloud")
    # ax.legend()
    # ax.grid()

    # fig,ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 8))
    # points = NoThrust_cloud[:, :, :3].reshape(-1, 3)  # (N*4230, 3)
    # faces, volume = alpha_shape_faces_and_volume(points)
    # ax.add_collection3d(Poly3DCollection(faces, alpha=0.3, facecolor='red', label=f"No Thrust (Pred Volume: {volume:.2f} km^3)"))
    # ax.set_xlabel("x (km)")
    # ax.set_ylabel("y (km)")
    # ax.set_zlabel("z (km)")
    # ax.set_title("Alpha Shape of Entire Predicted No Thrust Reachability Point Cloud")
    # ax.legend()
    # ax.grid()

    if args.animate:
        # animate the change in shape of the point cloud over time for both classes
        n_time = NoThrust_cloud.shape[1]
        n_frames = 100
        step = max(1, n_time // n_frames)
        frame_indices = list(range(0, n_time, step))

        # pre-compute axis limits from all points so axes stay fixed
        all_pts = np.concatenate([
            NoThrust_cloud[:, :, :3].reshape(-1, 3),
            Electric_cloud[:, :, :3].reshape(-1, 3),
        ], axis=0)
        x_lim = (all_pts[:, 0].min(), all_pts[:, 0].max())
        y_lim = (all_pts[:, 1].min(), all_pts[:, 1].max())
        z_lim = (all_pts[:, 2].min(), all_pts[:, 2].max())

        # precompute alpha shapes for each sampled frame
        print(f"Precomputing alpha shapes for {len(frame_indices)} frames...")
        precomputed = []  # list of [(faces, volume), ...] per frame, one entry per class
        clouds = [NoThrust_cloud, Electric_cloud]
        for t in frame_indices:
            frame_data = []
            for cloud in clouds:
                pts = cloud[:, t, :3]
                try:
                    faces, volume = alpha_shape_faces_and_volume(pts)
                    frame_data.append((faces, volume))
                except Exception:
                    frame_data.append((None, 0.0))
            precomputed.append(frame_data)
        print("Precomputation done.")

        fig_anim, ax_anim = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 8))

        def update(frame_idx):
            ax_anim.cla()
            ax_anim.set_xlim(x_lim)
            ax_anim.set_ylim(y_lim)
            ax_anim.set_zlim(z_lim)
            ax_anim.set_xlabel("x (km)")
            ax_anim.set_ylabel("y (km)")
            ax_anim.set_zlabel("z (km)")
            t = frame_indices[frame_idx]
            ax_anim.set_title(f"Reachability Point Cloud (step {t} / {n_time - 1})")
            for i, label in enumerate(CLASS_LABELS):
                faces, volume = precomputed[frame_idx][i]
                if faces is not None:
                    ax_anim.add_collection3d(Poly3DCollection(
                        faces, alpha=0.3, facecolor=colors[i], edgecolor='k',
                        label=f"{label} (Vol: {volume:.2f} km\u00b3)"
                    ))
                else:
                    pts = clouds[i][:, t, :3]
                    ax_anim.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                                    color=colors[i], alpha=0.3, label=label, s=5)
            ax_anim.legend()
            ax_anim.grid()

        anim = FuncAnimation(fig_anim, update, frames=len(frame_indices), interval=100, repeat=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        anim_path = os.path.join(".", f"reachability_animation_{timestamp}.mp4")
        print(f"Saving animation to {anim_path} ...")
        anim.save(anim_path, writer="ffmpeg", fps=10)
        print("Animation saved.")

    # plot average radius of the point cloud from the origin over time for each class
    fig_radius, ax_radius = plt.subplots(figsize=(10, 6))
    time_steps = NoThrust_cloud.shape[1]
    t = np.arange(time_steps) / 1440  # convert to days
    NoThrust_radius = np.linalg.norm(NoThrust_cloud[:, :, :3], axis=2).mean(axis=0)  # (time_steps,)
    Electric_radius = np.linalg.norm(Electric_cloud[:, :, :3], axis=2).mean(axis=0)  # (time_steps,)
    # subtract earths radius to get altitude above surface
    earth_radius = 6371.0  # km
    NoThrust_radius -= earth_radius
    Electric_radius -= earth_radius
    ax_radius.plot(t,NoThrust_radius, label="No Thrust", color='red')
    ax_radius.plot(t,Electric_radius, label="Electric", color='blue')
    ax_radius.set_xlabel("Time (days)")
    ax_radius.set_ylabel("Average Altitude (km)")
    ax_radius.set_title("Average Altitude of Reachability Point Cloud Over Time")
    ax_radius.legend()
    ax_radius.grid()

if __name__ == "__main__":
    args = parse_args()
    if args.webagg:
        plt.switch_backend('webagg')
    main(args)

    plt.show()
