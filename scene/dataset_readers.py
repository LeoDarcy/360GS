#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

(r"""
------------------------------- for Matterport Dataset ------------------------------------------
""")
from scene.layout import Layout
from scene.c2w_slerp import c2w_slerp
(r"""
------------------------------- for Matterport Dataset ------------------------------------------
""")

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    if 'nx' in vertices:
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
        norm =  np.sqrt(np.sum(np.square(normals), -1))
        norm = norm + (norm < 0.000001).astype(normals.dtype) * (1 - norm)
        normals /= norm[..., None]
    else:
        normals = np.zeros_like(positions)
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb, normals : np.ndarray=None):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    if normals is None:
        normals = np.zeros_like(xyz)
    else:
        norm =  np.sqrt(np.sum(np.square(normals), -1))
        norm = norm + (norm < 0.000001).astype(normals.dtype) * (1 - norm)
        normals /= norm[..., None]

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

(r"""
------------------------------- for Matterport Dataset ------------------------------------------
""")

def readCamerasFromPanorama(room_id, iamge_root, cam_root, frame_ids, extenstion, neg_yz, video_ratio=1, highlight_args=None):
    W, H = 1024, 512 # 1024, 512
    cam_infos = []

    if highlight_args is not None:
        l = highlight_args["start"]
        r = highlight_args["end"]
        video_ratio = highlight_args["ratio"]
        frame_ids = frame_ids[l:r + 1]
        frame_ids = [frame_ids[i // video_ratio]  for i in range(len(frame_ids) * video_ratio - (video_ratio - 1))]
        for idx, frame_id in enumerate(frame_ids):
            cam_path = os.path.join(cam_root, f"{room_id}_{frame_id}.json")
            with open(cam_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                c2w = np.array(meta["c2w"])

            mini_idx = idx % video_ratio

            if mini_idx != 0:
                oth_cam_path = os.path.join(cam_root, f"{room_id}_{frame_ids[idx + video_ratio - mini_idx]}.json")
                with open(oth_cam_path, "r", encoding="utf-8") as f:
                    oth_meta = json.load(f)
                    oth_c2w = np.array(oth_meta["c2w"])
                c2w = c2w_slerp(c2w, oth_c2w, 1 / video_ratio * mini_idx)
                
            if neg_yz:
                # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
                c2w[:3, 1:3] *= -1
            
            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)

            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_name = f"{room_id}_{frame_id}{extenstion}"
            image_path = os.path.join(iamge_root, image_name)

            image = Image.open(image_path)
            image = image.resize((W, H))

            # Use any number instead of Fov 
            FovY = 3.14 * 0.8 / 2 # 1.0 * image.height 
            FovX = 3.14 * 1.5 / 2 # 1.0 * image.width

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
    elif video_ratio <= 1:
        for idx, frame_id in enumerate(frame_ids):
            cam_path = os.path.join(cam_root, f"{room_id}_{frame_id}.json")
            with open(cam_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                c2w = np.array(meta["c2w"])

            if neg_yz:
                # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
                c2w[:3, 1:3] *= -1
            
            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)

            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_name = f"{room_id}_{frame_id}{extenstion}"
            image_path = os.path.join(iamge_root, image_name)

            image = Image.open(image_path)
            image = image.resize((W, H))

            # Use any number instead of Fov 
            FovY = 3.14 * 0.8 / 2 # 1.0 * image.height 
            FovX = 3.14 * 1.5 / 2 # 1.0 * image.width

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
    else:
        frame_ids = [frame_ids[i // video_ratio]  for i in range(len(frame_ids) * video_ratio - (video_ratio - 1))]
        for idx, frame_id in enumerate(frame_ids):
            cam_path = os.path.join(cam_root, f"{room_id}_{frame_id}.json")
            with open(cam_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                c2w = np.array(meta["c2w"])

            mini_idx = idx % video_ratio

            if mini_idx != 0:
                oth_cam_path = os.path.join(cam_root, f"{room_id}_{frame_ids[idx + video_ratio - mini_idx]}.json")
                with open(oth_cam_path, "r", encoding="utf-8") as f:
                    oth_meta = json.load(f)
                    oth_c2w = np.array(oth_meta["c2w"])
                c2w = c2w_slerp(c2w, oth_c2w, 1 / video_ratio * mini_idx)
                
            if neg_yz:
                # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
                c2w[:3, 1:3] *= -1
            
            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)

            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_name = f"{room_id}_{frame_id}{extenstion}"
            image_path = os.path.join(iamge_root, image_name)

            image = Image.open(image_path)
            image = image.resize((W, H))

            # Use any number instead of Fov 
            FovY = focal2fov(image.width * 0.2, image.height) # 1.0 * image.height 
            FovX = focal2fov(image.width * 0.2, image.width) # 1.0 * image.width

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
    return cam_infos


def readPanoramaInfo(path, room_id, eval, extension=".jpg", n_inputs=16, **kwargs):
    if extension == ".png":
        cam_label = "geometry_info" # "geometry_info" # "trans_labels_uv_negativez" "trans_labels"
        lay_label = "trans_labels_uv_negativez"
    else:
        cam_label = "trans_labels"
        lay_label = "trans_labels"

    image_root = os.path.join(path, "img")
    cam_root = os.path.join(path, cam_label)

    print(f"Panorama Dataset, img_root: {image_root}, cam_root: {cam_root}")

    frame_ids = sorted(list(filter(lambda file : str(file).startswith(room_id + "_") and str(file).endswith(extension), os.listdir(image_root))))
    frame_ids = [frame_id[len(room_id + "_"):-len(extension)] for frame_id in frame_ids]

    print(f"frame_ids: {frame_ids}")

    cam_infos = readCamerasFromPanorama(room_id,  image_root, cam_root, frame_ids, extension, neg_yz=(extension == ".png"))

    if "video_ratio" in kwargs and kwargs["video_ratio"] > 1:
        frame_ids = sorted(list(filter(lambda file : str(file).startswith(room_id + "_") and str(file).endswith(extension), os.listdir(image_root))), key=lambda x : int(x[len(room_id + "_"):-len(extension)]))
        frame_ids = [frame_id[len(room_id + "_"):-len(extension)] for frame_id in frame_ids]
        cam_infos = readCamerasFromPanorama(room_id,  image_root, cam_root, frame_ids, extension, neg_yz=(extension == ".png"), video_ratio=kwargs["video_ratio"], highlight_args=kwargs["highlight_args"])

        train_cam_infos = cam_infos
        test_cam_infos = [] 
    else:
        if not eval:
            train_cam_infos = cam_infos
            test_cam_infos = []
        else:
            all_indices = np.linspace(0, len(cam_infos) - 1, len(cam_infos)).astype(np.int32).tolist()
            train_indices = np.linspace(0, len(cam_infos) - 1, n_inputs).astype(np.int32).tolist()
            test_indices = [x for x in all_indices if x not in train_indices]
            
            train_cam_infos = [cam_infos[x] for x in train_indices]
            test_cam_infos = [cam_infos[x] for x in test_indices]

            print(f"Train_indices: {train_indices}, Test_indices: {test_indices}")

    #??????Following the Gaussian Splatting, but it's unclear if it works for this dataset
    nerf_normalization = getNerfppNorm(train_cam_infos) 


    ply_path = os.path.join(path, f"points3d_{n_inputs}shots", f"{room_id}.ply")
    os.makedirs(os.path.join(path, f"points3d_{n_inputs}shots"), exist_ok=True)

    if not os.path.exists(ply_path):# not os.path.exists(ply_path):
            # Since this data set has no colmap data, we start with random points
        num_pts = 100_000 # 100_000
        if "use_layout" in kwargs and kwargs["use_layout"]:
            layout_root = os.path.join(path, lay_label)
            with open(os.path.join(layout_root, f"{room_id}_{frame_ids[0]}.json"), "r", encoding="utf-8") as f:
                # with open(os.path.join(layout_root, f"{room_id}_{frame_ids[0]}.json"), "r", encoding="utf-8") as f:
                meta = json.load(f)
                meta["max_corners"] = int(1e9)
                layout = Layout(**meta)
                
                xyz, normals, shs = layout.toPoints(num_pts=num_pts)
                
        
                pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=normals)
        else:
            print(f"Generating random point cloud ({num_pts})...")   
            # We create random points inside the bounds of the panorama scenes
            scene_scale = 2.6 # 2.6
            xyz = np.random.random((num_pts, 3)) * scene_scale - scene_scale / 2
            shs = np.random.random((num_pts, 3)) / 255.0
            pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        storePly(ply_path, xyz, SH2RGB(shs) * 255, pcd.normals)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

(r"""
------------------------------- for Matterport Dataset ------------------------------------------
""")

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "Panorama": readPanoramaInfo 
}
