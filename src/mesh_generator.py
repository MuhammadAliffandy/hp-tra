"""
Parametric 3D Volumetric Mannequin Mesh Synthesizer
Iterasi 2: True 3D Mesh Reconstruction from SMPL 24 Joint Rotations & Shape Beta
"""

import numpy as np

# SMPL 24 Kinematic Tree Parents
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]

# Base template joint offsets (standard anatomical human proportions in meters)
TEMPLATE_OFFSETS = np.array([
    [0.0, 0.0, 0.0],          # 0: Pelvis
    [0.09, -0.08, 0.0],       # 1: L_Hip
    [-0.09, -0.08, 0.0],      # 2: R_Hip
    [0.0, 0.12, 0.0],         # 3: Spine1
    [0.0, -0.38, 0.0],        # 4: L_Knee
    [0.0, -0.38, 0.0],        # 5: R_Knee
    [0.0, 0.14, 0.0],         # 6: Spine2
    [0.0, -0.38, 0.0],        # 7: L_Ankle
    [0.0, -0.38, 0.0],        # 8: R_Ankle
    [0.0, 0.14, 0.0],         # 9: Spine3
    [0.0, -0.08, 0.14],       # 10: L_Foot
    [0.0, -0.08, 0.14],       # 11: R_Foot
    [0.0, 0.12, 0.0],         # 12: Neck
    [0.08, 0.08, -0.02],      # 13: L_Collar
    [-0.08, 0.08, -0.02],     # 14: R_Collar
    [0.0, 0.18, 0.0],         # 15: Head
    [0.10, 0.0, 0.0],         # 16: L_Shoulder
    [-0.10, 0.0, 0.0],        # 17: R_Shoulder
    [0.0, -0.28, 0.0],        # 18: L_Elbow
    [0.0, -0.28, 0.0],        # 19: R_Elbow
    [0.0, -0.25, 0.0],        # 20: L_Wrist
    [0.0, -0.25, 0.0],        # 21: R_Wrist
    [0.0, -0.10, 0.0],        # 22: L_Hand
    [0.0, -0.10, 0.0],        # 23: R_Hand
], dtype=np.float32)

def forward_kinematics(rotmats: np.ndarray, shape_beta: np.ndarray):
    """
    Computes global 3D joint positions using forward kinematics.
    rotmats: [24, 3, 3] rotation matrices
    shape_beta: [10] shape parameters
    Returns:
      joints_3d: [24, 3] global joint coordinates in meters
      global_rotmats: [24, 3, 3] global orientation of each bone
    """
    scale_y = 1.0 + 0.05 * float(shape_beta[0])
    scale_fat = 1.0 + 0.06 * float(shape_beta[1])

    offsets = TEMPLATE_OFFSETS.copy()
    offsets[:, 1] *= scale_y
    offsets[:, 0] *= scale_fat
    offsets[:, 2] *= scale_fat

    joints_3d = np.zeros((24, 3), dtype=np.float32)
    global_rotmats = np.zeros((24, 3, 3), dtype=np.float32)

    for i in range(24):
        p = SMPL_PARENTS[i]
        local_R = rotmats[i]
        if p == -1:
            global_rotmats[i] = local_R
            joints_3d[i] = offsets[i]
        else:
            global_rotmats[i] = np.dot(global_rotmats[p], local_R)
            joints_3d[i] = joints_3d[p] + np.dot(global_rotmats[p], offsets[i])

    return joints_3d, global_rotmats

def create_cylinder_mesh(p1, p2, r1, r2, num_segments=12):
    """
    Creates a 3D tapered cylinder between p1 and p2 with radius r1 at p1 and r2 at p2.
    """
    v = p2 - p1
    length = np.linalg.norm(v)
    if length < 1e-5:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32)

    dir_z = v / length
    arbitrary = np.array([0.0, 1.0, 0.0]) if abs(dir_z[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    dir_x = np.cross(dir_z, arbitrary)
    dir_x /= np.linalg.norm(dir_x)
    dir_y = np.cross(dir_z, dir_x)

    angles = np.linspace(0, 2 * np.pi, num_segments, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    circle_dir = cos_a[:, None] * dir_x + sin_a[:, None] * dir_y
    ring1 = p1 + r1 * circle_dir
    ring2 = p2 + r2 * circle_dir

    vertices = np.vstack([ring1, ring2])
    faces = []
    for i in range(num_segments):
        nxt = (i + 1) % num_segments
        faces.append([i, nxt, num_segments + i])
        faces.append([nxt, num_segments + nxt, num_segments + i])

    return vertices, np.array(faces, dtype=np.int32)

def create_ellipsoid_mesh(center, radii, num_lat=10, num_lon=14):
    """
    Creates a 3D triangulated sphere/ellipsoid mesh.
    """
    rx, ry, rz = radii
    lats = np.linspace(-np.pi / 2, np.pi / 2, num_lat)
    lons = np.linspace(-np.pi, np.pi, num_lon, endpoint=False)

    lat_grid, lon_grid = np.meshgrid(lats, lons)
    x = center[0] + rx * np.cos(lat_grid) * np.cos(lon_grid)
    y = center[1] + ry * np.sin(lat_grid)
    z = center[2] + rz * np.cos(lat_grid) * np.sin(lon_grid)

    vertices = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
    faces = []
    for i in range(num_lon):
        for j in range(num_lat - 1):
            p1 = i * num_lat + j
            p2 = ((i + 1) % num_lon) * num_lat + j
            p3 = p2 + 1
            p4 = p1 + 1
            faces.append([p1, p2, p3])
            faces.append([p1, p3, p4])

    return vertices, np.array(faces, dtype=np.int32)

def build_volumetric_mannequin(joints_3d: np.ndarray, shape_beta: np.ndarray):
    """
    Combines kinematic primitives into a unified 3D mesh parameterized by body shape beta.
    Returns:
      vertices: [V, 3] in meters
      faces: [F, 3] triangle indices (0-indexed)
    """
    fat = max(0.6, 1.0 + 0.05 * float(shape_beta[1]))
    all_verts = []
    all_faces = []
    v_offset = 0

    def add_mesh(v, f):
        nonlocal v_offset
        if len(v) == 0:
            return
        all_verts.append(v)
        all_faces.append(f + v_offset)
        v_offset += len(v)

    # 1. Head
    head_pos = joints_3d[15]
    v_h, f_h = create_ellipsoid_mesh(head_pos, [0.09 * fat, 0.12 * fat, 0.10 * fat])
    add_mesh(v_h, f_h)

    # 2. Neck
    v_n, f_n = create_cylinder_mesh(joints_3d[9], joints_3d[12], 0.05 * fat, 0.045 * fat)
    add_mesh(v_n, f_n)

    # 3. Torso (Ribcage / Chest)
    v_t, f_t = create_cylinder_mesh(joints_3d[0], joints_3d[9], 0.14 * fat, 0.16 * fat, num_segments=16)
    add_mesh(v_t, f_t)

    # 4. Limbs
    limb_pairs = [
        # Left Arm
        (13, 16, 0.05 * fat, 0.05 * fat),
        (16, 18, 0.055 * fat, 0.045 * fat),
        (18, 20, 0.045 * fat, 0.035 * fat),
        (20, 22, 0.035 * fat, 0.02 * fat),
        # Right Arm
        (14, 17, 0.05 * fat, 0.05 * fat),
        (17, 19, 0.055 * fat, 0.045 * fat),
        (19, 21, 0.045 * fat, 0.035 * fat),
        (21, 23, 0.035 * fat, 0.02 * fat),
        # Left Leg
        (0, 1, 0.09 * fat, 0.08 * fat),
        (1, 4, 0.085 * fat, 0.065 * fat),
        (4, 7, 0.065 * fat, 0.045 * fat),
        (7, 10, 0.045 * fat, 0.03 * fat),
        # Right Leg
        (0, 2, 0.09 * fat, 0.08 * fat),
        (2, 5, 0.085 * fat, 0.065 * fat),
        (5, 8, 0.065 * fat, 0.045 * fat),
        (8, 11, 0.045 * fat, 0.03 * fat),
    ]

    for j1, j2, r1, r2 in limb_pairs:
        v_l, f_l = create_cylinder_mesh(joints_3d[j1], joints_3d[j2], r1, r2)
        add_mesh(v_l, f_l)

    mesh_v = np.vstack(all_verts)
    mesh_f = np.vstack(all_faces)
    return mesh_v, mesh_f

def export_to_obj(vertices: np.ndarray, faces: np.ndarray, filepath: str):
    """
    Exports 3D mesh to Wavefront .OBJ format.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("# Neural Gesture Mannequin — 3D Parametric Reconstruction\n")
        f.write("# Author: Muhammad Aliffandy (Apple Developer Academy)\n")
        f.write(f"# Vertices: {len(vertices)} | Faces: {len(faces)}\n")
        for v in vertices:
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
        for tri in faces:
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")
    print(f"💾 Exported 3D mesh to {filepath} ({len(vertices)} verts, {len(faces)} faces)")
