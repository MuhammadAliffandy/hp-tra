"""
2D-Anchored Parametric 3D Volumetric Mannequin Synthesizer
Neural Gesture Mannequin — Iterasi 2

Pipelines:
1. 2D Kinematic Lifting: Lifts 17 high-accuracy 2D keypoints (PCK 84.7%) into 3D metric space without gravity inversion.
2. 3D Volumetric Inflation: Wraps 3D anatomical geometric primitives (tapered cylinders, torso prism, head ellipsoid) around the skeleton parameterized by predicted muscle volume radii.
3. Multi-Angle Orthographic & Perspective Rendering.
4. Wavefront .OBJ 3D file exporter.
"""

from typing import Optional
import numpy as np

def lift_2d_to_3d_joints(coords_2d: np.ndarray, volume_radii: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Lifts 17 normalized 2D joint coordinates [x, y] in [0, 1] to 3D metric coordinates [X, Y, Z] in meters.
    Pelvis is positioned at origin (0, 0, 0).
    Vertical Y is UP (+), horizontal X is RIGHT (+), depth Z is FORWARD (+).
    """
    # 0: Nose, 5: L_Shoulder, 6: R_Shoulder, 7: L_Elbow, 8: R_Elbow, 9: L_Wrist, 10: R_Wrist
    # 11: L_Hip, 12: R_Hip, 13: L_Knee, 14: R_Knee, 15: L_Ankle, 16: R_Ankle
    l_sh = coords_2d[5]
    r_sh = coords_2d[6]
    l_hip = coords_2d[11]
    r_hip = coords_2d[12]

    mid_sh_2d = (l_sh + r_sh) / 2.0
    pelvis_2d = (l_hip + r_hip) / 2.0

    torso_len_2d = float(np.linalg.norm(mid_sh_2d - pelvis_2d))
    torso_len_2d = max(0.08, torso_len_2d)

    # Standard adult torso length (pelvis to mid-shoulder) is ~0.48 meters
    scale = 0.48 / torso_len_2d

    joints_3d = np.zeros((19, 3), dtype=np.float32)

    # 17 standard COCO joints
    for i in range(17):
        pt = coords_2d[i]
        joints_3d[i, 0] = (pt[0] - pelvis_2d[0]) * scale
        joints_3d[i, 1] = (pelvis_2d[1] - pt[1]) * scale  # Invert Y so up is +

    # Derived joints
    joints_3d[17, 0] = (mid_sh_2d[0] - pelvis_2d[0]) * scale
    joints_3d[17, 1] = (pelvis_2d[1] - mid_sh_2d[1]) * scale  # 17: Mid-Shoulder
    joints_3d[18] = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # 18: Pelvis (Root)

    # Solve Depth (Z) via Kinematic Foreshortening
    joints_3d[18, 2] = 0.0
    joints_3d[17, 2] = 0.0
    joints_3d[5, 2] = 0.0
    joints_3d[6, 2] = 0.0
    joints_3d[11, 2] = 0.0
    joints_3d[12, 2] = 0.0
    joints_3d[0, 2] = 0.03  # Nose slightly in front

    def solve_depth(parent_idx, child_idx, std_len, forward_bias=0.0):
        p_xy = joints_3d[parent_idx, :2]
        c_xy = joints_3d[child_idx, :2]
        d_2d = float(np.linalg.norm(p_xy - c_xy))
        dz_sq = max(0.0, std_len**2 - d_2d**2)
        dz = np.sqrt(dz_sq)
        if forward_bias >= 0:
            joints_3d[child_idx, 2] = joints_3d[parent_idx, 2] + dz
        else:
            joints_3d[child_idx, 2] = joints_3d[parent_idx, 2] - dz

    # Legs (biomechanically slant downward and slightly backward)
    solve_depth(11, 13, 0.38, forward_bias=0.0)    # L_Thigh
    solve_depth(13, 15, 0.36, forward_bias=-0.02)  # L_Calf
    solve_depth(12, 14, 0.38, forward_bias=0.0)    # R_Thigh
    solve_depth(14, 16, 0.36, forward_bias=-0.02)  # R_Calf

    # Arms (elbows / wrists can articulate forward)
    l_arm_forward = 1.0 if joints_3d[9, 1] > joints_3d[7, 1] else 0.0
    r_arm_forward = 1.0 if joints_3d[10, 1] > joints_3d[8, 1] else 0.0

    solve_depth(5, 7, 0.28, forward_bias=l_arm_forward)
    solve_depth(7, 9, 0.25, forward_bias=l_arm_forward)
    solve_depth(6, 8, 0.28, forward_bias=r_arm_forward)
    solve_depth(8, 10, 0.25, forward_bias=r_arm_forward)

    return joints_3d

def create_cylinder_mesh(p1, p2, r1, r2, num_segments=12):
    """Creates a 3D tapered cylinder between p1 and p2."""
    v = p2 - p1
    length = float(np.linalg.norm(v))
    if length < 1e-5:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32)

    dir_z = v / length
    arbitrary = np.array([0.0, 1.0, 0.0]) if abs(dir_z[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    dir_x = np.cross(dir_z, arbitrary)
    dir_x /= np.linalg.norm(dir_x)
    dir_y = np.cross(dir_z, dir_x)

    angles = np.linspace(0, 2 * np.pi, num_segments, endpoint=False)
    circle_dir = np.cos(angles)[:, None] * dir_x + np.sin(angles)[:, None] * dir_y
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
    """Creates a 3D triangulated sphere/ellipsoid mesh."""
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

def build_3d_mannequin_from_joints(joints_3d: np.ndarray, volume_radii: Optional[np.ndarray] = None):
    """
    Constructs a complete 3D volumetric humanoid mannequin mesh from 3D joints and volume radii.
    Returns:
      vertices: [V, 3] in meters
      faces: [F, 3] triangle indices (0-indexed)
    """
    if volume_radii is None:
        volume_radii = np.zeros(6, dtype=np.float32)

    # Radii: [upper_arm, lower_arm, thigh, calf, neck, torso]
    r_uarm  = 0.045 * (1.0 + 0.3 * float(volume_radii[0]))
    r_larm  = 0.038 * (1.0 + 0.3 * float(volume_radii[1]))
    r_thigh = 0.075 * (1.0 + 0.3 * float(volume_radii[2]))
    r_calf  = 0.055 * (1.0 + 0.3 * float(volume_radii[3]))
    r_neck  = 0.040 * (1.0 + 0.3 * float(volume_radii[4]))
    r_torso = 0.125 * (1.0 + 0.3 * float(volume_radii[5]))

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

    # 1. Head (Ellipsoid)
    head_center = (joints_3d[0] + (joints_3d[3] + joints_3d[4]) / 2.0) / 2.0
    v_h, f_h = create_ellipsoid_mesh(head_center, [0.08, 0.11, 0.09], num_lat=10, num_lon=14)
    add_mesh(v_h, f_h)

    # 2. Neck
    neck_top = head_center - np.array([0.0, 0.08, 0.0])
    v_n, f_n = create_cylinder_mesh(joints_3d[17], neck_top, r_neck, r_neck * 0.9)
    add_mesh(v_n, f_n)

    # 3. Torso (Pelvis to Mid-Shoulder)
    v_t, f_t = create_cylinder_mesh(joints_3d[18], joints_3d[17], r_torso * 0.9, r_torso * 1.15, num_segments=16)
    add_mesh(v_t, f_t)

    # 4. Shoulder Clavicle Connectors
    v_sc_l, f_sc_l = create_cylinder_mesh(joints_3d[17], joints_3d[5], r_uarm, r_uarm)
    v_sc_r, f_sc_r = create_cylinder_mesh(joints_3d[17], joints_3d[6], r_uarm, r_uarm)
    add_mesh(v_sc_l, f_sc_l)
    add_mesh(v_sc_r, f_sc_r)

    # 5. Pelvis Hip Connectors
    v_p_l, f_p_l = create_cylinder_mesh(joints_3d[18], joints_3d[11], r_thigh * 0.9, r_thigh)
    v_p_r, f_p_r = create_cylinder_mesh(joints_3d[18], joints_3d[12], r_thigh * 0.9, r_thigh)
    add_mesh(v_p_l, f_p_l)
    add_mesh(v_p_r, f_p_r)

    # 6. Limbs
    limbs = [
        # Left Arm
        (joints_3d[5], joints_3d[7], r_uarm, r_uarm * 0.85),  # Upper Arm
        (joints_3d[7], joints_3d[9], r_uarm * 0.85, r_larm),   # Forearm
        # Right Arm
        (joints_3d[6], joints_3d[8], r_uarm, r_uarm * 0.85),
        (joints_3d[8], joints_3d[10], r_uarm * 0.85, r_larm),
        # Left Leg
        (joints_3d[11], joints_3d[13], r_thigh, r_thigh * 0.8), # Thigh
        (joints_3d[13], joints_3d[15], r_thigh * 0.8, r_calf),  # Calf
        # Right Leg
        (joints_3d[12], joints_3d[14], r_thigh, r_thigh * 0.8),
        (joints_3d[14], joints_3d[16], r_thigh * 0.8, r_calf),
    ]

    for p1, p2, r1, r2 in limbs:
        v_l, f_l = create_cylinder_mesh(p1, p2, r1, r2)
        add_mesh(v_l, f_l)

    # 7. Hands & Feet End Effectors (Ellipsoids)
    v_hl, f_hl = create_ellipsoid_mesh(joints_3d[9], [0.035, 0.045, 0.025], num_lat=6, num_lon=8)
    v_hr, f_hr = create_ellipsoid_mesh(joints_3d[10], [0.035, 0.045, 0.025], num_lat=6, num_lon=8)
    add_mesh(v_hl, f_hl)
    add_mesh(v_hr, f_hr)

    foot_l = joints_3d[15] + np.array([0.0, -0.02, 0.06])
    foot_r = joints_3d[16] + np.array([0.0, -0.02, 0.06])
    v_fl, f_fl = create_ellipsoid_mesh(foot_l, [0.04, 0.03, 0.08], num_lat=6, num_lon=8)
    v_fr, f_fr = create_ellipsoid_mesh(foot_r, [0.04, 0.03, 0.08], num_lat=6, num_lon=8)
    add_mesh(v_fl, f_fl)
    add_mesh(v_fr, f_fr)

    return np.vstack(all_verts), np.vstack(all_faces)

def export_to_obj(vertices: np.ndarray, faces: np.ndarray, filepath: str):
    """Exports 3D mesh to standard Wavefront .OBJ format."""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("# Neural Gesture Mannequin — 2D-Anchored 3D Volumetric Reconstruction\n")
        f.write("# Author: Muhammad Aliffandy (Apple Developer Academy)\n")
        f.write(f"# Vertices: {len(vertices)} | Faces: {len(faces)}\n")
        for v in vertices:
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
        for tri in faces:
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")
    print(f"💾 Exported 3D mesh to {filepath} ({len(vertices)} verts, {len(faces)} faces)")

# ==============================================================================
# Backward-Compatibility Exports & Aliases
# ==============================================================================
build_volumetric_mannequin = build_3d_mannequin_from_joints

SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]

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

