import numpy as np
from scipy.spatial.transform import Rotation as R

T = np.array([
    [-1.0000000e+00, -1.2246468e-16, 0.0000000e+00, 4.2500000e-01],
    [ 1.2246468e-16, -1.0000000e+00, 0.0000000e+00, 3.1500000e-01],
    [ 0.0000000e+00,  0.0000000e+00, 1.0000000e+00,-5.0000000e-02],
    [ 0.0000000e+00,  0.0000000e+00, 0.0000000e+00, 1.0000000e+00],
], dtype=np.float32)

# matrix -> 7D [x, y, z, qw, qx, qy, qz]
pos = T[:3, 3]
quat_xyzw = R.from_matrix(T[:3, :3]).as_quat()
qwqxqyqz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
pose7d = np.concatenate([pos, qwqxqyqz])

# save .npy
import os
os.makedirs("/home/xinhai/projects/automoma/assets/object/Refrigerator/10000/grasp", exist_ok=True)
np.save("/home/xinhai/projects/automoma/assets/object/Refrigerator/10000/grasp/0000.npy", pose7d)

print(pose7d)