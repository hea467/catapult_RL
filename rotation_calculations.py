import numpy as np
from scipy.spatial.transform import Rotation as R

def transform_pts_wrt_com(points, init_pose, goal_pose, com):
    rot0 = R.from_quat(init_pose[3:])
    x1, y1, z1, rot1 = goal_pose[0], goal_pose[1], goal_pose[2], R.from_quat(goal_pose[3:])
    rotation_diff = rot1 * rot0.inv()
    rotation_matrix = rotation_diff.as_matrix()[:3, :3]
    rotated_points = np.dot(points - com, rotation_matrix.T) + np.array([x1, y1, z1])
    return rotated_points

points = np.array([[0.0, -0.005, 0.075], [0.06, 0.02, 0.105], [0.08, -0.04,0.105],
                   [0.06, -0.1, 0.105], [0.0, -0.08, 0.075], [-0.06,0.02,0.105], 
                   [-0.08,-0.04, 0.105], [-0.06,-0.1,0.105], [0.0, -0.075, 0.276], 
                    ])
com = np.mean(points, axis=0)

goal_rot = R.from_euler('x', 30*np.pi/180, degrees=False).as_quat()
rotated_points = transform_pts_wrt_com(points, [*com, *R.from_euler('xyz', (0,0,0)).as_quat()], [*com, *goal_rot], com)
print(rotated_points)
for p in rotated_points:
    print(f"{round(p[0], 5)} {round(p[1], 5)} {round(p[2], 5)}")