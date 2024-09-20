import numpy as np
import quaternion

def c2w_slerp(m0, m1, t):
    q0 = quaternion.from_rotation_matrix(m0[:3, :3])
    q1 = quaternion.from_rotation_matrix(m1[:3, :3])

    q = quaternion.slerp(q0, q1, 0, 1, t)

    m = np.eye(4)
    m[:3, :3] = quaternion.as_rotation_matrix(q)

    m[:3, 3] = m0[:3, 3] * (1 - t) + m1[:3, 3] * t

    return m