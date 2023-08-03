import os.path

import numpy as np
import quaternion

transdir = os.path.expanduser("C:/Users/User/Desktop/Test/trans.npy")


def sM2vec(sM):
    return np.array([sM[2][1], sM[0][2], sM[1][0]])


def degradeEtoS(Ematrix):
    sMOMG = Ematrix[:3, :3]
    VEL = Ematrix[:3, 3].reshape(3, 1)
    OMG = sM2vec(sMOMG).reshape(3, 1)
    theta = np.linalg.norm(OMG)
    omg = OMG / theta
    vel = VEL / theta
    screw = np.vstack((omg, vel))
    return (screw * theta).reshape(1, 6)


to_pose = lambda R, t: np.concatenate((
    quaternion.as_float_array(quaternion.from_rotation_matrix(np.reshape(R, (3, 3)))),
    t))

pose_composition = lambda p1, p2: np.concatenate((
    quaternion.as_float_array(np.quaternion(*p1[:4]) * np.quaternion(*p2[:4])),
    p1[4:] + quaternion.as_rotation_matrix(np.quaternion(*p1[:4])) @ p2[4:]))

pose_to_R_t = lambda pose: (quaternion.as_rotation_matrix(np.quaternion(*pose[:4])), pose[4:])

if __name__ == '__main__':
    """
        This file define R t which is original R and t,
        and R_t and t_t which is the trans between rgb camera and event camera
        
        You can generate dataset from unreal engine wi
    """
    R = np.array([0.976781, 0.013642, 0.203538, -0.065466])
    t = np.array([-153.531555, -33.654785, 151.457611]) * 0.01

    i_0 = np.array((.0, .0, .0, 1.)).reshape(1, 4)
    # Trans 1:
    # x, y, z: 0.5 degree + 0.003 shift
    R_t = np.array([[0.9999238, -0.0086882, 0.0087644],
                    [0.0087644, 0.9999238, -0.0086882],
                    [-0.0086882, 0.0087644, 0.9999238]])
    t_t = np.array([0.003, 0.003, 0.003])
    T = np.concatenate((R_t, t_t.reshape(3, 1)), axis=1)
    T = np.concatenate((T, i_0), axis=0)
    se3 = degradeEtoS(T)
    #

    R = quaternion.as_rotation_matrix(np.quaternion(*R))

    x = pose_composition(to_pose(R, t), to_pose(R_t, t_t))
    print(x[:4])
    print(x[4:])
    print(se3)

    np.save(transdir, se3[0])
