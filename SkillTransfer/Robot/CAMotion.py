import math

import numpy as np
import quaternion
import scipy.spatial.transform as scitransform
from Filter.Filter import MotionFilter
from scipy.spatial.transform import Rotation as R

"""
##### IMPORTANT #####
If you are using average rotations of three or more people, please cite the following paper
see also: https://github.com/UCL/scikit-surgerycore

Thompson S, Dowrick T, Ahmad M, et al.
SciKit-Surgery: compact libraries for surgical navigation.
International Journal of Computer Assisted Radiology and Surgery. May 2020.
DOI: 10.1007/s11548-020-02180-5
"""
import sksurgerycore.algorithms.averagequaternions as aveq


class CAMotion:
    originPositions = {}
    inversedMatrixforPosition = {}
    inversedMatrix = {}

    beforePositions = {}
    weightedPositions = {}

    beforeRotations = {}
    weightedRotations = {}

    beforeRotateAngles =  {}
    weightedRotateAngles = {}

    def __init__(self, defaultParticipantNum: int, otherRigidBodyNum: int) -> None:
        for i in range(defaultParticipantNum):
            self.originPositions["participant" + str(i + 1)] = np.zeros(3)
            self.inversedMatrixforPosition["participant" + str(i + 1)] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            self.inversedMatrix["participant" + str(i + 1)] = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

            self.beforePositions["participant" + str(i + 1)] = np.zeros(3)
            self.weightedPositions["participant" + str(i + 1)] = np.zeros(3)

            self.beforeRotations["participant" + str(i + 1)] = np.array([0, 0, 0, 1])
            self.weightedRotations["participant" + str(i + 1)] = np.array([0, 0, 0, 1])

        for i in range(otherRigidBodyNum):
            self.originPositions["otherRigidBody" + str(i + 1)] = np.zeros(3)
            self.inversedMatrix["otherRigidBody" + str(i + 1)] = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            self.beforeRotateAngles["otherRigidBody" + str(i + 1)] = 0
            self.weightedRotateAngles["otherRigidBody" + str(i + 1)] = 0

        self.participantNum = defaultParticipantNum
        self.otherRigidBodyNum = otherRigidBodyNum
        self.before_position = [[0, 0, 0], [0, 0, 0]]
        self.customweightPosition = [0, 0, 0]
        self.before_sharedPosition = [0, 0, 0]

        n = 2
        fp = 10
        fs = 700
        self.filter_FB = MotionFilter()
        self.filter_FB.InitLowPassFilterWithOrder(fs, fp, n)
        self.get_pos_1_box = [[0]] * n
        self.get_pos_1_filt_box = [[0]] * n
        self.get_pos_2_box = [[0]] * n
        self.get_pos_2_filt_box = [[0]] * n
        self.get_rot_1_box = [[0]] * n
        self.get_rot_1_filt_box = [[0]] * n
        self.get_rot_2_box = [[0]] * n
        self.get_rot_2_filt_box = [[0]] * n

    def participant2robot(self, position: dict, rotation: dict, weight: list):
        # ----- numpy array to dict: position ----- #
        if type(position) is np.ndarray:
            position = self.NumpyArray2Dict(position)

        # ----- numpy array to dict: rotation ----- #
        if type(rotation) is np.ndarray:
            rotation = self.NumpyArray2Dict(rotation)

        # ----- Shared transform ----- #
        sharedPosition_left = [0, 0, 0]
        sharedPosition_right = [0, 0, 0]

        sharedRotation_euler_left = [0, 0, 0]
        sharedRotation_euler_right = [0, 0, 0]

        for i in range(self.participantNum):
            # ----- Position ----- #
            diffPos = position["participant" + str(i + 1)] - self.beforePositions["participant" + str(i + 1)]
            weightedPos = diffPos * weight[0][i] + self.weightedPositions["participant" + str(i + 1)]

            if i % 2 == 0:
                sharedPosition_left += weightedPos

            elif i % 2 == 1:
                sharedPosition_right += weightedPos

            self.weightedPositions["participant" + str(i + 1)] = weightedPos
            self.beforePositions["participant" + str(i + 1)] = position["participant" + str(i + 1)]

            # ----- Rotation ----- #
            qw, qx, qy, qz = self.beforeRotations["participant" + str(i + 1)][3], self.beforeRotations["participant" + str(i + 1)][0], self.beforeRotations["participant" + str(i + 1)][1], self.beforeRotations["participant" + str(i + 1)][2]
            mat4x4 = np.array([
                    [qw, qz, -qy, qx],
                    [-qz, qw, qx, qy],
                    [qy, -qx, qw, qz],
                    [-qx, -qy, -qz, qw]])
            currentRot = rotation["participant" + str(i + 1)]
            diffRot = np.dot(np.linalg.inv(mat4x4), currentRot)
            diffRotEuler = self.Quaternion2Euler(np.array(diffRot))

            weightedDiffRotEuler = list(map(lambda x: x * weight[1][i], diffRotEuler))
            weightedDiffRot = self.Euler2Quaternion(np.array(weightedDiffRotEuler))

            nqw, nqx, nqy, nqz = weightedDiffRot[3], weightedDiffRot[0], weightedDiffRot[1], weightedDiffRot[2]
            neomat4x4 = np.array([ [nqw, -nqz, nqy, nqx],
                                   [nqz, nqw, -nqx, nqy],
                                   [-nqy, nqx, nqw, nqz],
                                   [-nqx, -nqy, -nqz, nqw]])
            weightedRot = np.dot(neomat4x4, self.weightedRotations["participant" + str(i + 1)])

            if i % 2 == 0:
                sharedRotation_euler_left += self.Quaternion2Euler(weightedRot)

            elif i % 2 == 1:
                sharedRotation_euler_right += self.Quaternion2Euler(weightedRot)

            self.weightedRotations["participant" + str(i + 1)] = weightedRot
            self.beforeRotations["participant" + str(i + 1)] = rotation["participant" + str(i + 1)]

        self.posarm = dict(robot1=sharedPosition_left, robot2=sharedPosition_right)
        self.rotarm = dict(robot1=sharedRotation_euler_left, robot2=sharedRotation_euler_right)

        return self.posarm, self.rotarm

    def participant2robot_new(self, position: dict, rotation: dict, weight: list):
        order = [2, 1, 0, 3]
        keys_list = list(rotation.keys())

        # ----- change cordinate from motive to xArm (only rotation) ----- #
        for i in range(self.participantNum):
            key = keys_list[i]
            rotation[key] = [rotation[key][j] for j in order]

            if i % 2 == 0:
                rotation[key][1] = -1 * rotation[key][1]
            elif i % 2 == 1:
                rotation[key][2] = -1 * rotation[key][2]


        # ----- numpy array to dict: position ----- #
        if type(position) is np.ndarray:
            position = self.NumpyArray2Dict(position)

        # ----- numpy array to dict: rotation ----- #
        if type(rotation) is np.ndarray:
            rotation = self.NumpyArray2Dict(rotation)

        # ----- Shared transform ----- #
        sharedPosition_left = [0, 0, 0]
        sharedPosition_right = [0, 0, 0]

        sharedRotation_euler_left = [0, 0, 0]
        sharedRotation_euler_right = [0, 0, 0]

        for i in range(self.participantNum):
            # ----- Position ----- #
            diffPos = position["participant" + str(i + 1)] - self.beforePositions["participant" + str(i + 1)]
            weightedPos = diffPos * weight[0][i] + self.weightedPositions["participant" + str(i + 1)]

            if i % 2 == 0:
                sharedPosition_left += weightedPos

            elif i % 2 == 1:
                sharedPosition_right += weightedPos

            self.weightedPositions["participant" + str(i + 1)] = weightedPos
            self.beforePositions["participant" + str(i + 1)] = position["participant" + str(i + 1)]

            # ----- Rotation ----- #
            qw, qx, qy, qz = self.beforeRotations["participant" + str(i + 1)][3], self.beforeRotations["participant" + str(i + 1)][0], self.beforeRotations["participant" + str(i + 1)][1], self.beforeRotations["participant" + str(i + 1)][2]
            mat4x4 = np.array([
                    [qw, qz, -qy, qx],
                    [-qz, qw, qx, qy],
                    [qy, -qx, qw, qz],
                    [-qx, -qy, -qz, qw]])
            currentRot = rotation["participant" + str(i + 1)]
            diffRot = np.dot(np.linalg.inv(mat4x4), currentRot)
            diffRotEuler = self.Quaternion2Euler(np.array(diffRot))

            weightedDiffRotEuler = list(map(lambda x: x * weight[1][i], diffRotEuler))
            weightedDiffRot = self.Euler2Quaternion(np.array(weightedDiffRotEuler))

            nqw, nqx, nqy, nqz = weightedDiffRot[3], weightedDiffRot[0], weightedDiffRot[1], weightedDiffRot[2]
            neomat4x4 = np.array([ [nqw, -nqz, nqy, nqx],
                                   [nqz, nqw, -nqx, nqy],
                                   [-nqy, nqx, nqw, nqz],
                                   [-nqx, -nqy, -nqz, nqw]])
            weightedRot = np.dot(neomat4x4, self.weightedRotations["participant" + str(i + 1)])

            if i % 2 == 0:
                sharedRotation_euler_left += self.Quaternion2Euler(weightedRot)

            elif i % 2 == 1:
                sharedRotation_euler_right += self.Quaternion2Euler(weightedRot)

            self.weightedRotations["participant" + str(i + 1)] = weightedRot
            self.beforeRotations["participant" + str(i + 1)] = rotation["participant" + str(i + 1)]

        self.posarm = dict(robot1=sharedPosition_left, robot2=sharedPosition_right)
        self.rotarm = dict(robot1=sharedRotation_euler_left, robot2=sharedRotation_euler_right)

        return self.posarm, self.rotarm

    def participant2robot_new2(self, position: dict, rotation: dict, weight: list):
        # ----- モーションキャプチャ座標系からxArm座標系への変換（回転のみ） ----- #
        order = [2, 1, 0, 3]
        keys_list = list(rotation.keys())
        for i in range(self.participantNum):
            key = keys_list[i]
            rotation[key] = [rotation[key][j] for j in order]

            if i % 2 == 0:
                rotation[key][1] = -1 * rotation[key][1]
            elif i % 2 == 1:
                rotation[key][2] = -1 * rotation[key][2]

        # ----- 辞書に変換 ----- #
        if type(position) is np.ndarray:
            position = self.NumpyArray2Dict(position)
        if type(rotation) is np.ndarray:
            rotation = self.NumpyArray2Dict(rotation)

        # ----- 共有変換の初期化 ----- #
        sharedPosition_left = [0, 0, 0]
        sharedPosition_right = [0, 0, 0]

        sharedRotation_euler_left = [0, 0, 0]
        sharedRotation_euler_right = [0, 0, 0]

        sharedRotate_angle_left = 0
        sharedRotate_angle_right = 0

        for i in range(self.participantNum):
            # ----- position ----- #
            diffPos = position["participant" + str(i + 1)] - self.beforePositions["participant" + str(i + 1)]
            weightedPos = diffPos * weight[0][i] + self.weightedPositions["participant" + str(i + 1)]

            if i % 2 == 0:
                sharedPosition_left += weightedPos
            elif i % 2 == 1:
                sharedPosition_right += weightedPos

            self.weightedPositions["participant" + str(i + 1)] = weightedPos
            self.beforePositions["participant" + str(i + 1)] = position["participant" + str(i + 1)]

            # ----- rotation ----- #
            qw, qx, qy, qz = self.beforeRotations["participant" + str(i + 1)][3], self.beforeRotations["participant" + str(i + 1)][0], self.beforeRotations["participant" + str(i + 1)][1], self.beforeRotations["participant" + str(i + 1)][2]
            mat4x4 = np.array([
                    [qw, qz, -qy, qx],
                    [-qz, qw, qx, qy],
                    [qy, -qx, qw, qz],
                    [-qx, -qy, -qz, qw]])
            currentRot = rotation["participant" + str(i + 1)]
            diffRot = np.dot(np.linalg.inv(mat4x4), currentRot)
            diffRotEuler = self.Quaternion2Euler(np.array(diffRot))

            weightedDiffRotEuler = list(map(lambda x: x * weight[1][i], diffRotEuler))
            weightedDiffRot = self.Euler2Quaternion(np.array(weightedDiffRotEuler))

            nqw, nqx, nqy, nqz = weightedDiffRot[3], weightedDiffRot[0], weightedDiffRot[1], weightedDiffRot[2]
            neomat4x4 = np.array([ [nqw, -nqz, nqy, nqx],
                                [nqz, nqw, -nqx, nqy],
                                [-nqy, nqx, nqw, nqz],
                                [-nqx, -nqy, -nqz, nqw]])
            weightedRot = np.dot(neomat4x4, self.weightedRotations["participant" + str(i + 1)])

            if i % 2 == 0:
                sharedRotation_euler_left += self.Quaternion2Euler(weightedRot)
            elif i % 2 == 1:
                sharedRotation_euler_right += self.Quaternion2Euler(weightedRot)

            self.weightedRotations["participant" + str(i + 1)] = weightedRot
            self.beforeRotations["participant" + str(i + 1)] = rotation["participant" + str(i + 1)]

            # ----- rotate ----- #
            beforeRotateAngle = self.beforeRotateAngles["otherRigidBody" + str(i + 1)]
            currentRotateAngle = self.calculate_rotate_angle(currentRot["participant" + str(i + 1)], rotation['otherRigidBody' + str(i + 1)])
            diffRotateAngle = currentRotateAngle - beforeRotateAngle
            weightedDiffRotateAngle = list(map(lambda x: x * weight[1][i], diffRotateAngle))
            weightedRotateAngle = self.weightedRotateAngles["otherRigidBody" + str(i + 1)] + weightedDiffRotateAngle

            if i % 2 == 0:
                sharedRotate_angle_left += weightedRotateAngle
            elif i % 2 == 1:
                sharedRotate_angle_right += weightedRotateAngle

            self.weightedRotateAngles["otherRigidBody" + str(i + 1)] = weightedRotateAngle
            self.beforeRotateAngles["otherRigidBody" + str(i + 1)] = currentRotateAngle

            if abs(sharedRotate_angle_left) < 90:
                sharedRotation_euler_left = self.add_rotation(sharedRotation_euler_left, sharedRotate_angle_left)
            else:
                pass

            if abs(sharedRotate_angle_right) < 90:
                sharedRotation_euler_right = self.add_rotation(sharedRotation_euler_right, sharedRotate_angle_right)
            else:
                pass

        self.posarm = dict(robot1=sharedPosition_left, robot2=sharedPosition_right)
        self.rotarm = dict(robot1=sharedRotation_euler_left, robot2=sharedRotation_euler_right)

        return self.posarm, self.rotarm
    
    def calculate_rotate_angle(self, quat1, quat2):
        """
        Calculate the rotation angle difference in degrees between two quaternions, including direction (sign),
        assuming rotation around the local z-axis.
        
        Parameters:
        quat1, quat2 (list or np.array): Two 4-dimensional vectors representing the quaternions.
        
        Returns:
        float: The rotation angle difference in degrees, with sign indicating direction.
        """
        # Normalize the quaternions
        quat1 = quat1 / np.linalg.norm(quat1)
        quat2 = quat2 / np.linalg.norm(quat2)
        
        # Convert quaternions to scipy Rotation objects
        rot1 = R.from_quat(quat1)
        rot2 = R.from_quat(quat2)
        
        # Calculate the relative rotation
        relative_rotation = rot2 * rot1.inv()
        
        # Extract the rotation angle in radians
        angle_rad = 2 * np.arccos(np.clip(relative_rotation.as_quat()[-1], -1.0, 1.0))
        
        # Get the rotation vector
        rotvec = relative_rotation.as_rotvec()
        
        # Determine the sign of the rotation angle based on the z-component of the rotation vector
        angle_deg = np.degrees(angle_rad)
        if rotvec[2] > 0:  # Assuming rotation around the local z-axis
            angle_deg = -angle_deg
        
        return angle_deg

    # ----------------------------------------------------------------------------------------------------------------------------------------------

    def SetOriginPosition(self, position) -> None:
        """
        Set the origin position

        Parameters
        ----------
        position: dict, numpy array
            Origin position
        """
        # ----- numpy array to dict: position ----- #
        if type(position) is np.ndarray:
            position = self.NumpyArray2Dict(position)

        for i in range(self.participantNum):
            self.originPositions["participant" + str(i + 1)] = position["participant" + str(i + 1)]

        for i in range(self.otherRigidBodyNum):
            self.originPositions["otherRigidBody" + str(i + 1)] = position["otherRigidBody" + str(i + 1)]

    def GetRelativePosition(self, position) -> None:
        """
        Get the relative position

        Parameters
        ----------
        position: dict, numpy array
            Position to compare with the origin position.
            [x, y, z]

        Returns
        ----------
        relativePos: dict
            Position relative to the origin position.
            [x, y, z]
        """

        # ----- numpy array to dict: position ----- #
        if type(position) is np.ndarray:
            position = self.NumpyArray2Dict(position)

        relativePos = {}
        for i in range(self.participantNum):
            relativePos["participant" + str(i + 1)] = position["participant" + str(i + 1)] - self.originPositions["participant" + str(i + 1)]

        for i in range(self.otherRigidBodyNum):
            relativePos["otherRigidBody" + str(i + 1)] = position["otherRigidBody" + str(i + 1)] - self.originPositions["otherRigidBody" + str(i + 1)]

        return relativePos

    def SetInversedMatrix(self, rotation) -> None:
        """
        Set the inversed matrix

        Parameters
        ----------
        rotation: dict, numpy array
            Quaternion.
            Rotation for inverse matrix calculation
        """

        # ----- numpy array to dict: rotation ----- #
        if type(rotation) is np.ndarray:
            rotation = self.NumpyArray2Dict(rotation)

        for i in range(self.participantNum):
            q = rotation["participant" + str(i + 1)]
            qw, qx, qy, qz = q[3], q[1], q[2], q[0]
            mat4x4 = np.array([ [qw, -qy, qx, qz],
                                [qy, qw, -qz, qx],
                                [-qx, qz, qw, qy],
                                [-qz, -qx, -qy, qw]])
            self.inversedMatrix["participant" + str(i + 1)] = np.linalg.inv(mat4x4)

        for i in range(self.otherRigidBodyNum):
            q = rotation["otherRigidBody" + str(i + 1)]
            qw, qx, qy, qz = q[3], q[1], q[2], q[0]
            mat4x4 = np.array([ [qw, -qy, qx, qz],
                                [qy, qw, -qz, qx],
                                [-qx, qz, qw, qy],
                                [-qz, -qx, -qy, qw]])
            self.inversedMatrix["otherRigidBody" + str(i + 1)] = np.linalg.inv(mat4x4)


    def GetRelativePosition_r(self, position):
        """
        Get the relative position
        Parameters
        ----------
        position: dict, numpy array
            Position to compare with the origin position.
            [x, y, z]

        Returns
        ----------
        relativePos: dict
            Position relative to the origin position.
            [x, y, z]
        """

        # ----- numpy array to dict: position ----- #
        if type(position) is np.ndarray:
            position = self.NumpyArray2Dict(position)

        relativePos = {}
        for i in range(self.participantNum):
            relativePos["participant" + str(i + 1)] = np.dot(self.inversedMatrixforPosition["participant" + str(i + 1)], position["participant" + str(i + 1)] - self.originPositions["participant" + str(i + 1)])
        relativePos["endEffector"] = np.dot(self.inversedMatrixforPosition["endEffector"], position["endEffector"] - self.originPositions["endEffector"])

        return relativePos

    def GetRelativeRotation(self, rotation):
        """
        Get the relative rotation

        Parameters
        ----------
        rotation: dict, numpy array
            Rotation to compare with the origin rotation.
            [x, y, z, w]

        Returns
        ----------
        relativeRot: dict
            Rotation relative to the origin rotation.
            [x, y, z, w]
        """

        # ----- numpy array to dict: rotation ----- #
        if type(rotation) is np.ndarray:
            rotation = self.NumpyArray2Dict(rotation)

        relativeRot = {}
        for i in range(self.participantNum):
            relativeRot["participant" + str(i + 1)] = np.dot(self.inversedMatrix["participant" + str(i + 1)], rotation["participant" + str(i + 1)])

        for i in range(self.otherRigidBodyNum):
            relativeRot["otherRigidBody" + str(i + 1)] = np.dot(self.inversedMatrix["otherRigidBody" + str(i + 1)], rotation["otherRigidBody" + str(i + 1)])

        return relativeRot

    def GetRelativeRotation_r(self, rotation):
        """
        Get the relative rotation

        Parameters
        ----------
        rotation: dict, numpy array
            Rotation to compare with the origin rotation.
            [x, y, z, w]

        Returns
        ----------
        relativeRot: dict
            Rotation relative to the origin rotation.
            [x, y, z, w]
        """

        # ----- numpy array to dict: rotation ----- #
        if type(rotation) is np.ndarray:
            rotation = self.NumpyArray2Dict(rotation)

        relativeRot = {}
        for i in range(self.participantNum):
            relativeRot["participant" + str(i + 1)] = np.dot(
                self.inversedMatrix["participant" + str(i + 1)],
                rotation["participant" + str(i + 1)],
            )
        relativeRot["endEffector"] = np.dot(
            self.inversedMatrix["endEffector"], rotation["endEffector"]
        )

        return relativeRot

    # def Quaternion2Euler(self, q, isDeg: bool = True):
    #     """
    #     クォータニオンからオイラー角を計算します。

    #     回転行列
    #     |m00 m01 m02 0|
    #     |m10 m11 m12 0|
    #     |m20 m21 m22 0|
    #     | 0   0   0  1|

    #     Parameters
    #     ----------
    #     q: np.ndarray
    #         クォータニオン。
    #         [x, y, z, w]
    #     isDeg: (Optional) bool
    #         このフラグがTrueの場合、返される角度は度単位になります。デフォルトはTrueです。

    #     Returns
    #     ----------
    #     rotEuler: np.ndarray
    #         オイラー角。
    #         [x, y, z]
    #     """
    #     qx = q[0]
    #     qy = q[1]
    #     qz = q[2]
    #     qw = q[3]

    #     # 回転行列の要素を計算
    #     m00 = 1 - 2 * (qy**2 + qz**2)
    #     m01 = 2 * (qx * qy + qw * qz)
    #     m02 = 2 * (qx * qz - qw * qy)
    #     m10 = 2 * (qx * qy - qw * qz)
    #     m11 = 1 - 2 * (qx**2 + qz**2)
    #     m12 = 2 * (qy * qz + qw * qx)
    #     m20 = 2 * (qx * qz + qw * qy)
    #     m21 = 2 * (qy * qz - qw * qx)
    #     m22 = 1 - 2 * (qx**2 + qy**2)

    #     # ジンバルロック条件を確認
    #     if m02 == 1:
    #         tx = math.atan2(m10, m11)
    #         ty = math.pi / 2
    #         tz = 0
    #     elif m02 == -1:
    #         tx = math.atan2(m21, m20)
    #         ty = -math.pi / 2
    #         tz = 0
    #     else:
    #         tx = math.atan2(m12, m22)
    #         ty = math.asin(m02)
    #         tz = math.atan2(m01, m00)

    #     if isDeg:
    #         tx = np.rad2deg(tx)
    #         ty = np.rad2deg(ty)
    #         tz = np.rad2deg(tz)

    #     rotEuler = np.array([tx, ty, tz])
    #     return rotEuler


    def Quaternion2Euler(self, q, isDeg: bool = True):
        """
        Calculate the Euler angle from the Quaternion.


        Rotation matrix
        |m00 m01 m02 0|
        |m10 m11 m12 0|
        |m20 m21 m22 0|
        | 0   0   0  1|

        Parameters
        ----------
        q: np.ndarray
            Quaternion.
            [x, y, z, w]
        isDeg: (Optional) bool
            Returned angles are in degrees if this flag is True, else they are in radians.
            The default is True.

        Returns
        ----------
        rotEuler: np.ndarray
            Euler angle.
            [x, y, z]
        """

        qx = q[0]
        qy = q[1]
        qz = q[2]
        qw = q[3]

        # 1 - 2y^2 - 2z^2
        m00 = 1 - (2 * qy**2) - (2 * qz**2)
        # 2xy + 2wz
        m01 = (2 * qx * qy) + (2 * qw * qz)
        # 2xz - 2wy
        m02 = (2 * qx * qz) - (2 * qw * qy)
        # 2xy - 2wz
        m10 = (2 * qx * qy) - (2 * qw * qz)
        # 1 - 2x^2 - 2z^2
        m11 = 1 - (2 * qx**2) - (2 * qz**2)
        # 2yz + 2wx
        m12 = (2 * qy * qz) + (2 * qw * qx)
        # 2xz + 2wy
        m20 = (2 * qx * qz) + (2 * qw * qy)
        # 2yz - 2wx
        m21 = (2 * qy * qz) - (2 * qw * qx)
        # 1 - 2x^2 - 2y^2
        m22 = 1 - (2 * qx**2) - (2 * qy**2)

        # 回転軸の順番がX->Y->Zの固定角(Rz*Ry*Rx)
        # if m01 == -1:
        # 	tx = 0
        # 	ty = math.pi/2
        # 	tz = math.atan2(m20, m10)
        # elif m20 == 1:
        # 	tx = 0
        # 	ty = -math.pi/2
        # 	tz = math.atan2(m20, m10)
        # else:
        # 	tx = -math.atan2(m02, m00)
        # 	ty = -math.asin(-m01)
        # 	tz = -math.atan2(m21, m11)

        # 回転軸の順番がX->Y->Zのオイラー角(Rx*Ry*Rz)
        if m02 == 1:
            tx = math.atan2(m10, m11)
            ty = math.pi / 2
            tz = 0
        elif m02 == -1:
            tx = math.atan2(m21, m20)
            ty = -math.pi / 2
            tz = 0
        else:
            tx = -math.atan2(-m12, m22)
            ty = -math.asin(m02)
            tz = -math.atan2(-m01, m00)

        if isDeg:
            tx = np.rad2deg(tx)
            ty = np.rad2deg(ty)
            tz = np.rad2deg(tz)

        rotEuler = np.array([tx, ty, tz])
        return rotEuler

    def ScipyQuaternion2Euler(self, q, sequence: str = "xyz", isDeg: bool = True):
        """
        Calculate the Euler angle from the Quaternion.
        Using scipy.spatial.transform.Rotation.as_euler

        Parameters
        ----------
        q: np.ndarray
            Quaternion.
            [x, y, z, w]
        sequence: (Optional) str
            Rotation sequence of Euler representation, specified as a string.
            The rotation sequence defines the order of rotations about the axes.
            The default is xyz.
        isDeg: (Optional) bool
            Returned angles are in degrees if this flag is True, else they are in radians.
            The default is True.

        Returns
        ----------
        rotEuler: np.ndarray
            Euler angle.
            [x, y, z]
        """

        quat = scitransform.Rotation.from_quat(q)
        rotEuler = quat.as_euler(sequence, degrees=isDeg)
        return rotEuler

    def Euler2Quaternion(self, e):
        """
        Calculate the Quaternion from the Euler angle.

        Parameters
        ----------
        e: np.ndarray
            Euler.
            [x, y, z]

        Returns
        ----------
        rotQuat: np.ndarray
            Quaternion
            [x, y, z, w]
        """

        roll = np.deg2rad(e[0])
        pitch = np.deg2rad(e[1])
        yaw = np.deg2rad(e[2])

        cosRoll = np.cos(roll / 2.0)
        sinRoll = np.sin(roll / 2.0)
        cosPitch = np.cos(pitch / 2.0)
        sinPitch = np.sin(pitch / 2.0)
        cosYaw = np.cos(yaw / 2.0)
        sinYaw = np.sin(yaw / 2.0)

        q0 = cosRoll * cosPitch * cosYaw + sinRoll * sinPitch * sinYaw
        q1 = sinRoll * cosPitch * cosYaw - cosRoll * sinPitch * sinYaw
        q2 = cosRoll * sinPitch * cosYaw + sinRoll * cosPitch * sinYaw
        q3 = cosRoll * cosPitch * sinYaw - sinRoll * sinPitch * cosYaw

        rotQuat = [q1, q2, q3, q0]
        return rotQuat

    def ScipyEuler2Quaternion(self, e, sequence: str = "xyz", isDeg: bool = True):
        """
        Calculate the Quaternion from the Euler angle.
        Using scipy.spatial.transform.Rotation.as_quat

        Parameters
        ----------
        e: np.ndarray
            Euler.
            [x, y, z]
        sequence: (Optional) str
            Rotation sequence of Euler representation, specified as a string.
            The rotation sequence defines the order of rotations about the axes.
            The default is xyz.
        isDeg: (Optional) bool
            If True, then the given angles are assumed to be in degrees. Default is True.

        Returns
        ----------
        rotQuat: np.ndarray
            Quaternion
            [x, y, z, w]
        """

        quat = scitransform.Rotation.from_euler(sequence, e, isDeg)
        rotQuat = quat.as_quat()
        return rotQuat

    def InversedRotation(self, rot, axes: list = []):
        """
        Calculate the inversed rotation.

        ----- CAUTION -----
        If "axes" is set, it will be converted to Euler angles during the calculation process, which may result in inaccurate rotation.
        In addition, the behavior near the singularity is unstable.

        Parameters
        ----------
        rot: np.ndarray
            Quaternion.
            [x, y, z, w]
        axes: (Optional) list[str]
            Axes to be inversed.
            If length of axes is zero, return inversed quaternion

        Returns
        ----------
        inversedRot: np.ndarray
            Inversed rotation
            [x, y, z, w]
        """

        if len(axes) == 0:
            quat = scitransform.Rotation.from_quat(rot)
            inversedRot = quat.inv().as_quat()
            return inversedRot

        rot = self.ScipyQuaternion2Euler(rot)

        for axis in axes:
            if axis == "x":
                rot[0] = -rot[0]
            elif axis == "y":
                rot[1] = -rot[1]
            elif axis == "z":
                rot[2] = -rot[2]

        inversedRot = self.ScipyEuler2Quaternion(rot)

        return inversedRot

    def NumpyArray2Dict(self, numpyArray, dictKey: str = "participant"):
        """
        Convert numpy array to dict.

        Parameters
        ----------
        numpyArray: numpy array
            Numpy array.
        dictKey: (Optional) str
            The key name of the dict.
        """

        if type(numpyArray) is np.ndarray:
            dictionary = {}
            if len(numpyArray.shape) == 1:
                dictionary[dictKey + str(1)] = numpyArray
            else:
                for i in range(len(numpyArray)):
                    dictionary[dictKey + str(i + 1)] = numpyArray[i]
        else:
            print("Type Error: argument is NOT Numpy array")
            return

        return dictionary

    def quaternion_to_rotation_matrix(self, quaternion):
        """
        クォータニオンを回転行列に変換する関数
        """
        x, y, z, w = quaternion
        rotation_matrix = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
        ])
        return rotation_matrix

    def add_rotation(self, original_euler, angle_deg):
        # オイラー角をクォータニオンに変換
        r_original = R.from_euler('xyz', original_euler, degrees=True)
        
        # 回転角度をクォータニオンに変換（y軸の回転として指定）
        r_rotation = R.from_euler('y', angle_deg, degrees=True)
        
        # クォータニオンの回転を適用
        r_result =  r_original * r_rotation
        
        # 結果をオイラー角に変換して返す
        return r_result.as_euler('xyz', degrees=True)