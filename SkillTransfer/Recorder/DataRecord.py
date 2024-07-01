import csv
import datetime
import math
import os
import threading

import numpy as np
import tqdm
from Recorder.UDPReceive import udprecv


class DataRecordManager:
    dictPosition = {}
    dictRotation = {}
    dictWeightPosition = {}
    dictWeightRotation = {}
    dictGripperValue_P = {}
    dictDurationTime = []
    dictRobotPosition = {}
    dictRobotRotation = {}
    dictGripperValue_R = {}
    dictRobotHead = []

    def __init__(self, participantNum: int = 2, otherRigidBodyNum: int = 0, otherRigidBodyNames: list = ["endEffector"], bendingSensorNum: int = 2, robotNum: int = 2) -> None:
        """
        Initialize class: DataRecordManager

        Parameters
        ----------
        participantNum: (Optional) int
            Number of participants
        otherRigidBodyNum: (Optional) int
            Number of rigid body objects except participants' rigid body
        otherRigidBodyNames: (Optional) list(str)
            Name list of rigid body objects except participants' rigid body
        bendingSensorNum: (Optional) int
            Number of bending sensors
        """

        self.participantNum = participantNum
        self.otherRigidBodyNum = otherRigidBodyNum
        self.otherRigidBodyNames = otherRigidBodyNames
        self.bendingSensorNum = bendingSensorNum
        self.robotNum = robotNum

        for i in range(self.participantNum):
            self.dictPosition["participant" + str(i + 1)] = []
            self.dictRotation["participant" + str(i + 1)] = []
            self.dictWeightPosition["participant" + str(i + 1)] = []
            self.dictWeightRotation["participant" + str(i + 1)] = []

        for i in range(self.otherRigidBodyNum):
            self.dictPosition["otherRigidBody" + str(i + 1)] = []
            self.dictRotation["otherRigidBody" + str(i + 1)] = []

        for i in range(self.bendingSensorNum):
            self.dictGripperValue_P["gripperValue_P" + str(i + 1)] = []

        for i in range(self.robotNum):
            self.dictRobotPosition["robot" + str(i + 1)] = []
            self.dictRobotRotation["robot" + str(i + 1)] = []

        for i in range(self.robotNum):
            self.dictGripperValue_R["gripperValue_R" + str(i + 1)] = []

        # forhead!!!!!!
        # self.udp = udprecv()  # クラス呼び出し
        # streamingThread = threading.Thread(target=self.udp.recv)
        # streamingThread.setDaemon(True)
        # streamingThread.start()

    def Record(self,
               position,
               rotation,
               weight,
            #    Gripper_P,
               robotpos,
               robotrot,
            #    Gripper_R,
               duration):
        """
        Record the data.

        Parameters
        ----------
        position: dict
            Position
        rotation: dict
            Rotation
        bendingSensor: dict
            Bending sensor values
        """

        self.dictDurationTime.append([duration])

        for i in range(self.participantNum):
            self.dictPosition["participant" + str(i + 1)].append(position["participant" + str(i + 1)])
            self.dictRotation["participant" + str(i + 1)].append(self.Quaternion2Euler(q=rotation["participant" + str(i + 1)]))
            self.dictWeightPosition["participant" + str(i + 1)].append(weight[0][i])
            self.dictWeightRotation["participant" + str(i + 1)].append(weight[1][i])

        for i in range(self.otherRigidBodyNum):
            self.dictPosition["otherRigidBody" + str(i + 1)].append(position["otherRigidBody" + str(i + 1)])
            self.dictRotation["otherRigidBody" + str(i + 1)].append(self.Quaternion2Euler(q=rotation["otherRigidBody" + str(i + 1)]))

        # for i in range(self.bendingSensorNum):
        #     self.dictGripperValue_P["gripperValue_P" + str(i + 1)].append([Gripper_P["gripperValue" + str(i + 1)]])

        for i in range(self.robotNum):
            self.dictRobotPosition["robot" + str(i + 1)].append(robotpos["robot" + str(i + 1)])
            self.dictRobotRotation["robot" + str(i + 1)].append(robotrot["robot" + str(i + 1)])

        # for i in range(self.robotNum):
        #     self.dictGripperValue_R["gripperValue_R" + str(i + 1)].append([float(Gripper_R["gripperValue" + str(i + 1)])])

        # self.dictRobotHead.append(self.udp.robot_head)

    def ExportSelf(self, dirPath: str = "ExportData", participant: str = "", conditions: str = "", number: str = ""):
        """
        Export the data recorded in DataRecordManager as CSV format.

        Parameters
        ----------
        dirPath: (Optional) str
            Directory path (not include the file name).
        """
        # transformHeader = ["time", "x", "y", "z", "qx", "qy", "qz", "qw", "weightpos", "weightrot"]
        transformHeader = ["time", "x", "y", "z", "roll", "pitch", "yaw", "weightpos", "weightrot"]
        GripperHeader = ["GripperValue"]
        robotHeader = ["time", "x", "y", "z", "roll", "pitch", "yaw"]
        headHeader = ["time", "x", "y", "z", "rx", "ry", "rz"]

        print("\n---------- DataRecordManager.ExportSelf ----------")
        print("Writing: Participant transform...")
        for i in tqdm.tqdm(range(self.participantNum), ncols=150):
            npDuration = np.array(self.dictDurationTime)
            npPosition = np.array(self.dictPosition["participant" + str(i + 1)])
            npRotation = np.array(self.dictRotation["participant" + str(i + 1)])
            npWeightPosition = np.array(self.dictWeightPosition["participant" + str(i + 1)])
            npWeightRotation = np.array(self.dictWeightRotation["participant" + str(i + 1)])
            npParticipantTransform = np.concatenate([npPosition, npRotation], axis=1)
            npTimeParticipantTransform = np.c_[npDuration, npParticipantTransform, npWeightPosition, npWeightRotation]
            self.ExportAsCSV(npTimeParticipantTransform, dirPath, "Transform_Participant_" + str(i + 1), participant, conditions, number, transformHeader)

        print("Writing: Other rigid body transform...")
        for i in tqdm.tqdm(range(self.otherRigidBodyNum), ncols=150):
            npDuration = np.array(self.dictDurationTime)
            npPosition = np.array(self.dictPosition["otherRigidBody" + str(i + 1)])
            npRotation = np.array(self.dictRotation["otherRigidBody" + str(i + 1)])
            npRigidBodyTransform = np.concatenate([npPosition, npRotation], axis=1)
            npTimeRigidBodyTransform = np.c_[npDuration, npRigidBodyTransform]
            self.ExportAsCSV(npTimeRigidBodyTransform, dirPath, "OtherRigidBody_" + str(i + 1), participant, conditions, number, transformHeader)

        # print("Writing: Participant Gripper value...")
        # for i in tqdm.tqdm(range(self.bendingSensorNum), ncols=150):
        #     npGripperValue_P = np.array(self.dictGripperValue_P["gripperValue_P" + str(i + 1)])
        #     self.ExportAsCSV(npGripperValue_P, dirPath, "GripperValue_Participant_" + str(i + 1), participant, conditions, number, GripperHeader)

        print("Writing: Robot transform...")
        for i in tqdm.tqdm(range(self.robotNum), ncols=150):
            npDuration = np.array(self.dictDurationTime)
            npRobotPosition = np.array(self.dictRobotPosition["robot" + str(i + 1)])
            npRobotRotation = np.array(self.dictRobotRotation["robot" + str(i + 1)])
            npRobotTransform = np.concatenate([npRobotPosition, npRobotRotation], axis=1)
            npTimeRobotTransform = np.c_[npDuration, npRobotTransform]
            self.ExportAsCSV(npTimeRobotTransform, dirPath, "Transform_Robot_" + str(i + 1), participant, conditions, number, robotHeader)

        # print("Writing: Robot Gripper value...")
        # for i in tqdm.tqdm(range(self.robotNum), ncols=150):
        #     npGripperValue_R = np.array( self.dictGripperValue_R["gripperValue_R" + str(i + 1)])
        #     self.ExportAsCSV( npGripperValue_R, dirPath, "GripperValue_Robot_" + str(i + 1), participant, conditions, number, GripperHeader)

        # print("Writing: Head value...")
        # npDuration = np.array(self.dictDurationTime)
        # npHead = np.array(self.dictRobotHead)
        # npHeadOutput = np.concatenate([npDuration, npHead], axis=1)
        # self.ExportAsCSV(npHeadOutput, dirPath, "Head_", participant, conditions, number, headHeader)
        # print("---------- Export completed ----------\n")

    def ExportAsCSV(self, data, dirPath, fileName, participant, conditions, number, header: list = []):
        """
        Export the data to CSV file.

        Parameters
        ----------
        data: array like
            Data to be exported.
        dirPath: str
            Directory path (not include the file name).
        fileName: str
            File name. (not include ".csv")
        header: (Optional) list
            Header of CSV file. If list is empty, CSV file not include header.
        """
        # ----- Check directory ----- #
        self.mkdir(dirPath)

        # exportPath = dirPath + '/' + participant + '_' + conditions + '_' + number + '_' + fileName + '_' + datetime.datetime.now().strftime('%Y%m%d_%H%M') + '.csv'
        exportPath = dirPath + "/" + fileName + "_" + datetime.datetime.now().strftime("%Y%m%d_%H%M") + ".csv"

        with open(exportPath, "w", newline="") as f:
            writer = csv.writer(f)

            if header:
                writer.writerow(header)
            writer.writerows(data)

    def mkdir(self, path):
        """
        Check existence of the directory, and if it does not exist, create a new one.

        Parameters
        ----------
        path: str
            Directory path
        """

        if not os.path.isdir(path):
            os.makedirs(path)

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
