import csv
import datetime
import math
import os
import threading
import time

import numpy as np
import tqdm
from Recorder.UDPReceive import udprecv


class DataRecordManager:
    dictPosition = {}
    dictRotation = {}
    dictWeightPosition = {}
    dictWeightRotation = {}
    dictGripperValue_P = {}
    dictTime = []
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
        current_time = time.time()
        formatted_time = time.strftime("%Y%m%d.%H%M%S", time.localtime(current_time))
        milliseconds = int((current_time - int(current_time)) * 1000)  # ミリ秒部分を計算
        formatted_time_with_milliseconds = f"{formatted_time}.{milliseconds:03d}"  # ミリ秒を含める

        # フォーマット済みの時刻を記録
        self.dictTime.append([formatted_time_with_milliseconds])

        self.dictDurationTime.append([duration])

        for i in range(self.participantNum):
            self.dictPosition["participant" + str(i + 1)].append(position["participant" + str(i + 1)])
            self.dictRotation["participant" + str(i + 1)].append(rotation["participant" + str(i + 1)])
            self.dictWeightPosition["participant" + str(i + 1)].append(weight[0][i])
            self.dictWeightRotation["participant" + str(i + 1)].append(weight[1][i])

        for i in range(self.otherRigidBodyNum):
            self.dictPosition["otherRigidBody" + str(i + 1)].append(position["otherRigidBody" + str(i + 1)])
            self.dictRotation["otherRigidBody" + str(i + 1)].append(rotation["otherRigidBody" + str(i + 1)])

        for i in range(self.robotNum):
            self.dictRobotPosition["robot" + str(i + 1)].append(robotpos["robot" + str(i + 1)])
            self.dictRobotRotation["robot" + str(i + 1)].append(robotrot["robot" + str(i + 1)])

    def ExportSelf(self, dirPath: str = "ExportData"):
        """
        Export the data recorded in DataRecordManager as CSV format.

        Parameters
        ----------
        dirPath: (Optional) str
            Directory path (not include the file name).
        """
        transformHeader = ["timestamp", "time", "x", "y", "z", "qx", "qy", "qz", "qw", "weightpos", "weightrot"]
        robotHeader = ["timestamp", "time", "x", "y", "z", "rx", "ry", "rz", "qx", "qy", "qz", "qw"]
        headHeader = ["timestamp", "time", "x", "y", "z", "rx", "ry", "rz"]

        print("\n---------- DataRecordManager.ExportSelf ----------")
        print("Writing: Participant transform...")
        for i in tqdm.tqdm(range(self.participantNum), ncols=150):
            npTime = np.array(self.dictTime)
            npDuration = np.array(self.dictDurationTime)
            npPosition = np.array(self.dictPosition["participant" + str(i + 1)])
            npRotation = np.array(self.dictRotation["participant" + str(i + 1)])
            npWeightPosition = np.array(self.dictWeightPosition["participant" + str(i + 1)])
            npWeightRotation = np.array(self.dictWeightRotation["participant" + str(i + 1)])
            npParticipantTransform = np.concatenate([npPosition, npRotation], axis=1)
            npTimeParticipantTransform = np.c_[npTime, npDuration, npParticipantTransform, npWeightPosition, npWeightRotation]
            self.ExportAsCSV(npTimeParticipantTransform, dirPath, "Transform_Participant_" + str(i + 1), transformHeader)

        print("Writing: Other rigid body transform...")
        for i in tqdm.tqdm(range(self.otherRigidBodyNum), ncols=150):
            npTime = np.array(self.dictTime)
            npDuration = np.array(self.dictDurationTime)
            npPosition = np.array(self.dictPosition["otherRigidBody" + str(i + 1)])
            npRotation = np.array(self.dictRotation["otherRigidBody" + str(i + 1)])
            npRigidBodyTransform = np.concatenate([npPosition, npRotation], axis=1)
            npTimeRigidBodyTransform = np.c_[npTime, npDuration, npRigidBodyTransform]
            self.ExportAsCSV(npTimeRigidBodyTransform, dirPath, "OtherRigidBody_" + str(i + 1), transformHeader)

        print("Writing: Robot transform...")
        for i in tqdm.tqdm(range(self.robotNum), ncols=150):
            npTime = np.array(self.dictTime)
            npDuration = np.array(self.dictDurationTime)
            npRobotPosition = np.array(self.dictRobotPosition["robot" + str(i + 1)])
            npRobotRotation = np.array(self.dictRobotRotation["robot" + str(i + 1)])
            npRobotTransform = np.concatenate([npRobotPosition, npRobotRotation], axis=1)
            npTimeRobotTransform = np.c_[npTime, npDuration, npRobotTransform]
            self.ExportAsCSV(npTimeRobotTransform, dirPath, "Transform_Robot_" + str(i + 1), robotHeader)

    def ExportAsCSV(self, data, dirPath, fileName, header: list = []):
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