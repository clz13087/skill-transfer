import datetime
import pprint
import threading
import time
import winsound
import csv
import os
import glob
import socket
import json
import matplotlib.pyplot as plt
from ctypes import windll
from datetime import datetime
from enum import Flag
from turtle import right
import numpy as np
from scipy.spatial.transform import Rotation as R
from FileIO.FileIO import FileIO
from matplotlib.pyplot import flag
from Participant.ParticipantMotion import ParticipantMotion
from Recorder.DataRecord import DataRecordManager
from Robot.CAMotion import CAMotion
from Robot.xArmTransform import xArmTransform
from xarm.wrapper import XArmAPI
from Participant.lstm_predictor import LSTMPredictor
from scipy.signal import butter, filtfilt

# ---------- Settings: Input mode ---------- #
motionDataInputMode = "optitrack"
gripperDataInputMode = "bendingsensor"

class ProcessorClass:
    def __init__(self) -> None:
        fileIO = FileIO()
        dat = fileIO.Read("config/settings.csv", ",")

        xArmIP_left = [addr for addr in dat if "xArmIPAddress_left" in addr[0]][0][1]
        initialpos_left = [addr for addr in dat if "initialpos_left" in addr[0]]
        initialrot_left = [addr for addr in dat if "initialrot_left" in addr[0]]
        initAngleList_left = [addr for addr in dat if "initAngleList_left" in addr[0]]

        xArmIP_right = [addr for addr in dat if "xArmIPAddress_right" in addr[0]][0][1]
        initialpos_right = [addr for addr in dat if "initialpos_right" in addr[0]]
        initialrot_right = [addr for addr in dat if "initialrot_right" in addr[0]]
        initAngleList_right = [addr for addr in dat if "initAngleList_right" in addr[0]]

        wirelessIP = [addr for addr in dat if "wirelessIPAddress" in addr[0]][0][1]
        localIP = [addr for addr in dat if "localIPAddress" in addr[0]][0][1]
        motiveserverIP = [addr for addr in dat if "motiveServerIPAddress" in addr[0]][0][1]
        motivelocalIP = [addr for addr in dat if "motiveLocalIPAddress" in addr[0]][0][1]
        frameRate = [addr for addr in dat if "frameRate" in addr[0]][0][1]

        lstmClientAddress = [addr for addr in dat if "lstmClientAddress" in addr[0]][0][1]
        lstmClientPort = [addr for addr in dat if "lstmClientPort" in addr[0]][0][1]
        lstmServerAddress = [addr for addr in dat if "lstmServerAddress" in addr[0]][0][1]
        lstmServerPort = [addr for addr in dat if "lstmServerPort" in addr[0]][0][1]

        isExportData =  [addr for addr in dat if "isExportData" in addr[0]][0][1]
        if isExportData == "False":
            isExportData = 0
        elif isExportData == "True":
            isExportData = 1
        dirPath = [addr for addr in dat if "dirPath" in addr[0]][0][1]

        participantNum = [addr for addr in dat if "participantNum" in addr[0]][0][1]
        gripperNum = [addr for addr in dat if "gripperNum" in addr[0]][0][1]
        otherRigidBodyNum = [addr for addr in dat if "otherRigidBodyNum" in addr[0]][0][1]
        robotNum = [addr for addr in dat if "robotNum" in addr[0]][0][1]
        idList = [addr for addr in dat if "idList" in addr[0]]
        differenceLimit = [addr for addr in dat if "differenceLimit" in addr[0]][0][1]

        recordedDataPath = [addr for addr in dat if "recordedDataPath" in addr[0]][0][1]

        weightListPos = [addr for addr in dat if "weightListPos" in addr[0]]
        weightListRot = [addr for addr in dat if "weightListRot" in addr[0]]

        self.xArmIpAddress_left = xArmIP_left
        self.initialpos_left = initialpos_left
        self.initislrot_left = initialrot_left
        self.initAngleList_left =  list(map(float, initAngleList_left[0][1:]))

        self.xArmIpAddress_right = xArmIP_right
        self.initialpos_right = initialpos_right
        self.initislrot_right = initialrot_right
        self.initAngleList_right =  list(map(float, initAngleList_right[0][1:]))

        self.wirelessIpAddress = wirelessIP
        self.localIpAddress = localIP
        self.motiveserverIpAddress = motiveserverIP
        self.motivelocalIpAddress = motivelocalIP
        self.frameRate = int(frameRate)

        self.lstmClientAddress = lstmClientAddress
        self.lstmClientPort = int(lstmClientPort)
        self.lstmServerAddress = lstmServerAddress
        self.lstmServerPort = int(lstmServerPort)

        self.isExportData = bool(isExportData)
        self.dirPath = dirPath

        self.participantNum = int(participantNum)
        self.gripperNum = int(gripperNum)
        self.otherRigidBodyNum = int(otherRigidBodyNum)
        self.robotNum = int(robotNum)
        self.idList = idList

        self.differenceLimit = float(differenceLimit)

        self.recordedDataPath = recordedDataPath

        self.weightListPos = weightListPos
        self.weightListRot = weightListRot

    def mainloop(self, isEnablexArm: bool = True):
        """
        Send the position and rotation to the xArm
        """

        # ----- Process info ----- #
        self.loopCount = 0
        self.taskTime = []
        self.errorCount = 0
        taskStartTime = 0
        ratiolist = []
        timelist = []

        # ----- Instantiating custom classes ----- #
        caMotion = CAMotion(defaultParticipantNum=2, otherRigidBodyNum=self.otherRigidBodyNum,differenceLimit=self.differenceLimit)
        transform_left = xArmTransform(initpos=self.initialpos_left, initrot=self.initislrot_left, initangle=self.initAngleList_left)
        transform_right = xArmTransform(initpos=self.initialpos_right, initrot=self.initislrot_right, initangle=self.initAngleList_right)
        dataRecordManager = DataRecordManager(participantNum=self.participantNum, otherRigidBodyNum=self.otherRigidBodyNum, bendingSensorNum=self.gripperNum, robotNum=self.robotNum)
        participantMotion = ParticipantMotion(defaultParticipantNum=2, otherRigidBodyNum=self.otherRigidBodyNum, motionInputSystem=motionDataInputMode, mocapServer=self.motiveserverIpAddress, mocapLocal=self.motivelocalIpAddress, idList=self.idList)
        lstmPredictor = LSTMPredictor(self.lstmClientAddress, self.lstmClientPort, self.lstmServerAddress, self.lstmServerPort)

        # ----- Load recorded data. ----- #
        for i in [3, 4]:
            participant_path = os.path.join(self.recordedDataPath, f"*Transform_Participant_{i-2}*.csv")
            globals()[f"participant{i}_data"] = self.load_csv_data(glob.glob(participant_path)[0])

        # ----- weight list ----- #
        weightListPosfloat = list(map(float, self.weightListPos[0][1:]))
        weightListRotfloat = list(map(float, self.weightListRot[0][1:]))
        weightList = [weightListPosfloat, weightListRotfloat]

        # ----- Initialize robot arm ----- #
        if isEnablexArm:
            arm_1 = XArmAPI(self.xArmIpAddress_left)
            self.InitializeAll(arm_1, transform_left)

            arm_2 = XArmAPI(self.xArmIpAddress_right)
            self.InitializeAll(arm_2, transform_right)

        # ----- Control flags ----- #
        isMoving = False

        try:
            while True:
                if isMoving:
                    # ----- Get relative----- #
                    localPosition = participantMotion.LocalPosition(loopCount=self.loopCount)
                    localRotation = participantMotion.LocalRotation(loopCount=self.loopCount)
                    relativePosition = caMotion.GetRelativePosition(position=localPosition)
                    relativeRotation = caMotion.GetRelativeRotation(rotation=localRotation)

                    # ----- record ----- #
                    for i in [3, 4]:
                        relativePosition[f"participant{i}"] = np.array(globals()[f"participant{i}_data"][min(self.loopCount, len(globals()[f"participant{i}_data"]) - 1)]["position"])
                        relativeRotation[f"participant{i}"] = np.array(globals()[f"participant{i}_data"][min(self.loopCount, len(globals()[f"participant{i}_data"]) - 1)]["rotation"])

                    # ----- lstm ----- #
                    # send_pos_rot = [value for array in [relativePosition["participant1"], relativePosition["participant2"], relativeRotation["participant1"],  relativeRotation["participant2"]] for value in array]
                    # send_pos_rot.insert(0, time.perf_counter() - taskStartTime)
                    # predictedList = lstmPredictor.predict_position_rotation(send_pos_rot)
                    # if predictedList:
                    #     relativePosition["participant5"], relativePosition["participant6"], relativeRotation["participant5"], relativeRotation["participant6"] = predictedList[0:3], predictedList[3:6], predictedList[6:10], predictedList[10:14]
                    # else:
                    #     relativePosition["participant5"], relativePosition["participant6"], relativeRotation["participant5"], relativeRotation["participant6"] = np.zeros(3), np.zeros(3), np.array([0, 0, 0, 1]), np.array([0, 0, 0, 1])

                    # ----- Difference calculation and transmission to transparent ----- #
                    # relativePosition_for_difference = relativePosition
                    # for i in [3, 4]:
                    #     relativePosition_for_difference[f"participant{i}"] = np.array(globals()[f"participant{i}_data"][min(self.loopCount + int(self.frameRate * 0.3), len(globals()[f"participant{i}_data"]) - 1)]["position"]) #lstmの予測秒数に合わせて，記録も予測秒数分先を用いる
                    # average_diff, left_diff, right_diff = caMotion.calculate_difference(relativePosition_for_difference)
                    # self.frameRate = 200 - (average_diff / self.differenceLimit) * (200 - 100)
                    # self.frameRate = 200
                    # data_to_send = {"frameRate": self.frameRate, "average_diff": average_diff, "left_diff": left_diff, "right_diff": right_diff}
                    # with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                    #     sock.sendto(json.dumps(data_to_send).encode(), ('133.68.108.26', 8000))

                    # ----- Control ratio varies depending on the deference. ----- #
                    # ratio = average_diff/self.differenceLimit
                    # ratiolist.append(ratio)
                    # timelist.append(time.perf_counter() - taskStartTime)
                    # weightList = [[1-ratio, 1-ratio, ratio, ratio, 0, 0], [1-ratio, 1-ratio, ratio, ratio, 0, 0]]
                    # weightList = [[1-ratio, 1-ratio, ratio, ratio, 0, 0], [0, 0, 1, 1, 0, 0]]
                    # weightList = [[0, 0, 1, 1, 0, 0], [1-ratio, 1-ratio, ratio, ratio, 0, 0]]
                    # weightList = [[0, 0, 1, 1, 0, 0], [0, 0, 1, 1, 0, 0]]
                    # print(weightList)

                    # ----- Calculate the integration ----- #
                    robotpos, robotrot = caMotion.participant2robot_all_quaternion(relativePosition, relativeRotation, weightList)
                
                    # ----- Send to xArm ----- #
                    if isEnablexArm:
                        arm_1.set_servo_cartesian(transform_left.Transform(relativepos=robotpos["robot1"], relativerot=robotrot["robot1"], isLimit=False))
                        arm_2.set_servo_cartesian(transform_right.Transform(relativepos=robotpos["robot2"], relativerot=robotrot["robot2"], isLimit=False))

                    # ----- Data recording ----- #
                    if self.isExportData:
                        dataRecordManager.Record(position=relativePosition, rotation=relativeRotation, weight=weightList, robotpos=robotpos, robotrot=robotrot, duration=time.perf_counter() - taskStartTime)

                    # ---------- fix framerate ---------- #
                    self.fix_framerate((time.perf_counter() - loop_start_time), 1/self.frameRate)
                    self.loopCount += 1
                    loop_start_time = time.perf_counter()

                else:
                    keycode = input('Input > "q": quit, "r": Clean error and init arm, "s": start control \n')
                    # ----- Quit program ----- #
                    if keycode == "q":
                        if isEnablexArm:
                            arm_1.disconnect()
                            arm_2.disconnect()
                        self.PrintProcessInfo()

                        windll.winmm.timeEndPeriod(1)
                        break

                    # ----- Reset xArm and gripper ----- #
                    elif keycode == "r":
                        if isEnablexArm:
                            self.InitializeAll(arm_1, transform_left)
                            self.InitializeAll(arm_2, transform_right)

                    # ----- Start streaming ----- #
                    elif keycode == "s":
                        # ----- A beep sounds after 5 seconds and send s-key to the Mac side ----- #
                        time.sleep(5)
                        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                            sock.sendto(b's', ('133.68.108.26', 8000))
                        winsound.Beep(1000,1000)

                        # ----- set initial pos and rot ----- #
                        caMotion.SetOriginPosition(participantMotion.LocalPosition())
                        caMotion.SetInversedMatrix(participantMotion.LocalRotation())

                        # ----- flag and tasktime ----- #
                        isMoving = True
                        taskStartTime = loop_start_time = time.perf_counter()

        except KeyboardInterrupt:
            print("\nKeyboardInterrupt >> Stop: mainloop()")

            self.taskTime.append(time.perf_counter() - taskStartTime)
            self.PrintProcessInfo()

            # Mac側にsキーを送信
            if self.loopCount > 100:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                    sock.sendto(b's', ('133.68.108.26', 8000))
                    print("Send stop to transparent")

            if self.isExportData:
                dataRecordManager.ExportSelf(dirPath=self.dirPath)

            # ----- Disconnect ----- #
            if isEnablexArm:
                arm_1.disconnect()
                arm_2.disconnect()

            windll.winmm.timeEndPeriod(1)

            if self.loopCount > 100:
                plt.plot(timelist, ratiolist, linestyle='-')
                plt.xlabel('Time')
                plt.ylabel('Ratio')
                plt.ylim(0, 1)
                plt.show()

        except:
            print("----- Exception has occurred -----")
            windll.winmm.timeEndPeriod(1)
            import traceback

            traceback.print_exc()

    def ConvertToModbusData(self, value: int):
        """
        Converts the data to modbus type.

        Parameters
        ----------
        value: int
            The data to be converted.
            Range: 0 ~ 800
        """

        if int(value) <= 255 and int(value) >= 0:
            dataHexThirdOrder = 0x00
            dataHexAdjustedValue = int(value)

        elif int(value) > 255 and int(value) <= 511:
            dataHexThirdOrder = 0x01
            dataHexAdjustedValue = int(value) - 256

        elif int(value) > 511 and int(value) <= 767:
            dataHexThirdOrder = 0x02
            dataHexAdjustedValue = int(value) - 512

        elif int(value) > 767 and int(value) <= 1123:
            dataHexThirdOrder = 0x03
            dataHexAdjustedValue = int(value) - 768

        modbus_data = [0x08, 0x10, 0x07, 0x00, 0x00, 0x02, 0x04, 0x00, 0x00]
        modbus_data.append(dataHexThirdOrder)
        modbus_data.append(dataHexAdjustedValue)

        return modbus_data

    def PrintProcessInfo(self):
        """
        Print process information.
        """

        print("----- Process info -----")
        print("Total loop count > ", self.loopCount)
        for ttask in self.taskTime:
            print("Task time\t > ", "{:.2f}".format(ttask), "[s]")
            print("Frame Rate\t > ", "{:.2f}".format(self.loopCount/ttask), "[fps]")
        print("Error count\t > ", self.errorCount)
        print("------------------------")

    def InitializeAll(self, robotArm, transform, isSetInitPosition=False, isSetInitAngle=True):
        """
        Initialize the xArm

        Parameters
        ----------
        robotArm: XArmAPI
            XArmAPI object.
        transform: xArmTransform
            xArmTransform object.
        isSetInitPosition: (Optional) bool
            True -> Set to "INITIAL POSITION" of the xArm studio
            False -> Set to "ZERO POSITION" of the xArm studio
        """

        robotArm.connect()
        if robotArm.warn_code != 0:
            robotArm.clean_warn()
        if robotArm.error_code != 0:
            robotArm.clean_error()
        robotArm.motion_enable(enable=True)
        robotArm.set_mode(0)  # set mode: position control mode
        robotArm.set_state(state=0)  # set state: sport state
        if isSetInitAngle:
            init_angle_list = transform.GetInitialAngle()
            robotArm.set_servo_angle(angle=init_angle_list, is_radian=False, wait=True)
        # if isSetInitPosition:
        #     initX, initY, initZ, initRoll, initPitch, initYaw = transform.GetInitialTransform()
        #     robotArm.set_position(x=initX, y=initY, z=initZ, roll=initRoll, pitch=initPitch, yaw=initYaw, wait=True)
        else:
            robotArm.reset(wait=True)
        print("Initialized > xArm")

        robotArm.set_mode(1)
        robotArm.set_state(state=0)

    def fix_framerate(self, process_duration, looptime):
        sleeptime = looptime - process_duration
        if sleeptime < 0:
            pass
        else:
            time.sleep(sleeptime)

    def load_csv_data(self, file_path):
        with open(file_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            data = []
            for row in reader:
                data.append({
                    "time": float(row["time"]),
                    "position": [float(row["x"]), float(row["y"]), float(row["z"])],
                    "rotation": [float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"])]
                })
        return data