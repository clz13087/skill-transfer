from math import pi
import numpy as np
from Filter.Filter import MotionFilter
from FileIO.FileIO import FileIO


class xArmTransform:
    """
    xArmの座標と回転を保持するクラス
    """

    x, y, z = 0, 0, 0
    roll, pitch, yaw = 0, 0, 0

    # ----- Minimum limitation ----- #
    __minX, __minY, __minZ = -999, -999, -999
    __minRoll, __minPitch, __minYaw = -180, -180, -180

    # ----- Maximum limitation ----- #
    __maxX, __maxY, __maxZ = 999, 999, 999
    __maxRoll, __maxPitch, __maxYaw = 180, 180, 180

    def __init__(self, initpos: list, initrot: list, mount: str):
        self.n = 4
        self.fp = 10
        self.fs = 180
        self.filter_robot = MotionFilter()
        self.filter_robot.InitLowPassFilterWithOrder(self.fs, self.fp, self.n)

        self.beforefilt = [[0, 0, 0, 0, 0, 0]] * self.n
        self.afterfilt = [[0, 0, 0, 0, 0, 0]] * self.n

        self.__initX, self.__initY, self.__initZ = int(initpos[0][1]), int(initpos[0][2]), int(initpos[0][3])
        self.__initRoll, self.__initPitch, self.__initYaw = int(initrot[0][1]), int(initrot[0][2]), int(initrot[0][3])
        self.mount = mount

    def GetInitialTransform(self):
        """
        Get the initial position and rotation.
        """

        return self.__initX, self.__initY, self.__initZ, self.__initRoll, self.__initPitch, self.__initYaw

    def Transform(self, relativepos: list, relativerot: list, isLimit=True, isOnlyPosition=False):
        """
        Converts from motive coordinate to xarm coordinate and limits values.
        """

        relativepos_mm = relativepos * 1000

        if self.mount == "left":
            x, y, z = relativepos_mm[2] + self.__initX, -1 * relativepos_mm[1] + self.__initY, relativepos_mm[0] + self.__initZ
            roll, pitch, yaw = relativerot[2] + self.__initRoll, -1 * relativerot[1] + self.__initPitch, relativerot[0] + self.__initYaw

        elif self.mount == "right":
            x, y, z = relativepos_mm[2] + self.__initX, relativepos_mm[1] + self.__initY, -1 * relativepos_mm[0] + self.__initZ
            roll, pitch, yaw = relativerot[2] + self.__initRoll, relativerot[1] + self.__initPitch, -1 * relativerot[0] + self.__initYaw

        elif self.mount == "flat":
            x, y, z = relativepos_mm[2] + self.__initX, relativepos_mm[0] + self.__initY, relativepos_mm[1] + self.__initZ
            roll, pitch, yaw = relativerot[2] + self.__initRoll, relativerot[0] + self.__initPitch, relativerot[1] + self.__initYaw

        if isOnlyPosition:
            roll, pitch, yaw = self.__initRoll, self.__initPitch, self.__initYaw

        if isLimit:
            if x > self.__maxX:
                x = self.__maxX
            elif x < self.__minX:
                x = self.__minX

            if y > self.__maxY:
                y = self.__maxY
            elif y < self.__minY:
                y = self.__minY

            if z > self.__maxZ:
                z = self.__maxZ
            elif z < self.__minZ:
                z = self.__minZ

            if roll > self.__maxRoll:
                roll = self.__maxRoll
            elif roll < self.__minRoll:
                roll = self.__minRoll

            if pitch > self.__maxPitch:
                pitch = self.__maxPitch
            elif pitch < self.__minPitch:
                pitch = self.__minPitch

            if yaw > self.__maxYaw:
                yaw = self.__maxYaw
            elif yaw < self.__minYaw:
                yaw = self.__minYaw

        return np.array([x, y, z, roll, pitch, yaw])

    def Transform_LPF(self, relativepos: list, relativerot: list, isLimit=True, isOnlyPosition=False):
        """
        Converts from motive coordinate to xarm coordinate and limits values.
        """
        relativepos_mm = relativepos * 1000

        self.beforefilt.append([self.__initX, self.__initY, self.__initZ, self.__initRoll, self.__initPitch, self.__initYaw])
        self.robotfilt = self.filter_robot.lowpass2(self.beforefilt, self.afterfilt)
        self.afterfilt.append(self.robotfilt)
        del self.beforefilt[0]
        del self.afterfilt[0]

        if self.mount == "left":
            x, y, z = self.robotfilt[2] + self.__initX, -1 * self.robotfilt[1] + self.__initY, self.robotfilt[0] + self.__initZ
            roll, pitch, yaw = self.robotfilt[5] + self.__initRoll, -1 * self.robotfilt[4] + self.__initPitch, self.robotfilt[3] + self.__initYaw

        elif self.mount == "right":
            x, y, z = self.robotfilt[2] + self.__initX, self.robotfilt[1] + self.__initY, -1 * self.robotfilt[0] + self.__initZ
            roll, pitch, yaw = self.robotfilt[5] + self.__initRoll, self.robotfilt[4] + self.__initPitch, -1 * self.robotfilt[3] + self.__initYaw

        elif self.mount == "flat":
            x, y, z = self.robotfilt[2] + self.__initX, self.robotfilt[0] + self.__initY, self.robotfilt[1] + self.__initZ
            roll, pitch, yaw = self.robotfilt[5] + self.__initRoll, self.robotfilt[3] + self.__initPitch, self.robotfilt[4] + self.__initYaw

        if isOnlyPosition:
            roll, pitch, yaw = self.__initRoll, self.__initPitch, self.__initYaw

        if isLimit:
            if x > self.__maxX:
                x = self.__maxX
            elif x < self.__minX:
                x = self.__minX

            if y > self.__maxY:
                y = self.__maxY
            elif y < self.__minY:
                y = self.__minY

            if z > self.__maxZ:
                z = self.__maxZ
            elif z < self.__minZ:
                z = self.__minZ

            if roll > self.__maxRoll:
                roll = self.__maxRoll
            elif roll < self.__minRoll:
                roll = self.__minRoll

            if pitch > self.__maxPitch:
                pitch = self.__maxPitch
            elif pitch < self.__minPitch:
                pitch = self.__minPitch

            if yaw > self.__maxYaw:
                yaw = self.__maxYaw
            elif yaw < self.__minYaw:
                yaw = self.__minYaw

        return np.array([x, y, z, roll, pitch, yaw])