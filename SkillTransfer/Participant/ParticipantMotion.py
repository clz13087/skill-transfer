import csv
import threading
import time
import numpy as np

from Gripper.Gripper import BendingSensorManager
from Filter.Filter import MotionFilter
from OptiTrack.OptiTrackStreaming import OptiTrackStreamingManager
from Gripper.UDP import UDPManager

# ----- Numeric range remapping ----- #
targetMin = 150
targetMax = 850
originalMin = 0
originalMax = 1

class ParticipantMotion:
    def __init__(self, defaultParticipantNum: int, otherRigidBodyNum: int, motionInputSystem: str = "optitrack", mocapServer: str = "", mocapLocal: str = "", idList: list = []) -> None:

        self.defaultParticipantNum = defaultParticipantNum
        self.otherRigidBodyNum = otherRigidBodyNum
        self.motionInputSystem = motionInputSystem
        self.udpManager = None
        self.recordedMotion = {}
        self.recordedGripperValue = {}
        self.recordedMotionLength = []
        self.InitBendingSensorValues = []
        self.idList = idList

        n = 2
        fp = 10
        fs = 700
        self.filter_FB = MotionFilter()
        self.filter_FB.InitLowPassFilterWithOrder(fs, fp, n)

        self.get_gripperValue_1_box = [[0]] * n
        self.get_gripperValue_1_filt_box = [[0]] * n
        self.get_gripperValue_2_box = [[0]] * n
        self.get_gripperValue_2_filt_box = [[0]] * n

        # ----- Initialize participants' motion input system ----- #
        if motionInputSystem == "optitrack":
            self.optiTrackStreamingManager = OptiTrackStreamingManager(defaultParticipantNum=defaultParticipantNum, otherRigidBodyNum=self.otherRigidBodyNum, mocapServer=mocapServer, mocapLocal=mocapLocal, idList=self.idList)

            # ----- Start streaming from OptiTrack ----- #
            streamingThread = threading.Thread(target=self.optiTrackStreamingManager.stream_run)
            streamingThread.setDaemon(True)
            streamingThread.start()

    def SetInitialBendingValue(self):
        """
        Set init bending value
        """

        if self.gripperInputSystem == "bendingsensor":
            self.InitBendingSensorValues = []

            for i in range(self.bendingSensorNum):
                self.InitBendingSensorValues.append(self.bendingSensors[i].bendingValue)

    def LocalPosition(self, loopCount: int = 0):
        """
        Local position

        Parameters
        ----------
        loopCount: (Optional) int
            For recorded motion.
            Count of loop.

        Returns
        ----------
        participants' local position: dict
        {'participant1': [x, y, z]}
        unit: [m]
        """

        dictPos = {}
        if self.motionInputSystem == "optitrack":
            dictPos = self.optiTrackStreamingManager.position

        return dictPos

    def LocalRotation(self, loopCount: int = 0):
        """
        Local rotation

        Parameters
        ----------
        loopCount: (Optional) int
            For recorded motion.
            Count of loop.

        Returns
        ----------
        participants' local rotation: dict
        {'participant1': [x, y, z, w] or [x, y, z]}
        """

        dictRot = {}
        if self.motionInputSystem == "optitrack":
            dictRot = self.optiTrackStreamingManager.rotation

        return dictRot

    def GripperControlValue(self, weight: list, loopCount: int = 0):
        """
        Value for control of the xArm gripper

        Parameters
        ----------
        loopCount: (Optional) int
            For recorded motion.
            Count of loop.

        Returns
        ----------
        Value for control of the xArm gripper: dict
        {'gripperValue1': float value}
        """

        sharedGripper_left = []
        sharedGripper_right = []

        dictGripperValue = {}
        dictbendingVal = {}
            
        for i in range(self.bendingSensorNum):
            dictbendingVal["gripperValue" + str(i + 1)] = self.bendingSensors[i].bendingValue

            if i % 2 == 0:
                sharedGripper_left += dictbendingVal * weight[i]
            
            elif i % 2 == 1:
                sharedGripper_right += dictbendingVal * weight[i]

        GripperValue1 = sharedGripper_left * (targetMax - targetMin) + targetMin
        GripperValue2 = sharedGripper_right * (targetMax - targetMin) + targetMin

        if GripperValue1 > targetMax:
            GripperValue1 = targetMax
        if GripperValue2 > targetMax:
            GripperValue2 = targetMax
        if GripperValue1 < targetMin:
            GripperValue1 = targetMin
        if GripperValue2 < targetMin:
            GripperValue2 = targetMin

        # ----- lowpass filter for left ----- #
        self.get_gripperValue_1_box.append([GripperValue1])
        get_gripperValue_1_filt = self.filter_FB.lowpass2(self.get_gripperValue_1_box, self.get_gripperValue_1_filt_box)
        self.get_gripperValue_1_filt_box.append(get_gripperValue_1_filt)
        del self.get_gripperValue_1_box[0]
        del self.get_gripperValue_1_filt_box[0]

        # ----- lowpass filter for right ----- #
        self.get_gripperValue_2_box.append([GripperValue2])
        get_gripperValue_2_filt = self.filter_FB.lowpass2(self.get_gripperValue_2_box, self.get_gripperValue_2_filt_box)
        self.get_gripperValue_2_filt_box.append(get_gripperValue_2_filt)
        del self.get_gripperValue_2_box[0]
        del self.get_gripperValue_2_filt_box[0]

        dictGripperValue["gripperValue1"] = get_gripperValue_1_filt
        dictGripperValue["gripperValue2"] = get_gripperValue_2_filt

        return dictGripperValue, dictbendingVal

    def calculate_difference(self, participant_positions):
        self.differenceLimit = 0.1
        order_pos = [2, 1, 0]
        participant_positions = self.reorder_and_negate(participant_positions, order_pos, 2, [1, 2])
        learner_left = np.array(participant_positions["participant1"])
        learner_right = np.array(participant_positions["participant2"])
        expert_left = np.array(participant_positions["participant3"])
        expert_right = np.array(participant_positions["participant4"])

        left_diff = np.linalg.norm(learner_left - expert_left)
        right_diff = np.linalg.norm(learner_right - expert_right)
        average_diff = (left_diff + right_diff) / 2

        capped_average_diff = min(average_diff, self.differenceLimit)
        capped_left_diff = min(left_diff, self.differenceLimit)
        capped_right_diff = min(right_diff, self.differenceLimit)

        return capped_average_diff, capped_left_diff, capped_right_diff
    
    def reorder_and_negate(self, data, order, n_keys, negate_idx):
        keys_list = list(data.keys())
        for i in range(n_keys):
            key = keys_list[i]
            data[key] = [data[key][j] for j in order]
            
            # Apply negation based on i (even/odd)
            data[key][negate_idx[i % 2]] = -1 * data[key][negate_idx[i % 2]]
        return data