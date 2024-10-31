from email.policy import default
from hashlib import new
from threading import local
from . import NatNetClient
import numpy as np

serverAddress = ''
localAddress = ''

class OptiTrackStreamingManager:
	# ---------- Variables ---------- #
	position = {}	# dict { 'ParticipantN': [x, y, z] }. 	N is the number of participants' rigid body. Unit = [m]
	rotation = {}	# dict { 'ParticipantN': [x, y, z, w]}. N is the number of participants' rigid body

	def __init__(self, defaultParticipantNum: int = 2, otherRigidBodyNum: int = 0, mocapServer: str = '', mocapLocal: str = '', idList: list = []):
		global serverAddress
		global localAddress
		self.defaultParticipanNum = defaultParticipantNum
		serverAddress = mocapServer
		localAddress = mocapLocal
		self.idList = idList[0]

		for i in range(defaultParticipantNum):
			self.position['participant'+str(i+1)] = np.zeros(3)
			self.rotation['participant'+str(i+1)] = np.array([0, 0, 0, 1])

		for i in range(otherRigidBodyNum):
			self.position['otherRigidBody'+str(i+1)] = np.zeros(3)
			self.rotation['otherRigidBody'+str(i+1)] = np.array([0, 0, 0, 1])


	# This is a callback function that gets connected to the NatNet client and called once per mocap frame.
	def receive_new_frame(self, data_dict):
		order_list=[ "frameNumber", "markerSetCount", "unlabeledMarkersCount", "rigidBodyCount", "skeletonCount",
					"labeledMarkerCount", "timecode", "timecodeSub", "timestamp", "isRecording", "trackedModelsChanged" ]
		dump_args = False
		if dump_args == True:
			out_string = "    "
			for key in data_dict:
				out_string += key + "="
				if key in data_dict :
					out_string += data_dict[key] + " "
				out_string+="/"
			print(out_string)

	# This is a callback function that gets connected to the NatNet client. It is called once per rigid body per frame
	def receive_rigid_body_frame( self, new_id, position, rotation ):
		"""
		Receives the position and rotation of the active RigidBody.
		Position: [x, y, z], Unit = [m]
		Rotation: [x, y, z, w]

		Parameters
		----------
		new_id: int
			RigidBody id
		position: array
			Position
		rotation: array
			Rotation
		"""
		if str(new_id) == self.idList[1]:
			self.position["participant1"] = np.array(position)
			self.rotation["participant1"] = np.array(rotation)
		elif str(new_id) == self.idList[2]:
			self.position["participant2"] = np.array(position)
			self.rotation["participant2"] = np.array(rotation)

		# if new_id > self.defaultParticipanNum:
		# 	self.position['otherRigidBody'+str(new_id - self.defaultParticipanNum)] = np.array(position)
		# 	self.rotation['otherRigidBody'+str(new_id - self.defaultParticipanNum)] = np.array(rotation)

	def stream_run(self):
		streamingClient = NatNetClient.NatNetClient(serverIP=serverAddress, localIP=localAddress)
		
		# Configure the streaming client to call our rigid body handler on the emulator to send data out.
		streamingClient.new_frame_listener = self.receive_new_frame
		streamingClient.rigid_body_listener = self.receive_rigid_body_frame
		streamingClient.run()