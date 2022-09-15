#!/usr/bin/env python3
import rospy
import openvino
from cv_bridge import CvBridge
import cv2
from openvino_models import *
from rospkg import RosPack

cap = cv2.VideoCapture(0)

if __name__ == "__main__":
	rospy.init_node("ScienceFair2023")
	rospy.loginfo("Science Fair 2023 Node Start!")
	
	dnn_pose = HumanPoseEstimation()
	rospy.loginfo("Entering Main Loop")
	while not rospy.is_shutdown():
	
		rospy.Rate(20).sleep()
		_, _frame = cap.read()
		img = _frame.copy()
		
		poses = dnn_pose.forward(img)
		img = dnn_pose.draw_poses(img, poses, 0.1)
		
		rospy.loginfo("Calculating poses")
		print(poses)
		
		for pose in poses:
			for i, p in enumerate(pose):
				x, y, c = map(int, p)
				cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
				if pose[0][0] != 0 and pose[0][1] != 0:
					cv2.circle(img, (int(pose[9][0]), int(pose[9][1])), 10, (255, 0, 0), -1) #Left Hand
					cv2.circle(img, (int(pose[10][0]), int(pose[10][1])), 10, (0, 255, 0), -1) #Right Hand
					cv2.circle(img, (int(pose[0][0]), int(pose[0][1])), 10, (0, 255, 0), -1)
					
					if int(pose[10][1]) < int(pose[0][1]) and int(pose[10][1]):
						cv2.putText(img, 'Raising', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
					if int(pose[9][1] < int(pose[0][1])) and int(pose[9][1]):
						cv2.putText(img, 'Raising', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
		cv2.imshow("frame", img)
		if cv2.waitKey(1) in [27, ord('q')]:
			break
			
