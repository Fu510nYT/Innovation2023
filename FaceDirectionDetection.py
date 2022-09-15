#!/usr/bin/env python3
import rospy
from openvino_models import *
import cv2


if __name__ == "__main__":
	rospy.init_node("FaceDirectionDetection")
	rospy.loginfo("FaceDirectionDetection Node Start!")
	
	cap = cv2.VideoCapture(0)
	head_pose_estimation = FacePoseDetection()
	Face_Detection = FaceDetection()
	
	while not rospy.is_shutdown():
		rospy.Rate(20).sleep()
		_, frame = cap.read()
		
		boxes = Face_Detection.forward(frame=frame)
		
		for x1, y1, x2, y2 in boxes:
			face = frame[y1:y2, x1:x2, :]
			yaw, pitch, roll = head_pose_estimation.forward(face)
			yaw = -1 *(yaw - 1)
			print(f"Yaw: {yaw} | Pitch: {pitch} | Roll: {roll}")
			cv2.putText(frame, "%.2f" % yaw, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 10)
		cv2.imshow("frame", frame)
		cv2.imshow("face", face)
		if cv2.waitKey(1) in [27, ord('q')]:
			break
			
	cv2.destroyAllWindows()
		
		
		
		
