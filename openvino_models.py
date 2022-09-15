#!/usr/bin/env python3
import cv2
import numpy as np
from numpy.lib.stride_tricks import as_strided
from openvino.runtime import Core
from torch import embedding
from OpenPoseDecoder import OpenPoseDecoder
from scipy.spatial import distance
import os


class IntelPreTrainedModel(object):
    def __init__(self, models_dir: str, model_group: str, model_name: str) -> None:
        if models_dir is None:
            if "OPENVINO_DIR" in os.environ: 
                models_dir = os.environ["OPENVINO_DIR"]
            else:
                models_dir = "/home/nathan"

        ie = Core()
        name = model_name
        path = "%s/%s/%s/FP16/%s.xml" % (models_dir, model_group, name, name)
        net = ie.read_model(model=path)
        self.net = ie.compile_model(model=net, device_name="CPU")
    
    def forward(self, inputs):
        return self.net(inputs=[inputs])
        
        
class FacePoseDetection(IntelPreTrainedModel):
	def __init__(self, models_dir: str = None) -> None:
		super().__init__(models_dir, "intel", "head-pose-estimation-adas-0001")
	def forward(self, frame):
		img = frame.copy()
		h, w, c = img.shape
		img = cv2.resize(img, (60, 60))
		img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
		img = np.expand_dims(img.transpose(2, 0, 1), 0)
		
		yaw = super().forward(img)[self.net.output("angle_y_fc")]
		pitch = super().forward(img)[self.net.output("angle_p_fc")]
		roll = super().forward(img)[self.net.output("angle_r_fc")]
		return yaw, pitch, roll

class FaceDetection(IntelPreTrainedModel):
    def __init__(self, models_dir: str = None) -> None:
        super().__init__(models_dir, "intel", "face-detection-adas-0001")

    def forward(self, frame): 
        # (B, C, H, W) => (1, 3, 384, 672) RGB
        img = frame.copy()
        h, w, c = img.shape
        img = cv2.resize(img, (672, 384))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img.transpose(2, 0, 1), 0)

        # (1, 1, 200, 7) [image_id, label, conf, x_min, y_min, x_max, y_max]
        boxes = super().forward(img)[self.net.output("detection_out")][0][0]

        # (N, 4) (x1, y1, x2, y2)
        res = []
        for id, label, conf, x1, y1, x2, y2 in boxes:
            if conf < 0.95: continue
            x1, y1 = int(x1 * w), int(y1 * h)
            x2, y2 = int(x2 * w), int(y2 * h)
            if x1 < 0 or y1 < 0: continue
            if x2 >= w or y2 >= h: continue
            if x1 >= x2 or y1 >= y2: continue
            res.append([x1, y1, x2, y2])
        return res
	
class HumanPoseEstimation(IntelPreTrainedModel):
    points_name = {
        0: "NOSE",
        1: "EYE_L",         2: "EYE_R",
        3: "EAR_L",         4: "EAR_R",
        5: "SHOULDER_L",    6: "SHOULDER_R",
        7: "ELBOW_L",       8: "ELBOW_R",
        9: "WRIST_L",       10:"WRIST_R",
        11:"HIP_L",         12:"HIP_R",
        13:"KNEE_L",        14:"KNEE_R",
        15:"ANKLE_L",       16:"ANKLE_R"
    }
    colors = (
        (255, 0, 0), (255, 0, 255), (170, 0, 255), (255, 0, 85), 
        (255, 0, 170), (85, 255, 0), (255, 170, 0), (0, 255, 0), 
        (255, 255, 0), (0, 255, 85), (170, 255, 0), (0, 85, 255),
        (0, 255, 170), (0, 0, 255), (0, 255, 255), (85, 0, 255), (0, 170, 255))

    default_skeleton = (
        (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), 
        (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), 
        (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6))

    def __init__(self, models_dir: str = None) -> None:
        super().__init__(models_dir, "intel", "human-pose-estimation-0001")
        self.decoder = OpenPoseDecoder()
    
    def forward(self, frame):
        # (B, C, H, W) => (1, 3, 256, 456) BGR
        img = frame.copy()
        h, w, c = img.shape
        img = cv2.resize(img, (456, 256))
        img = np.expand_dims(img.transpose(2, 0, 1), 0)
        out = super().forward(img)
        pafs = out[self.net.output("Mconv7_stage2_L1")]
        heatmaps = out[self.net.output("Mconv7_stage2_L2")]
        poses, scores = self.process_results(frame, pafs, heatmaps)
        return poses

    # 2d pooling in numpy (from: htt11ps://stackoverflow.com/a/54966908/1624463)
    def pool2d(cls, A, kernel_size, stride, padding, pool_mode="max"):
        """
        2D Pooling
        Parameters:
            A: input 2D array
            kernel_size: int, the size of the window
            stride: int, the stride of the window
            padding: int, implicit zero paddings on both sides of the input
            pool_mode: string, 'max' or 'avg'
        """
        # Padding
        A = np.pad(A, padding, mode="constant")

        # Window view of A
        output_shape = (
            (A.shape[0] - kernel_size) // stride + 1,
            (A.shape[1] - kernel_size) // stride + 1,
        )
        kernel_size = (kernel_size, kernel_size)
        A_w = as_strided(
            A,
            shape=output_shape + kernel_size,
            strides=(stride * A.strides[0], stride * A.strides[1]) + A.strides
        )
        A_w = A_w.reshape(-1, *kernel_size)

        # Return the result of pooling
        if pool_mode == "max":
            return A_w.max(axis=(1, 2)).reshape(output_shape)
        elif pool_mode == "avg":
            return A_w.mean(axis=(1, 2)).reshape(output_shape)

    # non maximum suppression
    def heatmap_nms(cls, heatmaps, pooled_heatmaps):
        return heatmaps * (heatmaps == pooled_heatmaps)
        
    # get poses from results
    def process_results(self, img, pafs, heatmaps):
        # this processing comes from
        # https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/common/python/models/open_pose.py
        pooled_heatmaps = np.array(
            [[self.pool2d(h, kernel_size=3, stride=1, padding=1, pool_mode="max") for h in heatmaps[0]]]
        )
        nms_heatmaps = self.heatmap_nms(heatmaps, pooled_heatmaps)

        # decode poses
        poses, scores = self.decoder(heatmaps, nms_heatmaps, pafs)
        output_shape = list(self.net.output(index=0).partial_shape)
        output_scale = img.shape[1] / output_shape[3].get_length(), img.shape[0] / output_shape[2].get_length()
        # multiply coordinates by scaling factor
        poses[:, :, :2] *= output_scale
        return poses, scores

    def draw_poses(cls, img, poses, point_score_threshold, skeleton=default_skeleton):
        if poses.size == 0:
            return img

        img_limbs = np.copy(img)
        for pose in poses:
            points = pose[:, :2].astype(np.int32)
            points_scores = pose[:, 2]
            # Draw joints.
            for i, (p, v) in enumerate(zip(points, points_scores)):
                if v > point_score_threshold:
                    cv2.circle(img, tuple(p), 1, cls.colors[i], 2)
            # Draw limbs.
            for i, j in skeleton:
                if points_scores[i] > point_score_threshold and points_scores[j] > point_score_threshold:
                    cv2.line(img_limbs, tuple(points[i]), tuple(points[j]), color=cls.colors[j], thickness=4)
        cv2.addWeighted(img, 0.4, img_limbs, 0.6, 0, dst=img)
        return img


