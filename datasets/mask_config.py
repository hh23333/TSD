
PIFPAF_KEYPOINTS = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
                    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
                    "right_knee", "left_ankle", "right_ankle"]

PIFPAF_JOINTS = ["left_ankle_to_left_knee", "left_knee_to_left_hip", "right_ankle_to_right_knee",
                 "right_knee_to_right_hip", "left_hip_to_right_hip", "left_shoulder_to_left_hip",
                 "right_shoulder_to_right_hip", "left_shoulder_to_right_shoulder", "left_shoulder_to_left_elbow",
                 "right_shoulder_to_right_elbow", "left_elbow_to_left_wrist", "right_elbow_to_right_wrist",
                 "left_eye_to_right_eye", "nose_to_left_eye", "nose_to_right_eye", "left_eye_to_left_ear",
                 "right_eye_to_right_ear", "left_ear_to_left_shoulder", "right_ear_to_right_shoulder"]

PIFPAF_PARTS = PIFPAF_KEYPOINTS + PIFPAF_JOINTS

PIFPAF_SINGLE_GROUPS = {k:k for k in PIFPAF_PARTS}
PIFPAF_PARTS_MAP = {k: i for i, k in enumerate(PIFPAF_PARTS)}
EIGHT_SYM_INDEX = [[1,2], [4,5], [6,7]]
EIGHT = {
        "head_mask": ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_eye_to_right_eye",
                                         "nose_to_left_eye", "nose_to_right_eye", "left_eye_to_left_ear",
                                         "right_eye_to_right_ear", "left_ear_to_left_shoulder",
                                         "right_ear_to_right_shoulder"],
        "left_arm_mask": ["left_shoulder", "left_elbow", "left_wrist", "left_shoulder_to_left_elbow",
                                             "left_elbow_to_left_wrist"],
        "right_arm_mask": ["right_shoulder", "right_elbow", "right_wrist", "right_shoulder_to_right_elbow",
                                              "right_elbow_to_right_wrist"],
        "torso_mask": ["left_hip", "right_hip", "left_hip_to_right_hip",
                                          "left_shoulder_to_left_hip", "right_shoulder_to_right_hip",
                                          "left_shoulder_to_right_shoulder"],
        "left_leg_mask": ["left_knee", "left_ankle_to_left_knee",
                                             "left_knee_to_left_hip", "left_hip_to_right_hip"],
        "right_leg_mask": ["right_knee", "right_ankle_to_right_knee",
                                         "right_knee_to_right_hip"],
        "left_feet_mask": ["left_ankle"],
        "right_feet_mask": ["right_ankle"],
        }

FIVE = {
        "head_mask": ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_eye_to_right_eye",
                                         "nose_to_left_eye", "nose_to_right_eye", "left_eye_to_left_ear",
                                         "right_eye_to_right_ear", "left_ear_to_left_shoulder",
                                         "right_ear_to_right_shoulder"],
        "arm_mask": ["left_shoulder", "left_elbow", "left_wrist", "left_shoulder_to_left_elbow",
                                             "left_elbow_to_left_wrist", "right_shoulder", "right_elbow", "right_wrist", "right_shoulder_to_right_elbow",
                                              "right_elbow_to_right_wrist"],
        "torso_mask": ["left_hip", "right_hip", "left_hip_to_right_hip",
                                          "left_shoulder_to_left_hip", "right_shoulder_to_right_hip",
                                          "left_shoulder_to_right_shoulder"],
        "leg_mask": ["left_knee", "left_ankle_to_left_knee",
                                             "left_knee_to_left_hip", "left_hip_to_right_hip", "right_knee", "right_ankle_to_right_knee",
                                         "right_knee_to_right_hip"],
        "feet_mask": ["left_ankle", "right_ankle"],
        }

THREE = {
    "head_mask": ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_eye_to_right_eye",
                            "nose_to_left_eye", "nose_to_right_eye", "left_eye_to_left_ear",
                            "right_eye_to_right_ear", "left_ear_to_left_shoulder",
                            "right_ear_to_right_shoulder"],
    "upper_body_mask": ["left_shoulder", "left_elbow", "left_wrist", "left_shoulder_to_left_elbow",
                                             "left_elbow_to_left_wrist", "right_shoulder", "right_elbow", "right_wrist", "right_shoulder_to_right_elbow",
                                              "right_elbow_to_right_wrist", "left_hip", "right_hip", "left_hip_to_right_hip",
                                          "left_shoulder_to_left_hip", "right_shoulder_to_right_hip",
                                          "left_shoulder_to_right_shoulder"],
    "lower_body_mask": ["left_knee", "left_ankle_to_left_knee",
                                             "left_knee_to_left_hip", "left_hip_to_right_hip", "right_knee", "right_ankle_to_right_knee",
                                         "right_knee_to_right_hip", "left_ankle", "right_ankle"]
}

ONE = {
    "fg": ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_eye_to_right_eye",
                            "nose_to_left_eye", "nose_to_right_eye", "left_eye_to_left_ear",
                            "right_eye_to_right_ear", "left_ear_to_left_shoulder",
                            "right_ear_to_right_shoulder", "left_shoulder", "left_elbow", "left_wrist", "left_shoulder_to_left_elbow",
                                             "left_elbow_to_left_wrist", "right_shoulder", "right_elbow", "right_wrist", "right_shoulder_to_right_elbow",
                                              "right_elbow_to_right_wrist", "left_hip", "right_hip", "left_hip_to_right_hip",
                                          "left_shoulder_to_left_hip", "right_shoulder_to_right_hip",
                                          "left_shoulder_to_right_shoulder", "left_knee", "left_ankle_to_left_knee",
                                             "left_knee_to_left_hip", "left_hip_to_right_hip", "right_knee", "right_ankle_to_right_knee",
                                         "right_knee_to_right_hip", "left_ankle", "right_ankle"]
}

SPLIT_DICT = {
    'eight': EIGHT,
    'five': FIVE,
    'three': THREE,
    'one': ONE
}

SYM_DICT = {
    'eight': EIGHT_SYM_INDEX,
}
    
