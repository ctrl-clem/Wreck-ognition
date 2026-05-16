import numpy as np

HF_REPO_ID = "clemboss17/wreck-ognition-weights"

WEIGHTS_MAP = {
    "post_only": "best_model_post_only_F1_21_04-epochs36-50.pth",
    "post_premask": "best_model_post_premask_F1_21_04_epochs36-50.pth",
    "pre_post": "best_model_pre_post_F1_21_04_epochs36-50.pth",
    "pre_post_premask": "best_model_pre_post_premask_F1_21_04_epochs36-50.pth"

}

DEFAULT_DATA = {
    "pre":"app/default_pictures/hurricane-matthew_00000108_pre_disaster.png",
    "post":"app/default_pictures/hurricane-matthew_00000108_post_disaster.png",
    "premask":"app/default_pictures/hurricane-matthew_00000108_pre_disaster_target.png"
}


CLASS_LABELS = {
    0 : "Background",
    1 : "No-Damage",
    2 : "Minor",
    3 : "Major",
    4 : "Destroyed"
}

CLASS_COLORS = np.array([
    [0, 0, 0],         # 0: Background (Black)
    [0, 255, 0],       # 1: No Damage (Green)
    [255, 255, 0],     # 2: Minor Damage (Yellow)
    [255, 165, 0],     # 3: Major Damage (Orange)
    [255, 0, 0]        # 4: Destroyed (Red)
], dtype=np.uint8)