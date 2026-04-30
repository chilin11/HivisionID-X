import functools
from hivision.creator.human_matting import extract_human, MATTING_MODELS
from hivision.creator.face_detector import (
    detect_face_mtcnn,
    detect_face_face_plusplus,
    detect_face_retinaface,
)

HUMAN_MATTING_MODELS = list(MATTING_MODELS.keys())

FACE_DETECT_MODELS = ["face++ (联网Online API)", "mtcnn", "retinaface-resnet50"]

def choose_handler(creator, matting_model_option=None, face_detect_option=None):
    # Strip UI suffix (e.g. " (很慢)")
    if matting_model_option:
        matting_model_option = matting_model_option.split(" ")[0]
        
    # Set Matting Handler
    if matting_model_option in MATTING_MODELS:
        creator.matting_handler = functools.partial(extract_human, model_name=matting_model_option)
    else:
        # Fallback to default
        creator.matting_handler = functools.partial(extract_human, model_name="birefnet-v1")

    # Set Detection Handler
    if (
        face_detect_option == "face_plusplus"
        or face_detect_option == "face++ (联网Online API)"
    ):
        creator.detection_handler = detect_face_face_plusplus
    elif face_detect_option == "retinaface-resnet50":
        creator.detection_handler = detect_face_retinaface
    else:
        creator.detection_handler = detect_face_mtcnn
