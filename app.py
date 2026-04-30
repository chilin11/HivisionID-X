import argparse
import os
from demo.processor import IDPhotoProcessor
from demo.ui import create_ui
from hivision.creator.choose_handler import HUMAN_MATTING_MODELS

root_dir = os.path.dirname(os.path.abspath(__file__))

# Add "(很慢)" suffix to slow models for UI display
SLOW_MODELS = ["birefnet-v1-lite", "birefnet-v1"]
HUMAN_MATTING_MODELS_CHOICE = []
for m in HUMAN_MATTING_MODELS:
    if m == "birefnet-portrait":
        HUMAN_MATTING_MODELS_CHOICE.append(f"{m} (最慢，最好)")
    elif any(s in m for s in SLOW_MODELS):
        HUMAN_MATTING_MODELS_CHOICE.append(f"{m} (很慢)")
    else:
        HUMAN_MATTING_MODELS_CHOICE.append(m)

FACE_DETECT_MODELS = ["face++ (联网Online API)", "mtcnn", "retinaface-resnet50"]
FACE_DETECT_MODELS_CHOICE = FACE_DETECT_MODELS

LANGUAGE = ["zh", "en", "ko", "ja"]

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--port", type=int, default=7860, help="The port number of the server"
    )
    argparser.add_argument(
        "--host", type=str, default="0.0.0.0", help="The host of the server"
    )
    argparser.add_argument(
        "--root_path",
        type=str,
        default=None,
        help="The root path of the server, default is None (='/'), e.g. '/myapp'",
    )
    args = argparser.parse_args()

    processor = IDPhotoProcessor()

    demo = create_ui(
        processor,
        root_dir,
        HUMAN_MATTING_MODELS_CHOICE,
        FACE_DETECT_MODELS_CHOICE,
        LANGUAGE,
    )

    # 如果RUN_MODE是Beast，打印已开启野兽模式
    if os.getenv("RUN_MODE") == "beast":
        print("[Beast mode activated.] 已开启野兽模式。")

    demo.launch(
        server_name=args.host,
        server_port=args.port,
        favicon_path=os.path.join(root_dir, "assets/hivision_logo.png"),
        root_path=args.root_path,
    )
