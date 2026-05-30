from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import FileResponse
from hivision import IDCreator
from hivision.error import FaceError
from hivision.creator.layout_calculator import (
    generate_layout_array,
    generate_layout_image,
)
from hivision.creator.choose_handler import choose_handler
from hivision.utils import (
    add_background,
    resize_image_to_kb,
    bytes_2_base64,
    base64_2_numpy,
    hex_to_rgb,
    add_watermark,
    save_image_dpi_to_bytes,
)
import numpy as np
import cv2
from starlette.middleware.cors import CORSMiddleware
from starlette.formparsers import MultiPartParser
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse, HTMLResponse
import os

MultiPartParser.max_part_size = 100 * 1024 * 1024
MultiPartParser.max_file_size = 100 * 1024 * 1024

app = FastAPI()
creator = IDCreator()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


@app.get("/favicon.ico")
async def favicon():
    path = os.path.join(ROOT_DIR, "assets", "hivision_logo.png")
    if os.path.exists(path):
        return FileResponse(path, media_type="image/png")
    return {"detail": "Not Found"}


@app.post("/idphoto")
async def idphoto_inference(
    input_image: UploadFile = File(None),
    input_image_base64: str = Form(None),
    height: int = Form(413),
    width: int = Form(295),
    human_matting_model: str = Form("birefnet-v1"),
    face_detect_model: str = Form("mtcnn"),
    dpi: int = Form(300),
    face_align: bool = Form(False),
    whitening_strength: int = Form(0),
    head_measure_ratio: float = Form(0.2),
    head_height_ratio: float = Form(0.45),
    top_distance_max: float = Form(0.12),
    top_distance_min: float = Form(0.10),
    brightness_strength: float = Form(0),
    contrast_strength: float = Form(0),
    sharpen_strength: float = Form(0),
    saturation_strength: float = Form(0),
):
    if input_image_base64:
        img = base64_2_numpy(input_image_base64)
    else:
        image_bytes = await input_image.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    choose_handler(creator, human_matting_model, face_detect_model)

    size = (int(height), int(width))
    try:
        result = creator(
            img,
            size=size,
            head_measure_ratio=head_measure_ratio,
            head_height_ratio=head_height_ratio,
            head_top_range=(top_distance_max, top_distance_min),
            face_alignment=face_align,
            whitening_strength=whitening_strength,
            brightness_strength=brightness_strength,
            contrast_strength=contrast_strength,
            sharpen_strength=sharpen_strength,
            saturation_strength=saturation_strength,
        )
    except FaceError:
        result_message = {"status": False, "error": "Face not detected"}
    except FileNotFoundError as e:
        result_message = {"status": False, "error": f"Model file missing: {str(e)}"}
    except Exception as e:
        result_message = {"status": False, "error": str(e)}
    else:
        result_image_hd_bytes = save_image_dpi_to_bytes(result.hd, None, dpi)
        result_message = {
            "status": True,
            "image_base64_hd": bytes_2_base64(result_image_hd_bytes),
        }

    return result_message


@app.post("/human_matting")
async def human_matting_inference(
    input_image: UploadFile = File(None),
    input_image_base64: str = Form(None),
    human_matting_model: str = Form("hivision_modnet"),
    dpi: int = Form(300),
):
    if input_image_base64:
        img = base64_2_numpy(input_image_base64)
    else:
        image_bytes = await input_image.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    choose_handler(creator, human_matting_model, None)

    try:
        result = creator(img, change_bg_only=True)
    except FaceError:
        result_message = {"status": False}
    else:
        result_image_standard_bytes = save_image_dpi_to_bytes(
            cv2.cvtColor(result.standard, cv2.COLOR_RGBA2BGRA), None, dpi
        )
        result_message = {
            "status": True,
            "image_base64": bytes_2_base64(result_image_standard_bytes),
        }
    return result_message


@app.post("/add_background")
async def photo_add_background(
    input_image: UploadFile = File(None),
    input_image_base64: str = Form(None),
    color: str = Form("000000"),
    kb: int = Form(None),
    dpi: int = Form(300),
    render: int = Form(0),
):
    render_choice = ["pure_color", "updown_gradient", "center_gradient"]

    if input_image_base64:
        img = base64_2_numpy(input_image_base64)
    else:
        image_bytes = await input_image.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    color = hex_to_rgb(color)
    color = (color[2], color[1], color[0])

    result_image = add_background(img, bgr=color, mode=render_choice[render]).astype(
        np.uint8
    )

    result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
    if kb:
        result_image_bytes = resize_image_to_kb(result_image, None, int(kb), dpi=dpi)
    else:
        result_image_bytes = save_image_dpi_to_bytes(result_image, None, dpi=dpi)

    return {
        "status": True,
        "image_base64": bytes_2_base64(result_image_bytes),
    }


@app.post("/generate_layout_photos")
async def generate_layout_photos(
    input_image: UploadFile = File(None),
    input_image_base64: str = Form(None),
    height: int = Form(413),
    width: int = Form(295),
    kb: int = Form(None),
    dpi: int = Form(300),
    keep_original_size: bool = Form(False),
    layout_height: int = Form(1205),
    layout_width: int = Form(1795),
    crop_line: bool = Form(False),
):
    if input_image_base64:
        img = base64_2_numpy(input_image_base64)
    else:
        image_bytes = await input_image.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    size = (int(height), int(width))

    typography_arr, typography_rotate = generate_layout_array(
        input_height=size[0],
        input_width=size[1],
        LAYOUT_HEIGHT=layout_height,
        LAYOUT_WIDTH=layout_width,
    )

    result_layout_image = generate_layout_image(
        img,
        typography_arr,
        typography_rotate,
        height=size[0],
        width=size[1],
        crop_line=crop_line,
        LAYOUT_HEIGHT=layout_height,
        LAYOUT_WIDTH=layout_width,
        keep_original_size=keep_original_size,
    ).astype(np.uint8)

    result_layout_image = cv2.cvtColor(result_layout_image, cv2.COLOR_RGB2BGR)
    if kb:
        result_layout_image_bytes = resize_image_to_kb(
            result_layout_image, None, int(kb), dpi=dpi
        )
    else:
        result_layout_image_bytes = save_image_dpi_to_bytes(
            result_layout_image, None, dpi=dpi
        )

    return {
        "status": True,
        "image_base64": bytes_2_base64(result_layout_image_bytes),
    }


@app.post("/watermark")
async def watermark(
    input_image: UploadFile = File(None),
    input_image_base64: str = Form(None),
    text: str = Form("Hello"),
    size: int = Form(20),
    opacity: float = Form(0.5),
    angle: int = Form(30),
    color: str = Form("#000000"),
    space: int = Form(25),
    kb: int = Form(None),
    dpi: int = Form(300),
):
    if input_image_base64:
        img = base64_2_numpy(input_image_base64)
    else:
        image_bytes = await input_image.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    try:
        result_image = add_watermark(img, text, size, opacity, angle, color, space)
        result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
        if kb:
            result_image_bytes = resize_image_to_kb(result_image, None, int(kb), dpi=dpi)
        else:
            result_image_bytes = save_image_dpi_to_bytes(result_image, None, dpi=dpi)
        return {
            "status": True,
            "image_base64": bytes_2_base64(result_image_bytes),
        }
    except Exception as e:
        return {"status": False, "error": str(e)}


@app.post("/set_kb")
async def set_kb(
    input_image: UploadFile = File(None),
    input_image_base64: str = Form(None),
    dpi: int = Form(300),
    kb: int = Form(50),
):
    if input_image_base64:
        img = base64_2_numpy(input_image_base64)
    else:
        image_bytes = await input_image.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    try:
        result_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        result_image_bytes = resize_image_to_kb(result_image, None, int(kb), dpi=dpi)
        return {
            "status": True,
            "image_base64": bytes_2_base64(result_image_bytes),
        }
    except Exception as e:
        return {"status": False, "error": str(e)}


@app.post("/idphoto_crop")
async def idphoto_crop_inference(
    input_image: UploadFile = File(None),
    input_image_base64: str = Form(None),
    height: int = Form(413),
    width: int = Form(295),
    face_detect_model: str = Form("mtcnn"),
    hd: bool = Form(True),
    dpi: int = Form(300),
    head_measure_ratio: float = Form(0.2),
    head_height_ratio: float = Form(0.45),
    top_distance_max: float = Form(0.12),
    top_distance_min: float = Form(0.10),
):
    if input_image_base64:
        img = base64_2_numpy(input_image_base64)
    else:
        image_bytes = await input_image.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    choose_handler(creator, face_detect_option=face_detect_model)

    size = (int(height), int(width))
    try:
        result = creator(
            img,
            size=size,
            head_measure_ratio=head_measure_ratio,
            head_height_ratio=head_height_ratio,
            head_top_range=(top_distance_max, top_distance_min),
            crop_only=True,
        )
    except FaceError:
        result_message = {"status": False}
    else:
        result_image_standard_bytes = save_image_dpi_to_bytes(
            cv2.cvtColor(result.standard, cv2.COLOR_RGBA2BGRA), None, dpi
        )
        result_message = {
            "status": True,
            "image_base64_standard": bytes_2_base64(result_image_standard_bytes),
        }
        if hd:
            result_image_hd_bytes = save_image_dpi_to_bytes(
                cv2.cvtColor(result.hd, cv2.COLOR_RGBA2BGRA), None, dpi
            )
            result_message["image_base64_hd"] = bytes_2_base64(result_image_hd_bytes)

    return result_message


web_ui_path = os.path.join(os.path.dirname(__file__), "web-ui", "dist")
index_html = os.path.join(web_ui_path, "index.html")


@app.get("/")
async def serve_index():
    if os.path.exists(index_html):
        return FileResponse(index_html)
    return HTMLResponse("<h1>Web UI not found</h1><p>Run from the project root.</p>")


# Serve other static assets (css, js, images) under /static/
if os.path.exists(web_ui_path):
    app.mount("/static", StaticFiles(directory=web_ui_path), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
