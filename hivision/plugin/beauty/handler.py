import cv2
from hivision.creator.context import Context
from hivision.plugin.beauty.whitening import make_whitening
from hivision.plugin.beauty.base_adjust import adjust_brightness_contrast_sharpen_saturation

class BaseBeautyFilter:
    def process(self, image, params):
        raise NotImplementedError

class WhiteningFilter(BaseBeautyFilter):
    def process(self, image, params):
        if params.whitening_strength > 0:
            return make_whitening(image, params.whitening_strength), True
        return image, False

class BasicAdjustFilter(BaseBeautyFilter):
    def process(self, image, params):
        if (
            params.brightness_strength != 0
            or params.contrast_strength != 0
            or params.sharpen_strength != 0
            or params.saturation_strength != 0
        ):
            img = adjust_brightness_contrast_sharpen_saturation(
                image,
                params.brightness_strength,
                params.contrast_strength,
                params.sharpen_strength,
                params.saturation_strength,
            )
            return img, True
        return image, False

# Pipeline can easily be expanded in the future
BEAUTY_PIPELINE = [
    WhiteningFilter(),
    BasicAdjustFilter(),
]

def beauty_face(ctx: Context):
    """
    Modular pipeline for face beautification.
    """
    middle_image = ctx.origin_image.copy()
    processed_any = False

    for filter_obj in BEAUTY_PIPELINE:
        middle_image, processed = filter_obj.process(middle_image, ctx.params)
        if processed:
            processed_any = True

    if processed_any:
        b, g, r = cv2.split(middle_image)
        _, _, _, alpha = cv2.split(ctx.matting_image)
        ctx.matting_image = cv2.merge((b, g, r, alpha))
