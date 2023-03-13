# Taken from https://github.com/QuwsarOhi/BanglaWriting/blob/main/writingMod.py

import cv2
import numpy as np
# For image drawing
from PIL import Image, ImageDraw, ImageFont, ImageEnhance


def change_contrast(img, level):
    # Input is a Image type
    # use Image.fromarray() on numpy
    factor = (259 * (level + 255)) / (255 * (259 - level))
    def contrast(c):
        return 128 + factor * (c - 128)
    return img.point(contrast)


def change_sharpness(img, level):
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(level)
    return img


def adjust(img, blevel, slevel, clevel, colevel):
    # brightness
    benhance = ImageEnhance.Brightness(img)
    img = benhance.enhance(blevel)
    # sharpness
    img = change_sharpness(img, slevel)
    # contrast
    img = change_contrast(img, clevel)
    # color
    cenhance = ImageEnhance.Color(img)
    img = cenhance.enhance(colevel)
    return img


def improveImage(img):
    # Added grayscale step
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    rgb_planes = cv2.split(img)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, 
                                 beta=255, norm_type=cv2.NORM_MINMAX, 
                                 dtype=cv2.CV_8UC1)

        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

    result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)
    return result_norm


def convert_image(img):
    img = improveImage(img)
    img = adjust(Image.fromarray(img), 
                  blevel=0.7, slevel=1, clevel=255, 
                  colevel=1)
    img = np.array(img)
    img = 255 - img
    img = img / 255
    
    return img


# Taken from https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized