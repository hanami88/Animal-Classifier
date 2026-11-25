import cv2
import numpy as np
from PIL import Image

def get_raw_view(pil_img: Image.Image) -> Image.Image:
    return pil_img.copy()

def get_denoise_view(pil_img: Image.Image) -> Image.Image:
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    denoised_bgr = cv2.bilateralFilter(img_bgr, d=7, sigmaColor=50, sigmaSpace=50)
    return Image.fromarray(cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB))

def get_clahe_view(pil_img: Image.Image) -> Image.Image:
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    clahe_bgr = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
    return Image.fromarray(cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2RGB))

def get_optimal_view(pil_img: Image.Image) -> Image.Image:
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    denoised_bgr = cv2.bilateralFilter(img_bgr, d=7, sigmaColor=50, sigmaSpace=50)
    lab = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    clahe_bgr = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
    return Image.fromarray(cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2RGB))

def get_geom_view(pil_img: Image.Image) -> Image.Image:
    optimal_img = get_optimal_view(pil_img)
    return optimal_img.transpose(Image.FLIP_LEFT_RIGHT).rotate(15)

