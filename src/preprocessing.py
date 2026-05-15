import cv2
import numpy as np


def to_grayscale(image):
    """
    Convierte la imagen a escala de grises.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.uint8)


def resize_image(image, width, height):
    """
    Redimensiona la imagen a las dimensiones indicadas.
    """
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return resized.astype(np.uint8)


def apply_blur(image, kernel_size=5):
    """
    Aplica un filtro de suavizado Gaussiano a la imagen.
    """
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return blurred.astype(np.uint8)


def adjust_brightness_contrast(image, alpha=1.0, beta=0):
    """
    Ajusta el brillo y el contraste de la imagen.
    """
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted.astype(np.uint8)


def apply_threshold(image, thresh_value=127):
    """
    Aplica umbralización binaria a una imagen en escala de grises.
    """
    if len(image.shape) != 2:
        raise ValueError("apply_threshold requiere una imagen en escala de grises (1 canal).")

    _, thresholded = cv2.threshold(
        image,
        thresh_value,
        255,
        cv2.THRESH_BINARY
    )

    return thresholded.astype(np.uint8)


def detect_edges(image, low=50, high=150):
    """
    Detecta bordes con el algoritmo de Canny.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    edges = cv2.Canny(gray, low, high)
    return edges.astype(np.uint8)


def full_pipeline(image, target_width=224, target_height=224):
    """
    Ejecuta el pipeline completo de preprocesamiento.
    """
    resized = resize_image(image, target_width, target_height)
    gray = to_grayscale(resized)
    blurred = apply_blur(gray, kernel_size=3)
    edges = detect_edges(blurred, low=50, high=150)

    return edges.astype(np.uint8)