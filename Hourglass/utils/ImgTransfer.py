from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import random


def Filp(img, heatmap):
    return ImageOps.mirror(img), np.flip(heatmap, axis=2)

def AdjustContrast(img, heatmap):
    return ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.3)), heatmap

def AdjustBrightness(img, heatmap):
    return ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3)), heatmap

def ColorEnhance(img, heatmap):
    return ImageEnhance.Color(img).enhance(random.uniform(0.7, 1.3)), heatmap

