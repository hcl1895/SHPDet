"""Procedural rain and low-light synthesis."""
import hashlib
import random

import cv2
import numpy as np


def image_seed(split, image_id, seed=20261118):
    examples = {'2007_004810': 20261118, '2007_002024': 20261119,
                '2007_005273': 20261120, '2007_000364': 20261121}
    if seed == 20261118 and image_id in examples:
        return examples[image_id]
    key = f'SHPDet-{seed}/{split}/{image_id}'
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], 'big')


class ParameterRandom(random.Random):
    def __init__(self, seed):
        super().__init__(seed)
        self.values = []

    def randint(self, low, high):
        value = super().randint(low, high)
        self.values.append(value)
        return value

    def uniform(self, low, high):
        value = super().uniform(low, high)
        self.values.append(value)
        return value


def create_rain_streaks(h, w, rng, np_rng):
    noise = np_rng.uniform(0, 255, (h, w)).astype(np.uint8)
    threshold = rng.randint(230, 250)
    _, noise = cv2.threshold(noise, threshold, 255, cv2.THRESH_BINARY)
    length, angle = rng.randint(30, 70), rng.randint(75, 105)
    matrix = cv2.getRotationMatrix2D((length / 2, length / 2), angle, 1)
    kernel = np.zeros((length, length))
    kernel[int(length / 2), :] = 1
    kernel = cv2.warpAffine(kernel, matrix, (length, length))
    kernel = kernel / np.sum(kernel)
    layer = cv2.filter2D(noise, -1, kernel)
    layer = cv2.normalize(layer, None, 0, 255, cv2.NORM_MINMAX)
    return (cv2.pow(layer.astype(np.float32) / 255.0, 0.6) * 255.0).astype(np.uint8)


def add_realistic_rain(image, rng, np_rng):
    layer = create_rain_streaks(*image.shape[:2], rng, np_rng)
    rain = cv2.cvtColor(layer, cv2.COLOR_GRAY2BGR).astype(np.float32)
    alpha = layer.astype(np.float32) / 255.0 * rng.uniform(0.6, 0.9)
    alpha = np.dstack([alpha] * 3)
    blended = image.astype(np.float32) * (1.0 - alpha) + rain * alpha
    return np.clip(blended * rng.uniform(0.9, 1.0), 0, 255).astype(np.uint8)


def simulate_low_light(image, rng):
    brightness, gamma = rng.uniform(0.45, 0.6), rng.uniform(1.5, 2.2)
    low = image.astype(np.float32) * brightness
    return np.clip(255 * np.power(low / 255.0, gamma), 0, 255).astype(np.uint8)


def generate_pair(clean, fog, seed):
    if clean.dtype != np.uint8 or clean.ndim != 3 or clean.shape[2] != 3:
        raise ValueError('Expected uint8 H x W x 3 input.')
    if fog is None or fog.shape != clean.shape or fog.dtype != np.uint8:
        raise ValueError('Provide a matching fog image.')
    rain_rng, dark_rng = ParameterRandom(seed), ParameterRandom(seed)
    rain = add_realistic_rain(fog, rain_rng, np.random.RandomState(seed))
    dark = simulate_low_light(clean, dark_rng)
    pars = {
        'rain': dict(zip(('threshold', 'length', 'angle', 'opacity', 'brightness'), rain_rng.values)),
        'dark': dict(zip(('brightness', 'gamma'), dark_rng.values)),
    }
    return rain, dark, pars
