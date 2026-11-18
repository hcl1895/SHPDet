"""Shared inference path for prediction and evaluation."""
from pathlib import Path
import numpy as np
from PIL import Image
import torch

from utils.runtime import build_model, load_checkpoint, category_vector
from utils.utils import resize_image, preprocess_input
from utils.utils_bbox import decode_outputs, non_max_suppression


def validate_evaluation_records(records):
    if not any(row['boxes'] for row in records):
        raise ValueError('Evaluation requires at least one non-difficult object from the selected categories.')
    for row in records:
        category_vector(row.get('weather'))


def load_detector(weights, device, config=None):
    if not Path(weights).is_file():
        raise FileNotFoundError(f'Checkpoint not found: {weights}')
    saved = torch.load(weights, map_location='cpu', weights_only=True)
    cfg = config or saved.get('config')
    if cfg is None:
        raise ValueError('A raw state_dict requires --config with its training configuration.')
    model = build_model(cfg, device)
    load_checkpoint(model, weights, device)
    model.eval()
    return model, cfg


@torch.no_grad()
def detect(model, image, weather, description, size=640, confidence=0.001, nms_iou=0.65):
    if size < 128 or size % 32:
        raise ValueError('The backbone requires input size >= 128 and divisible by 32.')
    if not 0 <= confidence <= 1 or not 0 <= nms_iou <= 1:
        raise ValueError('Confidence and NMS IoU must be in [0, 1].')
    image = image.convert('RGB')
    shape = np.array([image.height, image.width])
    resized = resize_image(image, (size, size), True)
    array = np.transpose(preprocess_input(np.asarray(resized, dtype=np.float32).copy()), (2, 0, 1))[None]
    tensor = torch.as_tensor(array, device=next(model.parameters()).device)
    outputs, _ = model(tensor, weather_texts=[description] if description else None, category_vectors=[category_vector(weather)])
    decoded = decode_outputs(outputs, [size, size])
    results = non_max_suppression(decoded, 5, [size, size], shape, True, confidence, nms_iou)[0]
    return [] if results is None else results.tolist()
