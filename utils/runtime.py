"""Shared configuration, descriptions, checkpoints and model construction."""
import json
from pathlib import Path

import torch

from data_preparation.prepare import CLASSES, WEATHERS

MODEL_VERSION = 'shpdet-paper-v2'

def validate_model_config(config):
    if config.get('phi', 's') != 's':
        raise ValueError('This restoration backbone supports phi="s" (YOLOX-s).')
    if type(config.get('use_prompt_fusion', True)) is not bool:
        raise ValueError('use_prompt_fusion must be a JSON boolean.')
    size = config.get('input_size', 640)
    if not isinstance(size, int) or size < 128 or size % 32:
        raise ValueError('The backbone requires input size >= 128 and divisible by 32.')


def build_model(config, device='cpu'):
    validate_model_config(config)
    from nets.yolo import YoloBody
    return YoloBody(len(CLASSES), config.get('phi', 's'), use_prompt_fusion=config.get('use_prompt_fusion', True), input_size=config.get('input_size', 640)).to(device)


def build_optimizer(model, lr, weight_decay=5e-4, momentum=0.937):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (no_decay if parameter.ndim < 2 or name.endswith('.bias') else decay).append(parameter)
    return torch.optim.SGD([{'params': decay, 'weight_decay': weight_decay}, {'params': no_decay, 'weight_decay': 0.0}], lr=lr, momentum=momentum, nesterov=True)


def load_checkpoint(model, path, device, for_training=False):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {path}. Set --weights to a trained checkpoint.')
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and checkpoint.get('model_version', MODEL_VERSION) != MODEL_VERSION:
        raise ValueError('Checkpoint architecture version does not match shpdet-paper-v2 (native prompts, MPRNet CAB, YOLOX-s). Train a new checkpoint; old weights cannot be loaded into this architecture.')
    state = checkpoint.get('model', checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if any(key.startswith('UDE.') for key in state):
        raise ValueError('Legacy contrastive checkpoints do not match the manuscript-aligned architecture.')
    model.load_state_dict(state, strict=True)
    return checkpoint


def load_descriptions(path, records):
    if not path:
        raise ValueError('Set the descriptions JSONL path when prompt fusion is enabled.')
    descriptions = {}
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        image_id, text = row['id'], row['text']
        if image_id in descriptions or not isinstance(text, str) or not text.strip():
            raise ValueError('Description IDs must be unique and text must be nonempty.')
        descriptions[image_id] = text.strip()
    missing = [r['id'] for r in records if r['id'] not in descriptions]
    if missing:
        raise ValueError(f'{len(missing)} images have no description; first ID: {missing[0]}')
    return descriptions


def category_vector(weather):
    if weather not in WEATHERS:
        raise ValueError('Weather must be fog, rain or dark.')
    vector = [0.0, 0.0, 0.0]
    vector[WEATHERS.index(weather)] = 1.0
    return vector


def resolve_device(value):
    if value == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(value)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Use --device cpu or a CUDA-enabled PyTorch installation.')
    return device
