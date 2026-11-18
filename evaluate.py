"""Evaluate detection AP on a JSONL manifest at a chosen IoU threshold."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', required=True); parser.add_argument('--manifest', required=True)
    parser.add_argument('--descriptions'); parser.add_argument('--config')
    parser.add_argument('--output', default='runs/evaluation'); parser.add_argument('--device', default='auto')
    parser.add_argument('--iou', type=float, default=0.5)
    parser.add_argument('--confidence', type=float, default=0.001); parser.add_argument('--nms-iou', type=float, default=0.65)
    args = parser.parse_args()
    if not 0 < args.iou <= 1:
        raise ValueError('Evaluation IoU must be in (0, 1].')
    from PIL import Image
    from data_preparation.prepare import CLASSES, read_manifest
    from utils.runtime import load_descriptions, resolve_device
    from utils.inference import load_detector, detect, validate_evaluation_records
    from utils.utils_map import get_map
    rows = read_manifest(args.manifest, check_images=True)
    validate_evaluation_records(rows)
    cfg = json.loads(Path(args.config).read_text(encoding='utf-8-sig')) if args.config else None
    model, cfg = load_detector(args.weights, resolve_device(args.device), cfg)
    texts = load_descriptions(args.descriptions, rows) if cfg['use_prompt_fusion'] else {}
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Choose a new evaluation output directory.')
    (output/'ground-truth').mkdir(parents=True); (output/'detection-results').mkdir()
    for row in rows:
        file_id = hashlib.sha256(row['id'].encode()).hexdigest()[:20]
        with Image.open(row['image']) as image:
            predictions = detect(model, image, row['weather'], texts.get(row['id']), cfg['input_size'], args.confidence, args.nms_iou)
        # The inherited VOC evaluator uses inclusive pixel coordinates.
        with (output/'ground-truth'/f'{file_id}.txt').open('w', encoding='utf-8') as f:
            for key in ['boxes', 'ignore_boxes']:
                for x1, y1, x2, y2, cls in row.get(key, []):
                    suffix = ' difficult' if key == 'ignore_boxes' else ''
                    f.write(f'{CLASSES[int(cls)]} {x1+1} {y1+1} {x2} {y2}{suffix}\n')
        with (output/'detection-results'/f'{file_id}.txt').open('w', encoding='utf-8') as f:
            for top, left, bottom, right, objectness, class_confidence, cls in predictions:
                f.write(f'{CLASSES[int(cls)]} {objectness*class_confidence} {left+1} {top+1} {right} {bottom}\n')
    (output/'evaluation_config.json').write_text(json.dumps({'iou': args.iou, 'confidence': args.confidence, 'nms_iou': args.nms_iou, 'input_size': cfg['input_size'], 'classes': CLASSES}, indent=2), encoding='utf-8')
    get_map(args.iou, False, path=str(output))


if __name__ == '__main__':
    main()
