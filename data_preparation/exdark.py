"""Convert native ExDark [left, top, width, height] annotations to JSONL."""
import argparse
from collections import defaultdict
from pathlib import Path

from data_preparation.prepare import CLASSES, validate_record, write_manifest

CLASS_MAP = {'People': 'person', 'Bicycle': 'bicycle', 'Car': 'car', 'Motorbike': 'motorbike', 'Bus': 'bus'}


def convert_exdark(image_root, annotation_root, image_list, origin):
    if origin not in [0, 1]:
        raise ValueError('Specify coordinate origin 0 or 1 for your annotation export.')
    images, annotations = defaultdict(list), defaultdict(list)
    for path in Path(image_root).rglob('*'):
        if path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            images[path.name].append(path.resolve())
    for path in Path(annotation_root).rglob('*.txt'):
        annotations[path.name.removesuffix('.txt')].append(path.resolve())
    names = [s.strip() for s in Path(image_list).read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate filenames in the evaluation list.')
    rows = []
    for name in names:
        if len(images[name]) != 1 or len(annotations[name]) != 1:
            raise ValueError(f'{name}: expected one image and one annotation file; check filenames and duplicates.')
        image, annotation = images[name][0], annotations[name][0]
        # Native ExDark folders encode the image-level category.
        if annotation.parent.name not in CLASS_MAP:
            continue
        boxes = []
        for number, line in enumerate(annotation.read_text(encoding='utf-8-sig').splitlines(), 1):
            line = line.strip()
            if not line or line.startswith(('%', '#')):
                continue
            fields = line.split()
            if len(fields) < 5:
                raise ValueError(f'{annotation.name}:{number}: expected class and l,t,w,h.')
            if fields[0] not in CLASS_MAP:
                continue
            left, top, width, height = map(float, fields[1:5])
            left -= origin; top -= origin
            boxes.append([left, top, left+width, top+height, CLASSES.index(CLASS_MAP[fields[0]])])
        row = {'id': name, 'image': str(image), 'weather': 'dark', 'boxes': boxes, 'annotation_origin': origin}
        validate_record(row, check_images=True)
        rows.append(row)
    if not rows:
        raise ValueError('No images from the five selected image-level categories were found.')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', required=True); parser.add_argument('--annotations', required=True)
    parser.add_argument('--list', required=True, help='One exact image filename, including extension, per line.')
    parser.add_argument('--origin', type=int, choices=[0, 1], required=True, help='Coordinate origin used by the annotation export.')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    rows = convert_exdark(args.images, args.annotations, args.list, args.origin)
    write_manifest(args.output, rows)
    print(f'Prepared {len(rows)} ExDark evaluation records from the five image-level categories.')


if __name__ == '__main__':
    main()
