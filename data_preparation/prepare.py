"""VOC manifests and weather synthesis. Boxes use zero-based xyxy coordinates."""
import argparse
import json
import os
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import cv2
from PIL import Image
from data_preparation.weather import generate_pair, image_seed

CLASSES = ['person', 'bicycle', 'car', 'motorbike', 'bus']
WEATHERS = ['fog', 'rain', 'dark']


def validate_record(row, check_images=False):
    if not isinstance(row.get('id'), str) or not row['id'].strip():
        raise ValueError('Every record requires a nonempty string id.')
    if not isinstance(row.get('image'), str) or not row['image']:
        raise ValueError(f"{row['id']}: image path is required.")
    if row.get('weather') not in [None, 'clean', *WEATHERS]:
        raise ValueError(f"{row['id']}: unsupported weather.")
    if not isinstance(row.get('boxes'), list):
        raise ValueError(f"{row['id']}: boxes must be a list.")
    size = None
    if check_images:
        with Image.open(row['image']) as im:
            size = im.size
        if row.get('clean_image'):
            with Image.open(row['clean_image']) as im:
                if im.size != size:
                    raise ValueError(f"{row['id']}: clean/degraded image sizes differ.")
    for box in row['boxes'] + row.get('ignore_boxes', []):
        if len(box) != 5 or not np.isfinite(box).all():
            raise ValueError(f"{row['id']}: invalid bounding box.")
        x1, y1, x2, y2, c = box
        if not (0 <= x1 < x2 and 0 <= y1 < y2 and c == int(c) and 0 <= c < len(CLASSES)):
            raise ValueError(f"{row['id']}: invalid bounding box coordinates or class.")
        if size and (x2 > size[0] or y2 > size[1]):
            raise ValueError(f"{row['id']}: bounding box exceeds image bounds.")


def read_manifest(path, check_images=False):
    path = Path(path).resolve()
    rows, seen = [], set()
    for number, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        for key in ['image', 'clean_image', 'depth']:
            if row.get(key):
                row[key] = str((path.parent / row[key]).resolve())
        validate_record(row, check_images)
        if row['id'] in seen:
            raise ValueError(f'{path.name}:{number}: duplicate image id.')
        seen.add(row['id']); rows.append(row)
    if not rows:
        raise ValueError(f'Empty manifest: {path}')
    return rows


def write_manifest(path, rows):
    path = Path(path).resolve()
    serialized, seen = [], set()
    for row in rows:
        validate_record(row)
        if row['id'] in seen:
            raise ValueError(f"Duplicate image id: {row['id']}")
        seen.add(row['id'])
        entry = dict(row)
        for key in ['image', 'clean_image', 'depth']:
            if entry.get(key):
                try:
                    entry[key] = Path(os.path.relpath(Path(entry[key]).resolve(), path.parent)).as_posix()
                except ValueError:
                    entry[key] = Path(entry[key]).resolve().as_posix()
        serialized.append(json.dumps(entry, ensure_ascii=False))
    if not serialized:
        raise ValueError('Refusing to write an empty manifest.')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(serialized) + '\n')


def check_disjoint(train, test):
    def identities(rows):
        return {os.path.normcase(str(Path(r.get('clean_image') or r['image']).resolve())) for r in rows}
    if identities(train) & identities(test):
        raise ValueError('Train/test overlap: the same source image appears in both splits.')


def voc_records(root, split_path, evaluation=False, weather='clean'):
    root = Path(root).resolve()
    ids = [s.strip() for s in Path(split_path).read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate IDs in the split list.')
    rows = []
    for raw_id in ids:
        image_id = Path(raw_id).stem
        if '/' in raw_id or '\\' in raw_id or image_id in ['.', '..']:
            raise ValueError('VOC split files must contain one image ID per line.')
        image = next((root/'JPEGImages'/f'{image_id}{ext}' for ext in ['.jpg', '.png', '.jpeg', '.JPG'] if (root/'JPEGImages'/f'{image_id}{ext}').is_file()), None)
        if image is None:
            raise FileNotFoundError(f'Image not found for {image_id}')
        xml = ET.parse(root/'Annotations'/f'{image_id}.xml').getroot()
        boxes, ignored = [], []
        for obj in xml.findall('object'):
            name = obj.findtext('name', '').strip()
            if name not in CLASSES:
                continue
            b = obj.find('bndbox')
            box = [float(b.findtext(k)) for k in ['xmin', 'ymin', 'xmax', 'ymax']]
            box[0] -= 1; box[1] -= 1
            box.append(CLASSES.index(name))
            (ignored if int(obj.findtext('difficult', '0')) else boxes).append(box)
        row = dict(id=image_id, image=str(image), boxes=boxes, ignore_boxes=ignored, weather=weather)
        if not evaluation:
            row['clean_image'] = str(image)
        validate_record(row, check_images=True)
        rows.append(row)
    return rows


def dataset_records(root, split, weather):
    base = Path(root).resolve() / f'voc_{weather}'
    rows = voc_records(base / split, base / 'ImageSets/Main' / f'{split}.txt', weather=weather)
    for row in rows:
        source_id = row['id']
        row.update(id=f'{weather}/{source_id}', source_id=source_id, split=split,
                   clean_image=str(base / split / 'JPEGImages_clean' / Path(row['image']).name))
        validate_record(row, check_images=True)
    return rows


def prepare_datasets(root, weathers, counts):
    if not weathers or len(set(weathers)) != len(weathers):
        raise ValueError('Choose distinct weather types.')
    data, reference = {}, {}
    for weather in weathers:
        for split, count in zip(['train', 'test'], counts):
            rows = dataset_records(root, split, weather)
            if len(rows) != count:
                raise ValueError(f'{weather}/{split}: expected {count} images, found {len(rows)}.')
            ids = [row['source_id'] for row in rows]
            if split in reference and ids != reference[split]:
                raise ValueError(f'{weather}/{split}: image IDs or order differ between weather splits.')
            reference[split] = ids
            data[weather, split] = rows
    if set(reference['train']) & set(reference['test']):
        raise ValueError('Train/test split image IDs overlap.')
    return data


def synthesize(rows, output, weathers=None, seed=20261118, split=None):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    weathers = weathers or WEATHERS
    if len(set(weathers)) != len(weathers) or any(w not in WEATHERS for w in weathers):
        raise ValueError('Choose distinct weather types: fog, rain, dark.')
    source_ids = [r.get('source_id', r['id']) for r in rows]
    if len(set(source_ids)) != len(source_ids) or any(Path(s).name != s or '\\' in s for s in source_ids):
        raise ValueError('Use one record per source image.')
    for row in rows:
        validate_record(row, check_images=True)
        if row.get('weather') != 'fog' or not row.get('clean_image'):
            raise ValueError('Provide a VOC-FOG manifest with paired clean images.')
    generated = []
    for row in rows:
        clean = row['clean_image']
        x = cv2.imdecode(np.fromfile(clean, dtype=np.uint8), cv2.IMREAD_COLOR)
        fog = cv2.imdecode(np.fromfile(row['image'], dtype=np.uint8), cv2.IMREAD_COLOR)
        source_id = row.get('source_id', row['id'])
        image_split = split or row.get('split', 'train')
        sample_seed = image_seed(image_split, source_id, seed)
        rain, dark, parameters = generate_pair(x, fog, sample_seed)
        for weather in weathers:
            target = output / f'voc_{weather}' / image_split / 'JPEGImages' / f'{source_id}.jpg'
            target.parent.mkdir(parents=True, exist_ok=True)
            if weather == 'fog':
                shutil.copyfile(row['image'], target)
                pars = {'source': 'existing'}
            else:
                y = rain if weather == 'rain' else dark
                pars = parameters[weather]
                ok, encoded = cv2.imencode('.jpg', y, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if not ok:
                    raise RuntimeError(f'Failed to encode {source_id}.')
                with target.open('xb') as f:
                    f.write(encoded.tobytes())
            entry = dict(row, id=f'{weather}/{source_id}', image=str(target), clean_image=clean,
                         weather=weather, parameters=pars, seed=sample_seed,
                         source_id=source_id, split=image_split)
            generated.append(entry)
    return generated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('index', help='Create manifests from the extracted VOC weather datasets.')
    p.add_argument('--root', required=True); p.add_argument('--output', required=True)
    p.add_argument('--weathers', nargs='+', choices=WEATHERS, default=WEATHERS)
    p.add_argument('--expected-counts', nargs=2, type=int, default=[9578, 2129], metavar=('TRAIN', 'TEST'))
    p = sub.add_parser('prepare-eval', help='Convert a VOC-format split, e.g. RTTS.')
    p.add_argument('--voc-root', required=True); p.add_argument('--list', required=True)
    p.add_argument('--weather', choices=WEATHERS, required=True); p.add_argument('--output', required=True)
    p = sub.add_parser('synthesize', help='Generate VOC-RAIN and VOC-DARK; retain existing fog images.')
    p.add_argument('--manifest', required=True); p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=20261118)
    p.add_argument('--weathers', nargs='+', choices=WEATHERS)
    p.add_argument('--split', choices=['train', 'test'])
    p = sub.add_parser('validate', help='Check images, boxes and pairs.')
    p.add_argument('--manifest', required=True); p.add_argument('--against')
    args = parser.parse_args()
    if args.command == 'index':
        out = Path(args.output)
        if out.exists():
            raise FileExistsError('Choose a new output directory.')
        data = prepare_datasets(args.root, args.weathers, args.expected_counts)
        for split in ['train', 'test']:
            combined = []
            for weather in args.weathers:
                write_manifest(out/weather/f'{split}.jsonl', data[weather, split])
                combined.extend(data[weather, split])
            write_manifest(out/split/'manifest.jsonl', combined)
            print(f'{split}: {len(combined)} records.')
    elif args.command == 'prepare-eval':
        rows = voc_records(args.voc_root, args.list, evaluation=True, weather=args.weather)
        write_manifest(args.output, rows); print(f'Prepared {len(rows)} records.')
    elif args.command == 'synthesize':
        rows = synthesize(read_manifest(args.manifest), args.output, args.weathers, args.seed, args.split)
        write_manifest(Path(args.output)/'manifest.jsonl', rows)
        print(f'Generated {len(rows)} images in {args.output}.')
    else:
        rows = read_manifest(args.manifest, check_images=True)
        if args.against:
            check_disjoint(rows, read_manifest(args.against, check_images=True))
        print(f'Validated {len(rows)} records.')


if __name__ == '__main__':
    main()
