# SHPDet

Soft-hard prompting for adverse-weather object detection.

## Installation

Python 3.10–3.12, PyTorch 2.5.1 and Git:

```bash
pip install -r requirements.txt
```

## Datasets

Download [VOC-FOG, VOC-RAIN and VOC-DARK](https://github.com/hcl1895/SHPDet-Datasets/releases/tag/v1.0) and extract the archives into `data/`.

| Dataset | Directory |
| --- | --- |
| VOC-FOG | `data/voc_fog/` |
| VOC-RAIN | `data/voc_rain/` |
| VOC-DARK | `data/voc_dark/` |
| RTTS | `data/RTTS/` |
| ExDark | `data/ExDark/` |

Create manifests using the supplied splits and paired clean images:

```bash
python -m data_preparation.prepare index --root data --output data/weather
```

Class order: `person, bicycle, car, motorbike, bus`.

All preprocessing scripts are in `data_preparation/`. To generate VOC-RAIN and VOC-DARK from the paired VOC-FOG data:

```bash
python -m data_preparation.prepare synthesize --manifest data/weather/fog/train.jsonl --output data/generated/train
python -m data_preparation.prepare synthesize --manifest data/weather/fog/test.jsonl --output data/generated/test
```

For RTTS, use `data_preparation.prepare prepare-eval`; for ExDark, use `data_preparation.exdark`. Run either command with `--help` for its path arguments.

## Degradation descriptions

Provide an image-ID-keyed description cache, or set `DASHSCOPE_API_KEY` and run:

```bash
python -m data_preparation.generate_descriptions --manifest data/weather/train/manifest.jsonl --output data/descriptions/train.jsonl
```

This command sends images to the configured Qwen3-VL API and uses the account's quota.

## Training

Set `train_manifest`, `train_descriptions` and `output` in `configs/train.json`, then run:

```bash
python train.py --config configs/train.json --device cuda:0
```

For generated images, use `data/generated/train/manifest.jsonl`. Checkpoints are saved to `runs/shpdet/last.pth` by default. Keep the same `input_size` when loading a checkpoint.

## Evaluation

Evaluate each dataset separately. The default metric is mAP@0.5. For VOC-FOG:

```bash
python -m data_preparation.generate_descriptions --manifest data/weather/fog/test.jsonl --output data/descriptions/fog_test.jsonl
python evaluate.py --weights runs/shpdet/last.pth --manifest data/weather/fog/test.jsonl --descriptions data/descriptions/fog_test.jsonl --output runs/voc_fog --device cuda:0
```

Use the corresponding manifest and description cache for VOC-RAIN, VOC-DARK, RTTS or ExDark.

## Acknowledgments

Built on [RDMNet](https://github.com/xfwang23/RDMNet), [TogetherNet](https://github.com/yz-wang/TogetherNet), [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX), [MPRNet](https://github.com/swz30/MPRNet), [CLIP](https://github.com/openai/CLIP) and [Restormer](https://github.com/swz30/Restormer). Included license texts are in [licenses](licenses/).
