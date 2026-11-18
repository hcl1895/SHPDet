"""Train SHPDet from paired JSONL manifests."""
import argparse
import json
from pathlib import Path
import random


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/train.json')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--weights', help='Initialize from a compatible checkpoint; otherwise start a new run.')
    parser.add_argument('--check-data', action='store_true', help='Validate inputs without starting training.')
    args = parser.parse_args()
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from nets.yolo_training import YOLOLoss, weights_init
    from nets.losses import restoration_loss, restoration_weight
    from data_preparation.prepare import read_manifest, check_disjoint
    from utils.dataset import PairedDataset, collate
    from utils.runtime import MODEL_VERSION, build_model, build_optimizer, load_checkpoint, load_descriptions, resolve_device, validate_model_config

    cfg = json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
    validate_model_config(cfg)
    if cfg['epochs'] < 1 or cfg['workers'] < 0 or not np.isfinite(cfg['learning_rate']) or cfg['learning_rate'] <= 0:
        raise ValueError('Use positive epochs and learning rate, and nonnegative workers.')
    train_records = read_manifest(cfg['train_manifest'], check_images=True)
    val_records = read_manifest(cfg['val_manifest'], check_images=True) if cfg.get('val_manifest') else []
    if val_records:
        check_disjoint(train_records, val_records)
    use_prompts = cfg['use_prompt_fusion']
    train_texts = load_descriptions(cfg['train_descriptions'], train_records) if use_prompts else {}
    val_texts = load_descriptions(cfg['val_descriptions'], val_records) if use_prompts and val_records else {}
    train_set = PairedDataset(cfg['train_manifest'], train_texts, cfg['input_size'], train=True)
    val_set = PairedDataset(cfg['val_manifest'], val_texts, cfg['input_size'], train=False) if val_records else None
    if not isinstance(cfg['batch_size'], int) or cfg['batch_size'] < 1:
        raise ValueError('Batch size must be a positive integer.')
    if 'degradation_contrast' in cfg['loss']:
        raise ValueError('Use the PPC configuration supplied with this model version.')
    if any(not np.isfinite(cfg['loss'][k]) or cfg['loss'][k] < 0 for k in ('detection', 'ppc')):
        raise ValueError('Loss weights must be finite and nonnegative.')
    if len(train_set) < cfg['batch_size']:
        raise ValueError('Training split is smaller than one complete batch.')
    if args.check_data:
        print(f'Validated {len(train_set)} training / {len(val_set) if val_set is not None else 0} validation records.'); return
    output = Path(cfg['output'])
    if output.exists():
        raise FileExistsError('Choose a new output directory to keep previous runs intact.')
    device = resolve_device(args.device)
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg['seed'])
    model = build_model(cfg, device)
    weights_init(model)
    if args.weights:
        load_checkpoint(model, args.weights, device, for_training=True)
    if use_prompts:
        model._init_clip_encoder()
    optimizer = build_optimizer(model, cfg['learning_rate'], cfg['weight_decay'], cfg['momentum'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['epochs'], eta_min=cfg['learning_rate'] * 0.01)
    detector_loss = YOLOLoss(5)
    loader_args = dict(batch_size=cfg['batch_size'], num_workers=cfg['workers'], collate_fn=collate, pin_memory=device.type == 'cuda')
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args) if val_set is not None else None
    output.mkdir(parents=True)
    (output/'config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    best = float('inf')
    for epoch in range(cfg['epochs']):
        model.train()
        totals = {'det': 0., 'restoration': 0., 'ppc': 0.}
        for images, targets, clean, texts, categories in train_loader:
            images = torch.as_tensor(images, device=device)
            clean = torch.as_tensor(clean, device=device)
            targets = [torch.as_tensor(t, device=device) for t in targets]
            optimizer.zero_grad(set_to_none=True)
            detected, restored, ppc = model(images, weather_texts=texts, category_vectors=categories)
            det = detector_loss(detected, targets)
            restoration = restoration_loss(restored, clean)
            loss = cfg['loss']['detection'] * det + restoration_weight(epoch) * restoration + cfg['loss']['ppc'] * ppc
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss at epoch {epoch+1}.')
            loss.backward(); optimizer.step()
            for key, value in [('det', det), ('restoration', restoration), ('ppc', ppc)]:
                totals[key] += float(value.detach())
        model.eval(); val_loss = 0.; val_count = 0
        with torch.no_grad():
            for images, targets, _, texts, categories in val_loader if val_loader is not None else []:
                detected, _ = model(torch.as_tensor(images, device=device), weather_texts=texts, category_vectors=categories)
                value = detector_loss(detected, [torch.as_tensor(t, device=device) for t in targets])
                val_loss += float(value) * len(images); val_count += len(images)
        val_loss = val_loss / val_count if val_count else None
        summary = {'epoch': epoch + 1, **{k: v/len(train_loader) for k, v in totals.items()}, 'val_detection_loss': val_loss, 'lr': optimizer.param_groups[0]['lr']}
        with (output/'metrics.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(summary) + '\n')
        scheduler.step()
        checkpoint = {'model_version': MODEL_VERSION, 'model': model.state_dict(), 'config': cfg, 'epoch': epoch+1, 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}
        torch.save(checkpoint, output/'last.pth')
        if val_loss is not None and val_loss < best:
            best = val_loss; torch.save(checkpoint, output/'best.pth')
        print(json.dumps(summary))


if __name__ == '__main__':
    main()
