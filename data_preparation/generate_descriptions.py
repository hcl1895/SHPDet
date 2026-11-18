"""Generate image-ID-keyed degradation descriptions with an OpenAI-compatible VLM."""
import argparse
import base64
import io
import json
import os
from pathlib import Path

from PIL import Image
from data_preparation.prepare import read_manifest

PROMPT = (
    'Task: Specialized image degradation tagging. Mandatory: English ONLY. '
    'Strictly under 30 words. No full sentences. Output strictly in this format: '
    'Category, Primary Artifact, Texture Detail, Severity. '
    'Category: Foggy, Rainy, or Low-light. '
    'Primary Artifact: describe the most visible damage. '
    'Texture Detail: describe the visible texture. Severity: Low, Medium, High, or Extreme.'
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--model', default='qwen3-vl-plus')
    parser.add_argument('--base-url', default='https://dashscope.aliyuncs.com/compatible-mode/v1')
    parser.add_argument('--api-key-env', default='DASHSCOPE_API_KEY')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    from openai import OpenAI
    key = os.environ.get(args.api_key_env)
    if not key:
        raise ValueError(f'Set the {args.api_key_env} environment variable.')
    records = read_manifest(args.manifest, check_images=True)
    output = Path(args.output)
    existing = {}
    if output.exists():
        if not args.resume:
            raise FileExistsError('Use a new output path or --resume.')
        for line in output.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            if not row.get('text') or row['id'] in existing:
                raise ValueError('Existing descriptions contain empty text or duplicate IDs.')
            existing[row['id']] = row
    output.parent.mkdir(parents=True, exist_ok=True)
    client = OpenAI(api_key=key, base_url=args.base_url)
    with output.open('a', encoding='utf-8') as stream:
        for record in records:
            if record['id'] in existing:
                continue
            buffer = io.BytesIO()
            with Image.open(record['image']) as im:
                im.convert('RGB').save(buffer, format='PNG')
            image_url = 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')
            result = client.chat.completions.create(model=args.model, messages=[{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': image_url}}, {'type': 'text', 'text': PROMPT}]}])
            text = (result.choices[0].message.content or '').strip()
            if not text:
                raise RuntimeError(f"Empty model response for {record['id']}; no replacement description was written.")
            stream.write(json.dumps({'id': record['id'], 'text': text, 'model': args.model}, ensure_ascii=False) + '\n')
            stream.flush()
            print(record['id'])


if __name__ == '__main__':
    main()
