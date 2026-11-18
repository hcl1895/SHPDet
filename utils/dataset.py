"""Paired loading; spatial/HSV augmentation retained from the supplied loader."""
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from data_preparation.prepare import read_manifest
from utils.utils import preprocess_input
from utils.runtime import category_vector

class PairedDataset(Dataset):
    def __init__(self, manifest, descriptions, input_size=640, train=False):
        self.records = read_manifest(manifest, check_images=True)
        self.descriptions = descriptions
        self.input_size = input_size
        self.train = train
        if input_size < 32 or input_size % 32:
            raise ValueError('Input size must be a positive multiple of 32.')
        if any(not r.get('clean_image') for r in self.records):
            raise ValueError('Paired training/validation requires clean_image in every record.')
        for record in self.records:
            category_vector(record.get('weather'))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image, boxes, clean = self.get_random_data(record, (self.input_size, self.input_size), random=self.train)
        image = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))
        clean = np.transpose(preprocess_input(np.array(clean, dtype=np.float32)), (2, 0, 1))
        if len(boxes):
            boxes[:, 2:4] -= boxes[:, :2]
            boxes[:, :2] += boxes[:, 2:4] / 2
        return image, boxes, clean, self.descriptions.get(record['id'], ''), category_vector(record['weather'])

    def rand(self, a=0, b=1):
        return np.random.rand() * (b-a) + a

    def get_random_data(self, record, input_shape, jitter=.3, hue=.1, sat=.7, val=.4, random=True):
        with Image.open(record['image']) as source:
            image = source.convert('RGB')
        with Image.open(record['clean_image']) as source:
            clearimg = source.convert('RGB')
        iw, ih = image.size
        h, w = input_shape

        box = np.asarray(record['boxes'], dtype=np.float32).reshape(-1, 5).copy()

        if not random:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2

            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)

            '''clear'''
            clearimg = clearimg.resize((nw, nh), Image.BICUBIC)
            new_clearimg = Image.new('RGB', (w, h), (128, 128, 128))
            new_clearimg.paste(clearimg, (dx, dy))
            clear_image_data = np.array(new_clearimg, np.float32)

            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box

            return image_data, box, clear_image_data

        new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        clearimg = clearimg.resize((nw, nh), Image.BICUBIC)

        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        '''clear'''
        new_clearimg = Image.new('RGB', (w, h), (128, 128, 128))
        new_clearimg.paste(clearimg, (dx, dy))
        clearimg = new_clearimg

        flip = self.rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            clearimg = clearimg.transpose(Image.FLIP_LEFT_RIGHT)

        image_data = np.array(image, np.uint8)
        clear_image_data = np.array(clearimg, np.uint8)

        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1

        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype

        hue1, sat1, val1 = cv2.split(cv2.cvtColor(clear_image_data, cv2.COLOR_RGB2HSV))
        dtype1 = clear_image_data.dtype

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        x1 = np.arange(0, 256, dtype=r.dtype)
        lut_hue1 = ((x1 * r[0]) % 180).astype(dtype)
        lut_sat1 = np.clip(x1 * r[1], 0, 255).astype(dtype)
        lut_val1 = np.clip(x1 * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        clear_image_data = cv2.merge((cv2.LUT(hue1, lut_hue1), cv2.LUT(sat1, lut_sat1), cv2.LUT(val1, lut_val1)))
        clear_image_data = cv2.cvtColor(clear_image_data, cv2.COLOR_HSV2RGB)

        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip: box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]

        return image_data, box, clear_image_data


def collate(batch):
    images, boxes, clean, texts, categories = zip(*batch)
    return np.stack(images), list(boxes), np.stack(clean), list(texts), np.asarray(categories, dtype=np.float32)
