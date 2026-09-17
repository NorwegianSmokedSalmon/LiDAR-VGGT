"""Model-specific image loading with exact original-pixel -> network-pixel maps.

VGGT/Omega use their pinned official loaders. Pi3's area-limited resize follows
its official load_images_as_tensor, accepting an ordered list instead of a folder.
"""
from dataclasses import dataclass
import math

import numpy as np
from PIL import Image
import torch

from models.registry import use_repository


@dataclass
class PreparedImages:
    images: torch.Tensor
    pixel_transform: np.ndarray
    original_size: np.ndarray
    valid_bounds: np.ndarray  # inclusive pixel centers, x0 y0 x1 y1


def resize_affine(original_wh, resized_wh, crop_xy=(0, 0), pad_xy=(0, 0)):
    sx, sy = np.asarray(resized_wh, dtype=float) / np.asarray(original_wh, dtype=float)
    return np.array([[sx, 0, (sx - 1)/2 - sx*crop_xy[0] + pad_xy[0]],
                     [0, sy, (sy - 1)/2 - sy*crop_xy[1] + pad_xy[1]], [0, 0, 1.]])


def pi3_shape(width, height, pixel_limit=255000):
    scale = math.sqrt(pixel_limit / (width * height))
    target_w, target_h = width*scale, height*scale
    k, m = max(1, round(target_w/14)), max(1, round(target_h/14))
    while k*14*m*14 > pixel_limit:
        if k/m > target_w/target_h and k > 1:
            k -= 1
        elif m > 1:
            m -= 1
        else:
            raise ValueError('Image pixel budget is too small')
    return k*14, m*14


def omega_layout(width, height, resolution=512):
    # Official balanced loader: center-crop aspect into [0.5, 2], then resize.
    crop_w, crop_h = width, height
    if height/width < 0.5:
        crop_w = min(width, max(1, round(height/0.5)))
    elif height/width > 2:
        crop_h = min(height, max(1, round(width*2)))
    left, top = (width-crop_w)//2, (height-crop_h)//2
    tokens = (resolution//16)**2
    wp = math.sqrt(tokens/(crop_h/crop_w))
    hp = tokens/wp
    resized = max(1, round(wp))*16, max(1, round(hp))*16
    return (crop_w, crop_h), (left, top), resized


def prepare_images(paths, model_name):
    if not paths:
        raise ValueError('At least one image is required')
    sizes = []
    for path in paths:
        with Image.open(path) as image:
            sizes.append(image.size)
    transforms, bounds = [], []
    if model_name == 'pi3':
        resized = pi3_shape(*sizes[0])
        tensors = []
        for path, size in zip(paths, sizes):
            with Image.open(path) as image:
                image = image.convert('RGB').resize(resized, Image.Resampling.LANCZOS)
                tensors.append(torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float()/255.)
            transforms.append(resize_affine(size, resized))
            bounds.append([0, 0, resized[0]-1, resized[1]-1])
        images = torch.stack(tensors)
    elif model_name == 'vggt':
        use_repository(model_name)
        from vggt.utils.load_fn import load_and_preprocess_images
        images = load_and_preprocess_images([str(p) for p in paths], mode='crop')
        out_h, out_w = images.shape[-2:]
        for width, height in sizes:
            new_w, new_h = 518, round(height*518/width/14)*14
            cropped_h = min(new_h, 518)
            left, top = (out_w-new_w)//2, (out_h-cropped_h)//2
            affine = resize_affine((width, height), (new_w, new_h))
            affine[:2, 2] += [left, top-(new_h-cropped_h)//2]
            transforms.append(affine)
            bounds.append([left, top, left+new_w-1, top+cropped_h-1])
    elif model_name == 'vggt_omega':
        use_repository(model_name)
        from vggt_omega.utils.load_fn import load_and_preprocess_images
        images = load_and_preprocess_images([str(p) for p in paths], image_resolution=512, mode='balanced')
        out_h, out_w = images.shape[-2:]
        for width, height in sizes:
            crop_wh, crop_xy, resized = omega_layout(width, height)
            left, top = (out_w-resized[0])//2, (out_h-resized[1])//2
            transforms.append(resize_affine(crop_wh, resized, crop_xy, (left, top)))
            bounds.append([left, top, left+resized[0]-1, top+resized[1]-1])
    else:
        raise ValueError(f'Unknown model: {model_name}')
    return PreparedImages(images, np.asarray(transforms), np.asarray(sizes), np.asarray(bounds))
