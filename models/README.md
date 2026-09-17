# Reconstruction models

The adapters expose a shared interface for VGGT, the original Pi3, and VGGT-Omega-1B-512. Upstream repositories and weights are stored under `models/repos/` and `models/weights/`; both directories are excluded from version control.

## Download

```bash
python tools/download_models.py --models all
python tools/download_models.py --models vggt pi3
python tools/download_models.py --models vggt_omega --source modelscope
```

The default `--source auto` uses Hugging Face for VGGT/Pi3 and ModelScope for Omega. Source commits, weight revisions, sizes, and SHA-256 hashes are pinned in [`registry.json`](registry.json). Existing files are verified before reuse. `--sources-only` downloads model code without weights.

| Model option | Source | Weights |
| --- | --- | --- |
| `vggt` | [VGGT](https://github.com/facebookresearch/vggt) | [VGGT-1B](https://huggingface.co/facebook/VGGT-1B) |
| `pi3` | [Pi3](https://github.com/yyfz/Pi3) | [Pi3](https://huggingface.co/yyfz233/Pi3) |
| `vggt_omega` | [VGGT-Omega](https://github.com/facebookresearch/vggt-omega) | [ModelScope](https://modelscope.cn/models/facebook/VGGT-Omega) / [Hugging Face](https://huggingface.co/facebook/VGGT-Omega) |

When choosing Hugging Face for Omega, use an account with access to its weights. Model code and weights remain subject to their upstream licenses.

## Use

Pass `--model vggt`, `--model pi3`, or `--model vggt_omega` to any reconstruction entry point. The standalone inference command is:

```bash
python tools/run_inference.py --model pi3 \
  --image-dir /path/to/images --output-dir outputs/pi3_predictions
```

For prepared-data runners, `--checkpoint PATH` overrides the checkpoint and `--device auto|cuda|cpu` selects the device. Otherwise, checkpoints are resolved from `Weights` in the dataset config, then from the model registry. Pi3 uses the upstream PyTorch RoPE implementation without a separately compiled CUDA extension.

## Module layout

| Module | Responsibility |
| --- | --- |
| `registry.py` / `registry.json` | Model definitions and source/weight versions |
| `downloads.py` | ModelScope downloads and resume support |
| `backends.py` | Strict weight loading and model adapters |
| `preprocessing.py` | Model-specific image transforms |
| `geometry.py` | Camera conventions, depth unprojection, and output validation |
| `inference.py` | Chunking, inference, and serialization |
| `predictions.py` | Cache validation and shared prediction reuse |

## Prediction format

Each `chunk_<index>.npy` stores a dictionary; `manifest.json` records image order, chunk ranges, model identity, and preprocessing.

| Field | Shape | Meaning |
| --- | --- | --- |
| `world_points` | N,H,W,3 | Points in the chunk world frame |
| `world_points_conf` | N,H,W | Model-specific point confidence |
| `depth` / `depth_conf` | N,H,W | Camera-axis depth and confidence |
| `extrinsic` | N,3,4 | OpenCV world-to-camera transform |
| `intrinsic` | N,3,3 | Intrinsics in network-image coordinates |
| `images` | N,3,H,W | RGB values in [0,1] |
| `image_pixel_transform` | N,3,3 | Original-to-network pixel-center transform |
| `image_original_size` | N,2 | Original width and height |
| `image_valid_bounds` | N,4 | Valid network-image bounds, excluding padding |

VGGT uses its official crop preprocessing; Pi3 uses the official pixel budget and patch alignment; Omega uses the official 512 balanced preprocessing. Projection alignment applies the recorded pixel transform for each model.

Pi3 camera-to-world poses are inverted and confidence logits are converted with sigmoid. Its approximate intrinsics are fitted from local point rays; calibrated LiDAR projection uses the supplied calibration instead. Omega pointmaps are unprojected from depth and predicted camera parameters.

Confidence thresholds are model-specific. `null` values in projection configs select the registry defaults; use CLI confidence overrides only when needed. Geometry, confidence filtering, and preprocessing are tested under [`tests/`](../tests/).
