"""Run VGGT, Pi3 or VGGT-Omega and save canonical chunks for either registration pipeline."""
import argparse
from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import yaml
from models.registry import MODEL_NAMES, configured_checkpoint
from models.inference import natural_key, run_inference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODEL_NAMES, default='vggt')
    parser.add_argument('--image_dir', '--image-dir', required=True, type=Path)
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT/'config/config.yaml')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--chunk-size', type=int)
    parser.add_argument('--overlap', type=int)
    parser.add_argument('--max-chunks', type=int)
    parser.add_argument('--max-images', type=int)
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    args = parser.parse_args()
    if args.start_index < 0 or (args.max_images is not None and args.max_images < 1):
        parser.error('start-index must be nonnegative and max-images must be positive')
    config = yaml.safe_load(args.config.read_text())
    chunk = config.get('Chunk', {})
    legacy = config.get('Model', {})
    size = args.chunk_size if args.chunk_size is not None else legacy.get('chunk_size', chunk.get('chunk_size',40))
    overlap = args.overlap if args.overlap is not None else legacy.get('overlap',size-min(size,chunk.get('chunk_step',25)))
    images = sorted((p for p in args.image_dir.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png'}), key=natural_key)
    images = images[args.start_index:]
    if args.max_images is not None:
        images = images[:args.max_images]
    output = args.output_dir or PROJECT_ROOT/'outputs'/datetime.now().strftime(f'{args.model}_%Y%m%d_%H%M%S')
    run_inference(images, output, configured_checkpoint(args.model, config, args.checkpoint),
                  size, overlap, args.max_chunks, args.device, args.model)


if __name__ == '__main__':
    main()
