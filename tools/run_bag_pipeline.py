"""Raw bag -> shared FAST-LIO / keyframes / model -> baseline, correspondence, or both."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import yaml
from models.registry import MODEL_NAMES
from tools.bag.reader import bag_identity, inspect_bag
from tools.bag.frontend import ensure_frontend
from tools.bag.dataset import write_reconstruction_config
from tools.bag.reconstruction import reconstruct
from tools.extract_bag_keyframes import extract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bags', type=Path, nargs='+')
    parser.add_argument('--model', choices=MODEL_NAMES, default='pi3')
    parser.add_argument('--method', choices=['baseline', 'correspondence', 'both'], default='correspondence',
                        help='Registration method; both shares model predictions and saves separate maps')
    parser.add_argument('--refine-line', action='store_true', help='Optional baseline attitude refinement for straight trajectories')
    parser.add_argument('--preset', type=Path, help='Override automatic UAVScenes / MUN-FRL Lighthouse preset')
    parser.add_argument('--dataset-root', type=Path, default=ROOT/'data')
    parser.add_argument('--output-root', type=Path, default=ROOT/'outputs/from_bag')
    parser.add_argument('--docker-image', default='lemon-mapping:fastlio')
    parser.add_argument('--playback-rate', type=float, default=4.)
    parser.add_argument('--parallax-px', type=float)
    parser.add_argument('--image-time-offset', type=float)
    parser.add_argument('--start-offset', type=float, help='Image start in seconds from the preset time origin; FAST-LIO still starts at 0')
    parser.add_argument('--end-offset', type=float)
    parser.add_argument('--chunk-size', type=int, default=40)
    parser.add_argument('--chunk-step', type=int, default=25)
    parser.add_argument('--max-chunks', type=int, help='Smoke run only; default reconstructs every selected frame')
    parser.add_argument('--sample-ratio', type=float, default=.05)
    parser.add_argument('--prepare-only', action='store_true', help='Stop after dataset preparation')
    parser.add_argument('--pgo', action='store_true', help='Optimize per-frame poses across overlapping chunks after reconstruction')
    parser.add_argument('--pgo-config', type=Path, default=ROOT/'config/pgo.yaml')
    parser.add_argument('--pgo-tune', action='store_true', help='Select PGO ICP weight on held-out tuning pixels')
    args = parser.parse_args()
    if not 3 <= args.chunk_size or not 1 <= args.chunk_step < args.chunk_size or args.playback_rate <= 0:
        parser.error('Require chunk-size >= 3, 1 <= chunk-step < chunk-size and playback-rate > 0')
    if not np.isfinite([args.playback_rate, args.sample_ratio]).all() or not 0 < args.sample_ratio <= 1:
        parser.error('Require finite playback rate and sample-ratio in (0, 1]')
    if args.pgo_tune and not args.pgo:
        parser.error('--pgo-tune requires --pgo')
    if args.pgo and args.prepare_only:
        parser.error('--pgo cannot be combined with --prepare-only')
    if args.refine_line and args.method == 'correspondence':
        parser.error('--refine-line applies to baseline or both')
    if args.max_chunks is not None and args.max_chunks < 1:
        parser.error('max-chunks must be positive')
    cv2.setNumThreads(4)
    cv2.setRNGSeed(0)
    for bag in args.bags:
        bag = bag.resolve(strict=True)
        scene = bag.stem
        preset = args.preset
        if preset is None:
            prefix = next((s for s in ('amtown', 'amvalley', 'hkairport', 'hkisland', 'lighthouse') if scene.lower().startswith(s)), None)
            if prefix is None:
                parser.error(f'No preset for {scene}; specify --preset with calibration and topics')
            preset = ROOT/'config/bag'/f'{prefix}.yaml'
        config = yaml.safe_load(preset.read_text())
        if args.parallax_px is not None:
            config['selection']['parallax_px'] = args.parallax_px
        if args.image_time_offset is not None:
            config['image_time_offset_s'] = args.image_time_offset
        dataset = args.dataset_root.resolve()/(scene+'_raw')
        dataset.mkdir(parents=True, exist_ok=True)
        identity = bag_identity(bag)
        info_file = dataset/'bag_info.json'
        if info_file.exists():
            info = json.loads(info_file.read_text())
            if any(info.get(k) != v for k, v in identity.items()):
                raise ValueError('Existing dataset belongs to another bag')
        else:
            info = inspect_bag(bag, config['image_topic'], config['height_topic'])
            info_file.write_text(json.dumps(info, indent=2)+'\n')
        print(f'[{scene}] FAST-LIO from original bag', flush=True)
        frontend = ensure_frontend(bag, dataset, config, args.docker_image, args.playback_rate)
        print(f'[{scene}] Takeoff and parallax keyframes', flush=True)
        selected = extract(bag, dataset, config, start_offset=args.start_offset, end_offset=args.end_offset)
        subprocess.run([sys.executable, str(ROOT/'tools/preview_bag.py'), str(dataset)], check=True)
        write_reconstruction_config(dataset, config, selected, args.chunk_size, args.chunk_step)
        if args.prepare_only:
            print(f'[{scene}] Prepared {selected["selected_images"]} images: {dataset}', flush=True)
            continue
        reports = reconstruct(bag, dataset, args.output_root.resolve(), selected, frontend,
            model=args.model, method=args.method, chunk_size=args.chunk_size, chunk_step=args.chunk_step,
            max_chunks=args.max_chunks, sample_ratio=args.sample_ratio, refine_line=args.refine_line)
        if args.pgo:
            from PGO.pipeline import run_pgo
            pgo_config = yaml.safe_load(args.pgo_config.read_text())
            for name, report in reports.items():
                result = Path(report['output'])
                run_pgo(result, result/'pgo', result.parent/'pgo_inputs', pgo_config, tune=args.pgo_tune, reuse=True)


if __name__ == '__main__':
    main()
