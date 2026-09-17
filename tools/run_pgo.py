"""Python frame-level PGO for baseline or correspondence reconstruction results."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import yaml
from PGO.pipeline import run_pgo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-dir', type=Path, required=True, help='Completed baseline/correspondence directory')
    parser.add_argument('--output-dir', type=Path, help='Empty directory; default RESULT/pgo')
    parser.add_argument('--cache-dir', type=Path, help='Compact PGO inputs; default RESULT/../pgo_inputs')
    parser.add_argument('--config', type=Path, default=ROOT/'config/pgo.yaml')
    parser.add_argument('--regenerate-predictions', action='store_true', help='Restore compact inputs when dense predictions were pruned')
    parser.add_argument('--reuse', action='store_true', help='Reuse an identical completed PGO result after checksum validation')
    parser.add_argument('--prior-mode', choices=['sequence_first', 'chunk_first'], help='Override the prior anchoring policy')
    parser.add_argument('--tune', action='store_true', help='Select ICP weight using tuning pixels; test pixels remain held out')
    parser.add_argument('--prepare-only', action='store_true', help='Only build/reuse compact inputs')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    args = parser.parse_args()
    result = args.result_dir.resolve()
    config = yaml.safe_load(args.config.read_text())
    if args.prior_mode:
        config['prior'] = {'mode': args.prior_mode}
    run_pgo(result, args.output_dir or result/'pgo', args.cache_dir or result.parent/'pgo_inputs',
            config, regenerate=args.regenerate_predictions,
            tune=args.tune, device=args.device, prepare_only=args.prepare_only, reuse=args.reuse)


if __name__ == '__main__':
    main()
