"""Run/reuse the ROS1 FAST-LIO frontend in the existing Docker image."""
import grp
import json
import os
from pathlib import Path
import pwd
import shlex
import subprocess

import numpy as np
import yaml
from tools.bag.reader import bag_identity

ROOT = Path(__file__).resolve().parents[2]


def docker_command(arguments):
    """Use an already configured docker group even before the next login."""
    command = ['docker', *map(str, arguments)]
    socket = Path('/var/run/docker.sock')
    if os.access(socket, os.R_OK | os.W_OK):
        return command
    try:
        group = grp.getgrnam('docker')
    except KeyError as exc:
        raise RuntimeError('Docker socket is unavailable; configure Docker access first') from exc
    if pwd.getpwuid(os.getuid()).pw_name in group.gr_mem:
        return ['sg', 'docker', '-c', shlex.join(command)]
    raise RuntimeError('Current user has no Docker access')


def ensure_frontend(bag, dataset, config, image, rate):
    identity = bag_identity(bag)
    folder = dataset/'frontend'
    folder.mkdir(parents=True, exist_ok=True)
    requested = yaml.safe_load((ROOT/config['fastlio_config']).read_text())
    host_plan = {'bag': identity, 'image': image, 'config': requested, 'playback_rate': rate}
    plan_file = folder/'host_manifest.json'
    previous_plan = json.loads(plan_file.read_text()) if plan_file.exists() else None
    if previous_plan is not None and any(previous_plan.get(key) != value for key, value in host_plan.items()):
        raise ValueError('Frontend input/config/image changed; use a new dataset directory')
    result = folder/'sequence.json'
    if result.exists() and json.loads(result.read_text()).get('complete'):
        existing = yaml.safe_load((folder/'fastlio-config.yaml').read_text())
        existing.pop('use_sim_time', None)
        existing['lemon_export'].pop('directory', None)
        if existing != requested:
            raise ValueError('Saved frontend configuration differs from requested settings')
        info = json.loads((dataset/'bag_info.json').read_text()) if (dataset/'bag_info.json').exists() else None
        if info is not None and any(info.get(k) != v for k, v in identity.items()):
            raise ValueError('Existing dataset belongs to another bag')
        sequence = json.loads(result.read_text())
        if info is not None and abs(sequence['record_start']-info['record_start']) > .001:
            raise ValueError('Existing frontend time interval differs from this bag')
        rows = np.loadtxt(folder/'pose.txt', ndmin=2)
        tum = np.loadtxt(folder/'trajectory.tum', ndmin=2)
        if rows.shape != (sequence['keyframes'], 9) or tum.shape != (len(rows), 8):
            raise ValueError('Incomplete frontend pose files')
        if not all((folder/'pcd'/f'{int(i)}.pcd').is_file() for i in rows[:, 0]):
            raise ValueError('Missing frontend PCD files')
        if previous_plan is None:
            if info is None:
                raise ValueError('Cannot identify existing frontend without its bag metadata')
            plan_file.write_text(json.dumps(dict(host_plan, image_id=None, adopted_export=True), indent=2)+'\n')
        print(f"Reusing completed FAST-LIO: {len(rows)} paired scans (original image provenance retained)", flush=True)
        return sequence
    if (folder/'pose.json').exists():
        raise ValueError('Frontend output is incomplete; preserve it and use another dataset directory')
    host_plan['image_id'] = subprocess.check_output(
        docker_command(['image', 'inspect', image, '--format', '{{.Id}}']), text=True).strip()
    if previous_plan is not None and previous_plan.get('image_id') != host_plan['image_id']:
        raise ValueError('Docker image changed during an incomplete frontend run; use a new dataset directory')
    plan_file.write_text(json.dumps(host_plan, indent=2)+'\n')
    command = ['run', '--rm', '--init', '--cpus', '6', '--user', f'{os.getuid()}:{os.getgid()}',
               '--env', 'HOME=/tmp/lidar-vggt-ros', '--env', 'OMP_NUM_THREADS=4',
               '--mount', f'type=bind,src={Path(bag).resolve()},dst=/bags/input.bag,readonly',
               '--mount', f'type=bind,src={ROOT},dst=/work,readonly',
               '--mount', f'type=bind,src={folder.resolve()},dst=/result', image,
               'python3', '/work/tools/bag/run_fastlio_ros.py', '--bag', '/bags/input.bag',
               '--output', '/result', '--config', '/work/'+config['fastlio_config'], '--rate', str(rate)]
    subprocess.run(docker_command(command), check=True)
    sequence = json.loads(result.read_text())
    if not sequence.get('complete'):
        raise ValueError('FAST-LIO did not complete')
    return sequence
