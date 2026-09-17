"""Thin adapters around unmodified, pinned official model implementations."""
import torch

from models.geometry import camera_to_extrinsic, depth_to_world, fit_intrinsics_from_local_points
from models.registry import checkpoint_path, model_spec, use_repository


def cpu_array(tensor):
    return tensor.detach().float().cpu().numpy()


def canonicalize_pi3(prediction, images):
    local = cpu_array(prediction['local_points'][0])
    confidence = cpu_array(torch.sigmoid(prediction['conf'][0, ..., 0]))
    return {'world_points': cpu_array(prediction['points'][0]), 'world_points_conf': confidence,
            'depth': local[..., 2], 'depth_conf': confidence,
            'extrinsic': camera_to_extrinsic(cpu_array(prediction['camera_poses'][0])),
            'intrinsic': fit_intrinsics_from_local_points(local, confidence), 'images': cpu_array(images)}


def canonicalize_omega(prediction, images):
    from vggt_omega.utils.pose_enc import encoding_to_camera
    extrinsic, intrinsic = encoding_to_camera(prediction['pose_enc'], images.shape[-2:])
    extrinsic, intrinsic = cpu_array(extrinsic[0]), cpu_array(intrinsic[0])
    depth = cpu_array(prediction['depth'][0, ..., 0])
    confidence = cpu_array(prediction['depth_conf'][0])
    return {'world_points': depth_to_world(depth, extrinsic, intrinsic), 'world_points_conf': confidence,
            'depth': depth, 'depth_conf': confidence, 'extrinsic': extrinsic,
            'intrinsic': intrinsic, 'images': cpu_array(images)}


class ReconstructionModel:
    def __init__(self, name, device, checkpoint=None):
        self.name, self.device = name, torch.device(device)
        self.spec = model_spec(name)
        self.checkpoint = checkpoint_path(name) if checkpoint is None else checkpoint
        from pathlib import Path
        self.checkpoint = Path(self.checkpoint).resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f'Missing {name} weights: {self.checkpoint}. '
                                    f'Run python tools/download_models.py --models {name}, '
                                    'or supply an authorized local file with --checkpoint.')
        use_repository(name)
        if name == 'vggt':
            from vggt.models.vggt import VGGT
            self.model = VGGT()
        elif name == 'pi3':
            from pi3.models.pi3 import Pi3
            self.model = Pi3()
        elif name == 'vggt_omega':
            from vggt_omega.models import VGGTOmega
            self.model = VGGTOmega(autocast=False)
        else:
            raise ValueError(name)
        if self.checkpoint.suffix == '.safetensors':
            from safetensors.torch import load_file
            state = load_file(str(self.checkpoint), device='cpu')
        else:
            state = torch.load(self.checkpoint, map_location='cpu', weights_only=True, mmap=True)
        self.model.load_state_dict(state, strict=True)
        del state
        self.model.eval().to(self.device)

    def predict(self, images):
        native = self.model(images[None] if self.name == 'pi3' else images)
        if self.name == 'pi3':
            return canonicalize_pi3(native, images)
        if self.name == 'vggt_omega':
            return canonicalize_omega(native, images)
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        extrinsic, intrinsic = pose_encoding_to_extri_intri(native['pose_enc'], images.shape[-2:])
        arrays = {key: cpu_array(native[key][0]) for key in ['world_points', 'world_points_conf', 'depth_conf']}
        arrays.update(depth=cpu_array(native['depth'][0, ..., 0]), extrinsic=cpu_array(extrinsic[0]),
                      intrinsic=cpu_array(intrinsic[0]), images=cpu_array(images))
        return arrays
