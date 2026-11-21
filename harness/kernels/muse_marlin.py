from pathlib import Path

import torch

torch.ops.load_library(str(Path(__file__).with_name("muse_marlin_cuda.so")))


@torch.library.register_fake("muse_marlin::decode")
def _decode_fake(a, weight, scales, global_scale, workspace, n, k, expanded=True):
    return a.new_empty((1, n))


def decode(a, weight, scales, global_scale, workspace, n, k, expanded=True):
    return torch.ops.muse_marlin.decode(
        a, weight, scales, global_scale, workspace, n, k, expanded
    )
