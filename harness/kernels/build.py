import os
import shutil
from pathlib import Path

from torch.utils.cpp_extension import load

source = Path(__file__).resolve().parent
output = Path(os.environ["KERNELS_OUT"]).resolve()
output.mkdir(parents=True, exist_ok=True)
load(
    name="muse_marlin_cuda",
    sources=[str(source / "marlin_dispatch.cpp"), str(source / "marlin_decode.cu")],
    extra_include_paths=[str(source / "marlin")],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "-lineinfo", "--ptxas-options=-v"],
    build_directory=str(output),
    is_python_module=False,
    verbose=True,
)
shutil.copyfile(source / "muse_marlin.py", output / "muse_marlin.py")
