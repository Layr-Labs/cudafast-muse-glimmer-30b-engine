# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pick Inductor's Triton kernel configs by a fixed rule instead of by timing.

Inductor compiles a pointwise or reduction kernel with several candidate
configs and, at the kernel's first launch, keeps the fastest one
(``CachingAutotuner.autotune_to_one_config``). For a looped reduction the
candidates differ in ``R0_BLOCK``, the width of the fp32 partial-sum
accumulator, so they sum the row in different orders and give different
bits. When two candidates time within noise of each other, each boot can keep
a different one, and the same prompt then yields a different greedy stream
on different boots.

``install()`` replaces that timing with ``pick_config_index``: no candidate
is launched, the same config is kept on every boot, and the save-cache hook
still records it. The rule takes the candidate with

1. no register spills beyond Inductor's own threshold (timing scores those
   as infinitely slow, so they are never kept),
2. the widest reduction block (a whole row in one block when a candidate
   covers it; this is the class timing keeps on GB10 in the majority of
   boots),
3. then the most warps, the widest X/Y block and the fewest stages.

Configs that share the reduction block and give each row to one warp
compute bit-identical results, so step 3 only fixes the launch shape.
"""

from __future__ import annotations

import re
import time

_installed = False
_R_BLOCK = re.compile(r"^R\d*_BLOCK(_\d+)?$")
_XY_BLOCK = re.compile(r"^[XYZ]BLOCK(_\d+)?$")


def _product(kwargs: dict, pattern: re.Pattern) -> int:
    p = 1
    for key, value in kwargs.items():
        if pattern.match(key) and isinstance(value, int):
            p *= value
    return p


def pick_config_index(
    candidates: list[tuple[dict, int, int, int | None]], spill_threshold: int
) -> int:
    """Index of the candidate to keep.

    ``candidates`` holds one ``(kwargs, num_warps, num_stages, n_spills)`` per
    launcher, in the autotuner's order. Ties keep the earlier candidate.
    """
    if not candidates:
        raise ValueError("no candidate configs")
    usable = [
        i
        for i, (_, _, _, spills) in enumerate(candidates)
        if spills is None or spills <= spill_threshold
    ] or list(range(len(candidates)))

    def key(i: int) -> tuple[int, int, int, int]:
        kwargs, num_warps, num_stages, _ = candidates[i]
        return (
            _product(kwargs, _R_BLOCK),
            num_warps,
            _product(kwargs, _XY_BLOCK),
            -num_stages,
        )

    best = usable[0]
    for i in usable[1:]:
        if key(i) > key(best):
            best = i
    return best


def install() -> None:
    """Replace CachingAutotuner.autotune_to_one_config. Idempotent."""
    global _installed
    if _installed:
        return

    import os

    if os.environ.get("VLLM_AUTOTUNE_PIN", "1") == "0":
        return

    from torch._inductor.runtime import triton_heuristics as th
    from torch._inductor.triton_bundler import TritonBundler

    from vllm.logger import init_logger

    logger = init_logger(__name__)
    cls = th.CachingAutotuner
    for name in (
        "_release_static_launchers_except",
        "_prune_compile_results_to_launcher",
    ):
        if not hasattr(cls, name):
            raise RuntimeError(
                f"CachingAutotuner.{name} is missing: this torch does not match "
                "the one autotune_pin was written for"
            )

    def autotune_to_one_config(self, *args, **kwargs):
        start_time = time.time_ns()
        candidates = [
            (
                dict(launcher.config.kwargs),
                launcher.config.num_warps,
                launcher.config.num_stages,
                launcher.n_spills,
            )
            for launcher in self.launchers
        ]
        index = pick_config_index(
            candidates, self.inductor_meta.get("spill_threshold", 16)
        )
        launcher = self.launchers[index]
        self._release_static_launchers_except(launcher)
        self.launchers = [launcher]
        self._prune_compile_results_to_launcher(launcher)
        self.autotune_time_taken_ns = self.precompile_time_taken_ns + (
            time.time_ns() - start_time
        )
        logger.info(
            "[inductor-autotune-pinned] kernel=%s file=%s candidates=%d picked=%s",
            self.fn.__name__,
            os.path.basename(getattr(self, "filename", None) or "?"),
            len(candidates),
            launcher.config,
        )
        TritonBundler.put_winner(launcher.cache_hash)
        if self.save_cache_hook:
            self.save_cache_hook(
                launcher.config,
                self.autotune_time_taken_ns,
                found_by_coordesc=self.inductor_meta.get(
                    "coordinate_descent_tuning", False
                ),
                triton_cache_hash=launcher.cache_hash,
            )

    cls.autotune_to_one_config = autotune_to_one_config
    _installed = True
