# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Identity of the Muse assistant before and after DFlash config wrapping."""


def is_muse_glimmer_assistant(config: object) -> bool:
    """Accept Muse itself or its explicit EAGLEConfig.model; fail closed otherwise.

    SpeculativeConfig wraps DFlash with model_type="eagle" and keeps the
    original HF config at .model. Architectures are rewritten with a DFlash
    prefix, so neither the outer type nor the architecture alone is identity.
    Only the known single wrapper is unwrapped; arbitrary nested configs and
    target Muse configs do not enable these assistant-only paths.
    """
    if getattr(config, "model_type", None) == "eagle":
        config = getattr(config, "model", None)
    return getattr(config, "model_type", None) == "muse_glimmer_assistant"
