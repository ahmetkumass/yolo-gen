"""VLM family adapters.

Importing this package registers every concrete adapter with the
factory in :mod:`yologen.models.vlm.base`. Use :func:`create_vlm` to
get the right adapter for an HF model id.
"""

from yologen.models.vlm.base import (
    VLMBase,
    VLMWorkerPreprocessor,
    create_vlm,
    register_vlm,
    registered_adapters,
)

# Concrete adapters — importing them registers their model-name patterns
# with the factory. New families plug in here.
from yologen.models.vlm import qwen      # noqa: F401 — registers QwenVLM
from yologen.models.vlm import internvl  # noqa: F401 — registers InternVLM

from yologen.models.vlm.qwen import QwenVLM, create_qwen_vlm
from yologen.models.vlm.internvl import InternVLM, create_internvl

__all__ = [
    "VLMBase",
    "VLMWorkerPreprocessor",
    "create_vlm",
    "register_vlm",
    "registered_adapters",
    "QwenVLM",
    "InternVLM",
    # Back-compat aliases — existing callers may still use these
    "create_qwen_vlm",
    "create_internvl",
]
