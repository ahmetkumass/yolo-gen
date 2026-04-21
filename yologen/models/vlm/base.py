"""Abstract interface and factory for VLM family adapters.

Concrete adapters (QwenVLM, InternVLM, GLM4VVLM, ...) implement the
load / generate / train surface that the rest of YoloGen depends on.
A single :func:`create_vlm` entry-point chooses the right adapter
based on a HuggingFace model id (regex pattern match).

Required attributes that every adapter must expose after
:meth:`load_model`:

- ``model_name`` — the HF id this adapter was instantiated with
- ``model``     — the underlying (possibly LoRA-wrapped) ``nn.Module``
- ``processor`` — the HF ``AutoProcessor``-style tokenizer+image-processor
- ``tokenizer`` — the text tokenizer (many families expose it via
                  ``processor.tokenizer``; adapters can alias)

Required methods are enforced by :class:`VLMBase` (see below).
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type


class VLMWorkerPreprocessor(ABC):
    """Picklable CPU-only dataset preprocessor.

    ``VLMDataset`` hands one of these to each DataLoader worker. The
    worker keeps it across batches and calls :meth:`__call__` once per
    sample. Implementations must:

    1. Store only picklable configuration in ``__init__`` (model name,
       pixel limits, etc.); never touch CUDA.
    2. Lazy-load the HF processor / tokenizer on the first call so the
       load happens once per worker, not once per sample.
    3. Return a flat ``Dict[str, torch.Tensor]`` of CPU tensors ready
       to be moved to GPU by the trainer. The exact keys are family-
       specific (Qwen uses ``image_grid_thw``, InternVL uses
       ``image_flags``); the trainer forwards whatever is present.
    """

    @abstractmethod
    def __call__(
        self,
        image_path: str,
        question: str,
        system_prompt: Optional[str],
        answer: str,
    ) -> Dict[str, Any]:
        ...


class VLMBase(ABC):
    """Family-agnostic VLM adapter contract.

    Subclasses are expected to set ``self.model_name`` in ``__init__``
    and populate ``self.model`` / ``self.processor`` / ``self.tokenizer``
    during :meth:`load_model`.
    """

    model_name: str

    @classmethod
    @abstractmethod
    def build_worker_preprocessor(
        cls, model_name: str, **config
    ) -> VLMWorkerPreprocessor:
        """Return a picklable dataset-worker preprocessor for this family.

        Called by :class:`yologen.core.vlm_trainer.VLMTrainer` while
        constructing the training DataLoader; the returned object is
        shared across worker processes. ``config`` contains the
        trainer's ``min_pixels`` / ``max_pixels`` / family-specific
        knobs — adapters pick what they need and ignore the rest.
        """

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def load_model(self) -> None:
        """Download weights, apply quantization/LoRA, assign self.model."""

    @abstractmethod
    def save_adapter(self, path) -> None:
        """Persist LoRA weights + processor so the adapter can be
        reloaded via :meth:`load_adapter` or via the matching predictor."""

    @abstractmethod
    def load_adapter(self, path) -> None:
        """Re-attach a previously-saved LoRA adapter to the base model."""

    # ------------------------------------------------------------------
    # Training surface
    # ------------------------------------------------------------------

    @abstractmethod
    def prepare_input(
        self,
        image,
        question: str,
        bbox: Optional[List[int]] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return a CPU-resident dict of tensors ready for ``forward``.

        Dataset workers call this in parallel; implementations must not
        touch CUDA.  The caller moves tensors to GPU at step time.
        """

    @abstractmethod
    def forward(self, **inputs) -> Dict[str, Any]:
        """Training-mode forward. Accepts whatever ``prepare_input``
        produced plus a ``labels`` tensor. Returns a dict that MUST
        contain ``loss`` when labels are supplied."""

    @abstractmethod
    def get_trainable_parameters(self):
        """Iterable of parameters the optimizer should own (typically
        the LoRA weights)."""

    def print_trainable_parameters(self) -> None:
        """Default implementation — adapters may override for a richer
        breakdown."""
        total = 0
        for p in self.get_trainable_parameters():
            total += getattr(p, "numel", lambda: 0)()
        print(f"Trainable params: {total:,}")

    # ------------------------------------------------------------------
    # Inference surface
    # ------------------------------------------------------------------

    @abstractmethod
    def generate(
        self,
        image,
        question: str,
        bbox: Optional[List[int]] = None,
        box_thickness: int = 3,
        box_color: Tuple[int, int, int] = (255, 0, 0),
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 128,
    ) -> str:
        """Run inference and return the decoded text answer.

        ``bbox`` is drawn on the image before encoding so the adapter
        shares a consistent visual-prompting convention across
        families.
        """


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

# Ordered list of (compiled-pattern, adapter-class) pairs. Order matters:
# more specific patterns should be registered first if overlaps exist.
_VLM_REGISTRY: List[Tuple[re.Pattern, Type[VLMBase]]] = []


def register_vlm(pattern: str):
    """Class decorator that registers a :class:`VLMBase` subclass under
    a regex pattern matched against the HF model id.

    Example::

        @register_vlm(r"Qwen/(?:Qwen2\\.5-VL|Qwen3-VL).*")
        class QwenVLM(VLMBase):
            ...
    """
    compiled = re.compile(pattern)

    def decorator(cls: Type[VLMBase]) -> Type[VLMBase]:
        if not issubclass(cls, VLMBase):
            raise TypeError(f"{cls.__name__} must subclass VLMBase to be registered")
        _VLM_REGISTRY.append((compiled, cls))
        return cls

    return decorator


def create_vlm(model_name: str, **kwargs) -> VLMBase:
    """Instantiate the adapter registered for ``model_name``.

    Raises:
        ValueError: if no registered pattern matches.
    """
    for pattern, cls in _VLM_REGISTRY:
        if pattern.match(model_name):
            return cls(model_name=model_name, **kwargs)
    patterns = [p.pattern for p, _ in _VLM_REGISTRY]
    raise ValueError(
        f"No VLM adapter registered for {model_name!r}. "
        f"Registered patterns: {patterns}"
    )


def registered_adapters() -> List[Tuple[str, str]]:
    """Return ``[(pattern, class_name), ...]`` — useful for diagnostics
    and for generating docs."""
    return [(p.pattern, c.__name__) for p, c in _VLM_REGISTRY]
