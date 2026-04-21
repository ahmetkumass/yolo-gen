"""InternVL 3.5 adapter with QLoRA support.

Implements the :class:`~yologen.models.vlm.base.VLMBase` contract for the
OpenGVLab InternVL 3.5 family (1B / 4B / 8B and larger variants), so
the rest of YoloGen (VLMTrainer, VLMPredictor, NegativeMiner) can use
it interchangeably with Qwen-VL via the factory.

Architecture notes that shape this adapter:

- InternVL 3.5 ships as a ``trust_remote_code=True`` ``AutoModel`` — the
  model class isn't registered with the standard ``AutoModelForXxx``
  hierarchies, so we load it via ``AutoModel.from_pretrained``.
- Images are broken into dynamic tiles (up to ``max_num`` patches of
  ``image_size``x``image_size``) based on aspect ratio, plus an
  optional thumbnail. The tile tensor becomes ``pixel_values``.
- The LLM backbone for InternVL 3.5 is Qwen3, so LoRA target modules
  match the Qwen3 attention projection names
  (``q_proj``, ``k_proj``, ``v_proj``, ``o_proj``).
- ``model.chat(tokenizer, pixel_values, question, gen_cfg)`` is the
  official inference entry-point and already encodes the right chat
  template. For training we build input_ids directly via the
  tokenizer's ``apply_chat_template``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from PIL import Image

from yologen.models.vlm.base import VLMBase, VLMWorkerPreprocessor, register_vlm


# ---------------------------------------------------------------------------
# Tile-based image preprocessing (InternVL convention)
# ---------------------------------------------------------------------------

IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

DEFAULT_IMAGE_SIZE: int = 448
DEFAULT_MAX_NUM_TILES: int = 12
DEFAULT_MIN_NUM_TILES: int = 1


def _build_transform(input_size: int):
    """Normalization transform used for every tile after resizing."""
    from torchvision import transforms as T
    from torchvision.transforms.functional import InterpolationMode

    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: List[Tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> Tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - target)
        if diff < best_ratio_diff:
            best_ratio_diff = diff
            best_ratio = ratio
        elif diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(
    image: Image.Image,
    *,
    min_num: int = DEFAULT_MIN_NUM_TILES,
    max_num: int = DEFAULT_MAX_NUM_TILES,
    image_size: int = DEFAULT_IMAGE_SIZE,
    use_thumbnail: bool = True,
) -> List[Image.Image]:
    """Split an image into aspect-ratio-matched tiles of ``image_size``.

    Follows the reference implementation from the InternVL model card
    verbatim — any divergence breaks the model's positional encoding
    assumptions about tile layout.
    """
    orig_w, orig_h = image.size
    aspect = orig_w / orig_h

    target_ratios = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        },
        key=lambda x: x[0] * x[1],
    )
    best = _find_closest_aspect_ratio(aspect, target_ratios, orig_w, orig_h, image_size)

    tw = image_size * best[0]
    th = image_size * best[1]
    blocks = best[0] * best[1]
    resized = image.resize((tw, th))

    tiles: List[Image.Image] = []
    cols = tw // image_size
    for i in range(blocks):
        box = (
            (i % cols) * image_size,
            (i // cols) * image_size,
            ((i % cols) + 1) * image_size,
            ((i // cols) + 1) * image_size,
        )
        tiles.append(resized.crop(box))

    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


def load_image_tiles(
    image: "str | Path | Image.Image",
    *,
    input_size: int = DEFAULT_IMAGE_SIZE,
    max_num: int = DEFAULT_MAX_NUM_TILES,
) -> torch.Tensor:
    """Convenience wrapper: path/PIL -> stacked tile tensor ``[N, 3, H, W]``."""
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")
    else:
        image = image.convert("RGB") if image.mode != "RGB" else image

    transform = _build_transform(input_size)
    tiles = _dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    return torch.stack([transform(t) for t in tiles])


def _draw_bbox(image: Image.Image, bbox, box_color, box_thickness) -> Image.Image:
    """Render a bounding box on a copy of the image so the VLM sees the
    same visual prompt YoloGen uses everywhere else."""
    from PIL import ImageDraw

    out = image.copy()
    draw = ImageDraw.Draw(out)
    x1, y1, x2, y2 = [int(v) for v in bbox]
    # Pillow uses RGB tuples; caller convention is RGB here.
    draw.rectangle([x1, y1, x2, y2], outline=tuple(box_color), width=int(box_thickness))
    return out


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

SUPPORTED_SIZES = {
    "OpenGVLab/InternVL3_5-1B",
    "OpenGVLab/InternVL3_5-4B",
    "OpenGVLab/InternVL3_5-8B",
}


class InternVLWorkerPreprocessor(VLMWorkerPreprocessor):
    """Picklable preprocessor for InternVL DataLoader workers.

    Lazy-loads the tokenizer and the model config (for
    ``num_image_token``) on first call. The full model never enters a
    worker — only the tokenizer + tile logic.
    """

    def __init__(
        self,
        model_name: str,
        image_size: int = DEFAULT_IMAGE_SIZE,
        max_num_tiles: int = DEFAULT_MAX_NUM_TILES,
    ) -> None:
        self.model_name = model_name
        self.image_size = image_size
        self.max_num_tiles = max_num_tiles
        self._tokenizer = None
        self._num_image_token: Optional[int] = None

    def _ensure_tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoConfig, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True, use_fast=False
            )
            cfg = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
            self._num_image_token = int(getattr(cfg, "num_image_token", 256))
        return self._tokenizer, self._num_image_token

    def __call__(self, image_path, question, system_prompt, answer):
        tokenizer, num_image_token = self._ensure_tokenizer()

        pixel_values = load_image_tiles(
            image_path, input_size=self.image_size, max_num=self.max_num_tiles
        ).to(dtype=torch.bfloat16)
        num_tiles = pixel_values.shape[0]

        image_placeholder = (
            "<img>" + "<IMG_CONTEXT>" * (num_image_token * num_tiles) + "</img>"
        )
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        user_content = (
            question.replace("<image>", image_placeholder)
            if "<image>" in question
            else f"{image_placeholder}\n{question}"
        )
        messages.append({"role": "user", "content": user_content})

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        tokenized = tokenizer(prompt, return_tensors="pt", padding=False)
        image_flags = torch.ones(num_tiles, dtype=torch.long)

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "pixel_values": pixel_values,
            "image_flags": image_flags,
        }


@register_vlm(r"OpenGVLab/InternVL3(?:\.|_)?5.*")
class InternVLM(VLMBase):
    """InternVL 3.5 adapter with optional 4-bit/8-bit QLoRA fine-tuning.

    Example::

        vlm = InternVLM(
            model_name="OpenGVLab/InternVL3_5-4B",
            load_in_4bit=True,
            use_lora=True,
        )
        vlm.load_model()
        out = vlm.generate(image="frame.jpg",
                           question="Is there a weapon in the red box?",
                           bbox=[100, 200, 300, 400])
    """

    def __init__(
        self,
        model_name: str = "OpenGVLab/InternVL3_5-4B",
        *,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        use_lora: bool = True,
        lora_r: int = 64,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
        gradient_checkpointing: bool = True,
        device: str = "",
        # Tile-based image preprocessing knobs. ``max_pixels`` is kept
        # for API symmetry with QwenVLM but acts as a rough upper bound
        # translated into ``max_num_tiles`` below.
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        image_size: int = DEFAULT_IMAGE_SIZE,
        max_num_tiles: Optional[int] = None,
    ) -> None:
        if model_name not in SUPPORTED_SIZES and not model_name.startswith("OpenGVLab/"):
            raise ValueError(
                f"InternVLM was asked to load {model_name!r}, which does not "
                f"match the InternVL naming convention."
            )

        self.model_name = model_name
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        # InternVL 3.5 uses Qwen3 as the LLM backbone — LoRA target
        # modules match Qwen3 attention naming.
        self.lora_target_modules = lora_target_modules or [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ]
        self.gradient_checkpointing = gradient_checkpointing
        self.image_size = image_size

        # Translate ``max_pixels`` → ``max_num_tiles`` if not given
        # explicitly. One tile is image_size² pixels.
        if max_num_tiles is not None:
            self.max_num_tiles = int(max_num_tiles)
        elif max_pixels is not None:
            self.max_num_tiles = max(1, int(max_pixels // (image_size * image_size)))
        else:
            self.max_num_tiles = DEFAULT_MAX_NUM_TILES
        self.min_num_tiles = DEFAULT_MIN_NUM_TILES
        self.min_pixels = min_pixels  # retained for introspection only
        self.max_pixels = max_pixels

        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.model = None
        self.tokenizer = None
        self.processor = None  # InternVL has no unified processor; kept
        # for VLMBase attribute symmetry — consumers that only call
        # ``prepare_input`` / ``generate`` do not need it.

    @classmethod
    def build_worker_preprocessor(
        cls, model_name: str, **config
    ) -> InternVLWorkerPreprocessor:
        """Return a picklable DataLoader worker preprocessor."""
        image_size = int(config.get("image_size", DEFAULT_IMAGE_SIZE))
        max_num_tiles = config.get("max_num_tiles")
        if max_num_tiles is None and config.get("max_pixels") is not None:
            max_num_tiles = max(
                1, int(config["max_pixels"] // (image_size * image_size))
            )
        return InternVLWorkerPreprocessor(
            model_name=model_name,
            image_size=image_size,
            max_num_tiles=max_num_tiles or DEFAULT_MAX_NUM_TILES,
        )

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Download weights, apply quantization + LoRA, assign self.model."""
        from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

        quantization_config = None
        load_kwargs: Dict[str, Any] = {
            "dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }
        if self.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif self.load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
            load_kwargs["device_map"] = "auto"

        # flash-attn speeds up training but is an optional dep; fall
        # back to the default attention impl if it can't be imported.
        try:
            import flash_attn  # noqa: F401
            load_kwargs["use_flash_attn"] = True
        except Exception:
            load_kwargs["use_flash_attn"] = False

        self.model = AutoModel.from_pretrained(self.model_name, **load_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True, use_fast=False
        )

        # ``model.chat()`` sets ``img_context_token_id`` before every
        # call. In the training path we bypass chat() and call the
        # model's forward directly, so the attribute must be wired up
        # once at load time — otherwise the model's forward computes
        # ``selected = input_ids == self.img_context_token_id`` against
        # ``None`` and gets a Python ``False`` instead of a bool
        # tensor, blowing up with ``'bool' object has no attribute
        # 'sum'`` on the first training step.
        IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
        self.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids(
            IMG_CONTEXT_TOKEN
        )

        # No quantization + no device_map means we need to move
        # manually. Quantized loads already go through accelerate.
        if quantization_config is None:
            self.model = self.model.to(self.device)
        self.model.eval()

        if self.use_lora:
            self._apply_lora()

        # prepare_model_for_kbit_training upcasts every non-quantized
        # layer to fp32 for training numerical stability. Left alone,
        # that triggers BF16/FP32 mismatches at every module boundary
        # between the bf16 vision tower and the fp32 LLM embedding.
        # LoRA adapters stay fp32 (PEFT manages that); everything else
        # gets cast back to bf16 — matches the inference dtype without
        # hurting LoRA gradient flow.
        if self.load_in_4bit or self.load_in_8bit:
            self._cast_non_lora_to_bf16()

        if self.gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

    def _cast_non_lora_to_bf16(self) -> None:
        """Cast every non-LoRA, non-quantized param back to bf16.

        Context: ``prepare_model_for_kbit_training`` upcasts everything
        it doesn't quantize to fp32 for training numerical stability.
        For a multimodal model that means the ViT, the vision→LLM
        projector, the LLM embedding layer, and every LayerNorm end
        up fp32 while the 4-bit LLM weights stay in bnb's Params4bit
        storage. The bf16 pixel_values → fp32 weights mismatch
        surfaces as a BF16/FP32 error at the first conv / layernorm /
        index_put after the vision tower.

        LoRA adapters (names contain ``lora_`` in PEFT) are left in
        fp32 — that's what PEFT expects for stable gradient flow.
        Quantized params (``uint8`` / ``int8`` storage) are skipped;
        ``.to(dtype=bf16)`` on them either no-ops or raises.
        """
        for name, param in self.model.named_parameters():
            if "lora_" in name:
                continue
            if param.dtype in (torch.uint8, torch.int8):
                continue
            if param.dtype == torch.float32:
                param.data = param.data.to(dtype=torch.bfloat16)
        for name, buf in self.model.named_buffers():
            if buf.dtype == torch.float32:
                buf.data = buf.data.to(dtype=torch.bfloat16)

    def _apply_lora(self) -> None:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        if self.load_in_4bit or self.load_in_8bit:
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=self.gradient_checkpointing,
            )

        # NOTE: no ``task_type``. If we set CAUSAL_LM, PEFT wraps the
        # model in PeftModelForCausalLM, whose forward injects
        # ``inputs_embeds=None`` into every call. InternVLChatModel's
        # forward signature does not accept that keyword (input goes
        # through ``input_ids`` + ``pixel_values`` + ``image_flags``),
        # so training crashes with TypeError on the first step. The
        # bare PeftModel wrapper is pass-through, which is what we
        # want for a multimodal model with a custom chat-style forward.
        lora_cfg = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias="none",
            target_modules=self.lora_target_modules,
        )
        self.model = get_peft_model(self.model, lora_cfg)

    # ------------------------------------------------------------------
    # Trainable parameters
    # ------------------------------------------------------------------

    def get_trainable_parameters(self) -> Iterable[torch.nn.Parameter]:
        assert self.model is not None, "call load_model() first"
        return [p for p in self.model.parameters() if p.requires_grad]

    def print_trainable_parameters(self) -> None:
        assert self.model is not None, "call load_model() first"
        if hasattr(self.model, "print_trainable_parameters"):
            self.model.print_trainable_parameters()
            return
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        pct = 100.0 * trainable / total if total else 0.0
        print(
            f"trainable params: {trainable:,} || all params: {total:,} || "
            f"trainable%: {pct:.4f}"
        )

    # ------------------------------------------------------------------
    # Training I/O
    # ------------------------------------------------------------------

    def prepare_input(
        self,
        image,
        question: str,
        bbox: Optional[List[int]] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """Return CPU tensors for a single (image, question) training sample.

        The ``<image>`` token in the user message is expanded by the
        tokenizer into ``num_image_token`` placeholders per tile at
        training time; ``pixel_values`` carries the stacked tile tensor.
        """
        if self.tokenizer is None:
            raise RuntimeError("call load_model() before prepare_input()")

        img = Image.open(image) if isinstance(image, (str, Path)) else image
        img = img.convert("RGB") if img.mode != "RGB" else img
        if bbox is not None:
            img = _draw_bbox(img, bbox, (255, 0, 0), 3)

        pixel_values = load_image_tiles(
            img,
            input_size=self.image_size,
            max_num=self.max_num_tiles,
        ).to(dtype=torch.bfloat16)

        num_tiles = pixel_values.shape[0]
        # InternVL expects the <image> token to be expanded to
        # num_image_token placeholders per tile. The model config
        # exposes ``num_image_token`` (usually 256).
        num_image_token = getattr(self.model.config, "num_image_token", 256) if self.model is not None else 256
        image_placeholder = "<img>" + "<IMG_CONTEXT>" * (num_image_token * num_tiles) + "</img>"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {
                "role": "user",
                "content": question.replace("<image>", image_placeholder)
                if "<image>" in question
                else f"{image_placeholder}\n{question}",
            }
        )
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        tokenized = self.tokenizer(prompt, return_tensors="pt", padding=False)
        image_flags = torch.ones(num_tiles, dtype=torch.long)
        return {
            "input_ids": tokenized["input_ids"][0],
            "attention_mask": tokenized["attention_mask"][0],
            "pixel_values": pixel_values,
            "image_flags": image_flags,
        }

    def forward(self, **inputs) -> Dict[str, Any]:
        """Teacher-forcing forward. Expects ``input_ids``, ``attention_mask``,
        ``pixel_values``, ``image_flags``, and ``labels``."""
        assert self.model is not None, "call load_model() first"
        outputs = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            pixel_values=inputs.get("pixel_values"),
            image_flags=inputs.get("image_flags"),
            labels=inputs.get("labels"),
            return_dict=True,
        )
        return {
            "loss": getattr(outputs, "loss", None),
            "logits": getattr(outputs, "logits", None),
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

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
        """Run single-image inference via the model's native ``chat``
        entry-point, which takes care of InternVL's chat template."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("call load_model() before generate()")

        img = Image.open(image) if isinstance(image, (str, Path)) else image
        img = img.convert("RGB") if img.mode != "RGB" else img
        if bbox is not None:
            img = _draw_bbox(img, bbox, box_color, box_thickness)

        pixel_values = load_image_tiles(
            img, input_size=self.image_size, max_num=self.max_num_tiles
        ).to(dtype=torch.bfloat16)

        # Honour system_prompt via InternVL's native mechanism: setting
        # ``model.system_message`` is the documented way per the model
        # card. We restore the previous value to avoid leaking state.
        previous_system = getattr(self.model, "system_message", None)
        if system_prompt:
            self.model.system_message = system_prompt

        try:
            pixel_values_device = pixel_values.to(next(self.model.parameters()).device)
            gen_cfg = {
                "max_new_tokens": int(max_new_tokens),
                "do_sample": False,
            }
            prompt = question if "<image>" in question else f"<image>\n{question}"
            response = self.model.chat(
                self.tokenizer,
                pixel_values_device,
                prompt,
                gen_cfg,
            )
        finally:
            if system_prompt:
                self.model.system_message = previous_system

        return response

    # ------------------------------------------------------------------
    # Adapter save/load
    # ------------------------------------------------------------------

    def save_adapter(self, path) -> None:
        """Persist LoRA weights + tokenizer alongside metadata."""
        from peft import PeftModel

        if self.model is None:
            raise RuntimeError("call load_model() before save_adapter()")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if isinstance(self.model, PeftModel):
            self.model.save_pretrained(path)
        else:
            # Non-LoRA save — falls back to full model save (rare in
            # YoloGen since we always use LoRA, but kept for parity).
            self.model.save_pretrained(path)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(path)

    def load_adapter(self, path) -> None:
        """Re-attach a LoRA adapter onto an already-loaded base model."""
        from peft import PeftModel

        if self.model is None:
            raise RuntimeError("call load_model() before load_adapter()")
        self.model = PeftModel.from_pretrained(self.model, str(path))
        self.model.eval()


def create_internvl(
    model_name: str = "OpenGVLab/InternVL3_5-4B",
    **kwargs,
) -> InternVLM:
    """Convenience constructor mirroring :func:`create_qwen_vlm`."""
    return InternVLM(model_name=model_name, **kwargs)
