"""GLM-4.6V (Z.AI) adapter with QLoRA support.

Implements :class:`~yologen.models.vlm.base.VLMBase` for the z.ai GLM-V
family. The shape is simpler than InternVL: GLM ships as a first-class
HuggingFace model class (``Glm4vForConditionalGeneration``), the
processor exposes the standard ``apply_chat_template`` entry point, and
inference is plain ``model.generate()`` — no ``model.chat`` shim, no
``trust_remote_code`` modules to download.

The primary target for YoloGen is ``zai-org/GLM-4.6V-Flash`` (9B dense)
— it fits on a consumer GPU under QLoRA and pairs with the Qwen-VL /
InternVL 3.5 sizes the rest of the framework supports.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from PIL import Image

from yologen.models.vlm.base import VLMBase, VLMWorkerPreprocessor, register_vlm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _draw_bbox(
    image: Image.Image,
    bbox,
    box_color: Tuple[int, int, int] = (255, 0, 0),
    box_thickness: int = 3,
) -> Image.Image:
    """Render a red (or custom) box on a copy so the VLM sees the
    same visual prompt YoloGen uses across every adapter."""
    from PIL import ImageDraw

    out = image.copy()
    draw = ImageDraw.Draw(out)
    x1, y1, x2, y2 = [int(v) for v in bbox]
    draw.rectangle([x1, y1, x2, y2], outline=tuple(box_color), width=int(box_thickness))
    return out


def _load_glm_model_cls():
    """Return the GLM-4V HF model class, or a graceful fallback.

    The dedicated ``Glm4vForConditionalGeneration`` was added in
    transformers 4.55+. Older transformers pins can still load the
    weights via ``AutoModelForImageTextToText``.
    """
    try:
        from transformers import Glm4vForConditionalGeneration
        return Glm4vForConditionalGeneration
    except ImportError:
        try:
            from transformers import AutoModelForImageTextToText
            return AutoModelForImageTextToText
        except ImportError:
            from transformers import AutoModelForVision2Seq
            return AutoModelForVision2Seq


# ---------------------------------------------------------------------------
# Worker preprocessor
# ---------------------------------------------------------------------------


class GLMWorkerPreprocessor(VLMWorkerPreprocessor):
    """Picklable preprocessor for GLM-V DataLoader workers.

    Lazy-loads :class:`transformers.AutoProcessor` on first call so
    each worker performs one load, not once per sample. The processor
    handles the chat template, image preprocessing, and tokenization
    in a single call — no qwen-style ``process_vision_info`` detour.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._processor = None

    def _ensure_processor(self):
        if self._processor is None:
            from transformers import AutoProcessor
            self._processor = AutoProcessor.from_pretrained(
                self.model_name, trust_remote_code=True
            )
        return self._processor

    def __call__(self, image_path, question, system_prompt, answer):
        processor = self._ensure_processor()

        prompt_messages = []
        if system_prompt:
            prompt_messages.append({"role": "system", "content": system_prompt})
        prompt_messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": str(image_path)},
                    {"type": "text", "text": question},
                ],
            }
        )

        # GLM-4.6V emits a ``<think>…</think>`` reasoning block by
        # default. For a Yes/No verifier we want a direct answer; the
        # chat template recognises ``enable_thinking=False`` and
        # injects ``/nothink`` + an empty ``<think></think>`` turn so
        # the model skips reasoning — applied symmetrically to both
        # the prompt and full tokenisation below.
        tpl_kwargs = dict(
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )

        # Double-tokenize to build answer-only labels.
        prompt_inputs = processor.apply_chat_template(
            prompt_messages, add_generation_prompt=True, **tpl_kwargs
        )
        prompt_inputs.pop("token_type_ids", None)
        prompt_len = prompt_inputs["input_ids"].shape[-1]

        full_messages = prompt_messages + [
            {"role": "assistant", "content": answer}
        ]
        full_inputs = processor.apply_chat_template(
            full_messages, add_generation_prompt=False, **tpl_kwargs
        )
        full_inputs.pop("token_type_ids", None)

        labels = full_inputs["input_ids"].clone()
        labels[:, :prompt_len] = -100
        full_inputs["labels"] = labels
        return dict(full_inputs)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

SUPPORTED_MODELS = {
    "zai-org/GLM-4.6V-Flash",
    "zai-org/GLM-4.6V",
    "zai-org/GLM-4.5V",
    "zai-org/GLM-4.1V-9B-Thinking",
}


@register_vlm(r"zai-org/GLM-4\.[0-9]+V.*")
class GLMVLM(VLMBase):
    """Z.AI GLM-4.x-V adapter with optional 4-bit/8-bit QLoRA fine-tuning.

    Example::

        vlm = GLMVLM(model_name="zai-org/GLM-4.6V-Flash",
                     load_in_4bit=True, use_lora=True)
        vlm.load_model()
        out = vlm.generate(image="frame.jpg",
                           question="Is there a weapon in the red box?",
                           bbox=[100, 200, 300, 400])
    """

    def __init__(
        self,
        model_name: str = "zai-org/GLM-4.6V-Flash",
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
        # Kept for API parity with Qwen/InternVL; GLM's processor has
        # its own image-size handling and ignores these.
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
    ) -> None:
        if not model_name.startswith("zai-org/"):
            raise ValueError(
                f"GLMVLM was asked to load {model_name!r}, which does not "
                f"look like a zai-org GLM-V model id."
            )

        self.model_name = model_name
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules or [
            "q_proj", "k_proj", "v_proj", "o_proj",
        ]
        self.gradient_checkpointing = gradient_checkpointing
        self.min_pixels = min_pixels
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
        self.processor = None
        self.tokenizer = None

    # ------------------------------------------------------------------

    @classmethod
    def build_worker_preprocessor(
        cls, model_name: str, **config
    ) -> GLMWorkerPreprocessor:
        return GLMWorkerPreprocessor(model_name=model_name)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        from transformers import AutoConfig, AutoProcessor, BitsAndBytesConfig

        ModelCls = _load_glm_model_cls()

        # GLM-4.6V-Flash ships a config where text_config.rope_scaling
        # is ``None``, but transformers' Glm4vForConditionalGeneration
        # (4.57.x) dereferences ``self.rope_scaling["mrope_section"]``
        # in its attention forward. Patch the config before loading so
        # training and generation don't crash on the first forward.
        # head_dim = 4096 / 32 = 128; mrope_section [16, 24, 24] mirrors
        # the Qwen2/3-VL split and sums to the right half-dim window.
        config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
        if getattr(config, "text_config", None) is not None:
            if getattr(config.text_config, "rope_scaling", None) is None:
                config.text_config.rope_scaling = {
                    "rope_type": "default",
                    "mrope_section": [16, 24, 24],
                }

        load_kwargs: Dict[str, Any] = {
            "config": config,
            "dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }
        quantization_config = None
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

        self.model = ModelCls.from_pretrained(self.model_name, **load_kwargs)
        self.processor = AutoProcessor.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        # GLM exposes tokenizer via processor.tokenizer (standard HF
        # processor shape); alias for VLMBase attribute symmetry.
        self.tokenizer = getattr(self.processor, "tokenizer", None)

        if quantization_config is None:
            self.model = self.model.to(self.device)
        self.model.eval()

        if self.use_lora:
            self._apply_lora()

        # prepare_model_for_kbit_training upcasts non-quantized layers
        # to fp32; that includes vision tower, multimodal projector,
        # LLM embedding, and every LayerNorm. With bf16 pixel_values
        # the fp32 conv / layernorm weights trigger mismatches. LoRA
        # itself stays fp32 for gradient stability.
        if self.load_in_4bit or self.load_in_8bit:
            self._cast_non_lora_to_bf16()

        if self.gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

    def _apply_lora(self) -> None:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        if self.load_in_4bit or self.load_in_8bit:
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=self.gradient_checkpointing,
            )

        # GLM-V has a ConditionalGeneration forward that accepts
        # inputs_embeds, so PEFT's CausalLM wrapper is compatible —
        # we can set task_type="CAUSAL_LM" (unlike InternVL, whose
        # custom forward did not).
        lora_cfg = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=self.lora_target_modules,
        )
        self.model = get_peft_model(self.model, lora_cfg)

    def _cast_non_lora_to_bf16(self) -> None:
        """See :meth:`InternVLM._cast_non_lora_to_bf16` — same rationale."""
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
        if self.processor is None:
            raise RuntimeError("call load_model() before prepare_input()")

        img = Image.open(image) if isinstance(image, (str, Path)) else image
        img = img.convert("RGB") if img.mode != "RGB" else img
        if bbox is not None:
            img = _draw_bbox(img, bbox, (255, 0, 0), 3)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": question},
                ],
            }
        )
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,  # see worker preprocessor for rationale
        )
        inputs.pop("token_type_ids", None)
        return dict(inputs)

    def forward(self, **inputs) -> Dict[str, Any]:
        assert self.model is not None, "call load_model() first"
        inputs = {k: v for k, v in inputs.items() if k != "token_type_ids"}
        outputs = self.model(**inputs, return_dict=True)
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
        max_new_tokens: int = 64,
    ) -> str:
        if self.model is None or self.processor is None:
            raise RuntimeError("call load_model() before generate()")

        img = Image.open(image) if isinstance(image, (str, Path)) else image
        img = img.convert("RGB") if img.mode != "RGB" else img
        if bbox is not None:
            img = _draw_bbox(img, bbox, box_color, box_thickness)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": question},
                ],
            }
        )
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(next(self.model.parameters()).device)
        inputs.pop("token_type_ids", None)

        # GLM-4.6V's generation config recommends stochastic sampling
        # with a mild repetition penalty; pure greedy decoding makes
        # the base model loop on short Yes/No prompts (observed during
        # bring-up: "So we need to answer Yes or No only..." repeated
        # indefinitely). Fine-tuned adapters still converge on direct
        # answers, but keep the decoder out of a loop meanwhile.
        with torch.no_grad():
            gen_ids = self.model.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                do_sample=True,
                temperature=0.6,
                top_p=0.6,
                repetition_penalty=1.1,
            )
        prompt_len = inputs["input_ids"].shape[1]
        text = self.processor.decode(
            gen_ids[0][prompt_len:], skip_special_tokens=True
        )
        # GLM-4.6V wraps its final answer in ``<|begin_of_box|>…<|end_of_box|>``
        # tokens (retained through ``skip_special_tokens=True`` because
        # they're regular vocab entries) and — when thinking is on —
        # prefixes the whole thing with ``<think>…</think>``. Peel off
        # both so callers that want a plain Yes/No verdict
        # (VLMPredictor.verify) receive just the decoded answer.
        if "</think>" in text:
            text = text.split("</think>", 1)[1]
        elif text.lstrip().startswith("<think>"):
            text = text.split("<think>", 1)[1]
        for tok in ("<|begin_of_box|>", "<|end_of_box|>"):
            text = text.replace(tok, "")
        return text.strip()

    # ------------------------------------------------------------------
    # Adapter save / load
    # ------------------------------------------------------------------

    def save_adapter(self, path) -> None:
        from peft import PeftModel

        if self.model is None:
            raise RuntimeError("call load_model() before save_adapter()")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if isinstance(self.model, PeftModel):
            self.model.save_pretrained(path)
        else:
            self.model.save_pretrained(path)
        if self.processor is not None:
            self.processor.save_pretrained(path)

    def load_adapter(self, path) -> None:
        from peft import PeftModel

        if self.model is None:
            raise RuntimeError("call load_model() before load_adapter()")
        self.model = PeftModel.from_pretrained(self.model, str(path))
        self.model.eval()


def create_glm_vlm(
    model_name: str = "zai-org/GLM-4.6V-Flash",
    **kwargs,
) -> GLMVLM:
    """Convenience constructor mirroring :func:`create_qwen_vlm`."""
    return GLMVLM(model_name=model_name, **kwargs)
