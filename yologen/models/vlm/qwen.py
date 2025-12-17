"""
Qwen VLM Model with QLoRA Support

Loads Qwen2.5-VL models with optional 4-bit/8-bit quantization and LoRA adapters.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List

import torch


class QwenVLM:
    """
    Qwen Vision-Language Model wrapper with QLoRA support.

    Example:
        vlm = QwenVLM(model_name="Qwen/Qwen2.5-VL-7B-Instruct", load_in_4bit=True)
        vlm.load_model()
        output = vlm.generate(image="image.jpg", question="What is this?")
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        use_lora: bool = True,
        lora_r: int = 64,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = None,
        gradient_checkpointing: bool = True,
        device: str = "",
    ):
        """
        Initialize Qwen VLM.

        Args:
            model_name: HuggingFace model name
            load_in_4bit: Use 4-bit quantization (QLoRA)
            load_in_8bit: Use 8-bit quantization
            use_lora: Apply LoRA adapters
            lora_r: LoRA rank
            lora_alpha: LoRA alpha scaling
            lora_dropout: LoRA dropout
            lora_target_modules: Modules to apply LoRA (default: q_proj, v_proj)
            gradient_checkpointing: Enable gradient checkpointing
            device: Device to use
        """
        self.model_name = model_name
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules or ["q_proj", "v_proj", "k_proj", "o_proj"]
        self.gradient_checkpointing = gradient_checkpointing

        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.model = None
        self.processor = None
        self.tokenizer = None

    def load_model(self):
        """Load model with quantization and LoRA."""
        from transformers import (
            AutoModelForVision2Seq,
            AutoProcessor,
            BitsAndBytesConfig,
        )
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        print(f"Loading {self.model_name}...")

        # Quantization config
        bnb_config = None
        if self.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        elif self.load_in_8bit:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )

        # Load model - use AutoModelForVision2Seq for automatic class detection
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
        }

        if bnb_config:
            model_kwargs["quantization_config"] = bnb_config
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = self.device

        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_name,
            **model_kwargs,
        )

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        self.tokenizer = self.processor.tokenizer

        # Prepare for training
        if self.load_in_4bit or self.load_in_8bit:
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=self.gradient_checkpointing,
            )

        # Apply LoRA
        if self.use_lora:
            lora_config = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=self.lora_target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_config)

        # Enable gradient checkpointing
        if self.gradient_checkpointing and not (self.load_in_4bit or self.load_in_8bit):
            self.model.gradient_checkpointing_enable()

        print(f"Model loaded on {self.device}")

    def print_trainable_parameters(self):
        """Print number of trainable parameters."""
        if self.model is None:
            print("Model not loaded")
            return

        trainable = 0
        total = 0

        for name, param in self.model.named_parameters():
            total += param.numel()
            if param.requires_grad:
                trainable += param.numel()

        print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    def get_trainable_parameters(self):
        """Get list of trainable parameters for optimizer."""
        if self.model is None:
            return []
        return [p for p in self.model.parameters() if p.requires_grad]

    def prepare_input(
        self,
        image: str,
        question: str,
        bbox: Optional[List[float]] = None,
        draw_box: bool = True,
        box_color: tuple = (255, 0, 0),
        box_thickness: int = 3,
        system_prompt: str = None,
    ) -> Dict[str, Any]:
        """
        Prepare input for the model.

        Args:
            image: Path to image file
            question: Question text
            bbox: Optional bounding box [x1, y1, x2, y2] in pixels
            draw_box: Whether to draw red box on image (matches training)
            box_color: RGB color for box (default red)
            box_thickness: Box line thickness
            system_prompt: System prompt from config (for consistent behavior)

        Returns:
            Dictionary with input tensors
        """
        from PIL import Image, ImageDraw
        from qwen_vl_utils import process_vision_info

        # Load image
        img = Image.open(image).convert("RGB")

        # Draw red box on image if bbox provided (matches training approach)
        if bbox and draw_box:
            draw = ImageDraw.Draw(img)
            x1, y1, x2, y2 = [int(v) for v in bbox]
            # Draw rectangle with thickness
            for i in range(box_thickness):
                draw.rectangle(
                    [x1 - i, y1 - i, x2 + i, y2 + i],
                    outline=box_color
                )

        # Build messages
        content = []

        # Add image (with red box drawn if bbox provided)
        content.append({
            "type": "image",
            "image": img,
        })

        # Add question
        content.append({
            "type": "text",
            "text": question,
        })

        messages = []

        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": content})

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Process vision info
        image_inputs, video_inputs = process_vision_info(messages)

        # Create inputs
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # Move to device
        inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        return inputs

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass through the model.

        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            pixel_values: Image pixel values
            image_grid_thw: Image grid dimensions
            labels: Labels for loss calculation

        Returns:
            Dictionary with loss and logits
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels,
        )

        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
        }

    def generate(
        self,
        image: str,
        question: str,
        bbox: Optional[List[float]] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        box_thickness: int = 3,
        box_color: tuple = (255, 0, 0),
        system_prompt: str = None,
    ) -> str:
        """
        Generate answer for image question.

        Args:
            image: Path to image file
            question: Question text
            bbox: Optional bounding box [x1, y1, x2, y2] in pixels
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            box_thickness: Red box thickness (must match training)
            box_color: RGB color tuple (must match training)
            system_prompt: System prompt from config

        Returns:
            Generated answer string
        """
        inputs = self.prepare_input(
            image, question, bbox,
            box_thickness=box_thickness,
            box_color=box_color,
            system_prompt=system_prompt,
        )

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
            )

        # Decode only new tokens
        input_len = inputs['input_ids'].shape[1]
        generated_ids = output_ids[0][input_len:]

        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    def save_adapter(self, path: str):
        """Save LoRA adapter weights."""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self.use_lora:
            self.model.save_pretrained(path)
        else:
            self.model.save_pretrained(path)

        # Save processor
        self.processor.save_pretrained(path)

        print(f"Adapter saved to {path}")

    def load_adapter(self, path: str):
        """Load LoRA adapter weights."""
        from peft import PeftModel

        if self.model is None:
            raise RuntimeError("Load base model first")

        path = Path(path)

        # Wrap base model with saved adapter
        self.model = PeftModel.from_pretrained(
            self.model,
            str(path),
            is_trainable=False,
        )
        print(f"Adapter loaded from {path}")


def create_qwen_vlm(
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    load_in_4bit: bool = True,
    load_in_8bit: bool = False,
    use_lora: bool = True,
    lora_r: int = 64,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    gradient_checkpointing: bool = True,
    **kwargs,
) -> QwenVLM:
    """
    Create Qwen VLM instance.

    Convenience function for creating QwenVLM with common settings.

    Args:
        model_name: HuggingFace model name
        load_in_4bit: Use 4-bit quantization
        load_in_8bit: Use 8-bit quantization
        use_lora: Apply LoRA adapters
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        gradient_checkpointing: Enable gradient checkpointing

    Returns:
        QwenVLM instance
    """
    return QwenVLM(
        model_name=model_name,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        gradient_checkpointing=gradient_checkpointing,
        **kwargs,
    )
