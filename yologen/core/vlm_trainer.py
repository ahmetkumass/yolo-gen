"""
VLM Trainer with QLoRA

Fine-tune Vision-Language Models using QLoRA for memory efficiency.
"""

import gc
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any, List

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class VLMDataset(Dataset):
    """Dataset for VLM training with Q&A pairs.

    Each worker lazy-loads its own HF processor so the heavy image /
    tokenizer work runs in parallel off the main thread.  The main
    training loop only has to move the already-prepared tensors to GPU.
    """

    def __init__(
        self,
        jsonl_path: str,
        image_root: str,
        model_name: str,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        default_system_prompt: Optional[str] = None,
        max_samples: int = None,
    ):
        """
        Args:
            jsonl_path: Path to JSONL file with Q&A samples
            image_root: Root directory for images
            model_name: HF model id (used to instantiate the processor)
            min_pixels / max_pixels: processor image-size bounds
            default_system_prompt: system prompt to use when a sample has
                no per-sample system field (descriptive mode)
            max_samples: optional cap on the sample count
        """
        import random

        self.image_root = Path(image_root)
        self.model_name = model_name
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.default_system_prompt = default_system_prompt or ""
        self.samples = []

        with open(jsonl_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

        # Limit samples if max_samples is specified
        if max_samples and len(self.samples) > max_samples:
            total = len(self.samples)
            random.shuffle(self.samples)
            self.samples = self.samples[:max_samples]
            print(f"Using {max_samples} samples (from {total} total)")

        self._processor = None  # lazy; one per worker

    def _get_processor(self):
        if self._processor is None:
            from transformers import AutoProcessor
            kwargs = {"trust_remote_code": True}
            if self.min_pixels is not None:
                kwargs["min_pixels"] = self.min_pixels
            if self.max_pixels is not None:
                kwargs["max_pixels"] = self.max_pixels
            self._processor = AutoProcessor.from_pretrained(self.model_name, **kwargs)
        return self._processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from qwen_vl_utils import process_vision_info
        processor = self._get_processor()
        sample = self.samples[idx]
        image_path = str(self.image_root / sample['image'])

        system = sample.get('system') or self.default_system_prompt
        question = sample['question']
        answer = sample['answer']

        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": question},
                ],
            },
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return {
            'inputs': dict(inputs),   # CPU tensors; caller moves to GPU
            'system': system,
            'answer': answer,
        }


def _vlm_collate_single(batch):
    """Trainer runs at batch_size=1; unwrap the single already-prepared sample.

    Default collate would try to stack per-sample tensors whose shapes
    can differ (e.g. image_grid_thw), which fails.  Multi-batch would
    need a processor(pad=...)-based collate and is not enabled yet.
    """
    if len(batch) != 1:
        raise RuntimeError(
            "VLMTrainer only supports batch_size=1 (use gradient_accumulation "
            "instead); got a batch of size " + str(len(batch))
        )
    return batch[0]


def _clear_gpu_memory():
    """Clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class VLMTrainer:
    """
    VLM Trainer with QLoRA support.

    Supports both Qwen 2.5 VL and Qwen 3 VL model families.

    Example:
        # Qwen 3 VL (recommended)
        trainer = VLMTrainer(model="Qwen/Qwen3-VL-4B-Instruct")

        # Qwen 2.5 VL
        trainer = VLMTrainer(model="Qwen/Qwen2.5-VL-7B-Instruct")

        trainer.train(data="vlm_data/", epochs=3)
        trainer.save("adapter/")
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen3-VL-4B-Instruct",
        precision: str = "4bit",
        lora_r: int = 64,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        gradient_checkpointing: bool = True,
        device: str = "",
        min_pixels: int = None,
        max_pixels: int = None,
    ):
        """
        Initialize VLM trainer.

        Args:
            model: HuggingFace model name (Qwen2.5-VL or Qwen3-VL)
            precision: Quantization (4bit, 8bit, fp16)
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
            lora_dropout: LoRA dropout
            gradient_checkpointing: Enable gradient checkpointing
            device: Device to use
            min_pixels: Min image pixels (auto-calculated if None)
            max_pixels: Max image pixels (auto-calculated if None)
        """
        self.model_name = model
        self.precision = precision
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.gradient_checkpointing = gradient_checkpointing
        self.device = device
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        self.vlm = None
        self.best_adapter = None

    def _load_vlm(self):
        """Load VLM model with QLoRA."""
        if self.vlm is not None:
            return

        # Add parent path for yologen imports
        parent_path = Path(__file__).parent.parent.parent.parent
        if str(parent_path) not in sys.path:
            sys.path.insert(0, str(parent_path))

        try:
            from yologen.models.vlm.qwen import create_qwen_vlm

            self.vlm = create_qwen_vlm(
                model_name=self.model_name,
                load_in_4bit=(self.precision == "4bit"),
                load_in_8bit=(self.precision == "8bit"),
                use_lora=True,
                lora_r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                gradient_checkpointing=self.gradient_checkpointing,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )
            self.vlm.load_model()
            self.vlm.print_trainable_parameters()

        except ImportError as e:
            raise ImportError(
                f"VLM dependencies not installed: {e}\n"
                "Install with: pip install transformers>=4.37.0 accelerate>=0.25.0 "
                "peft>=0.7.0 bitsandbytes>=0.41.0 qwen-vl-utils"
            )

    def train(
        self,
        data: str,
        epochs: int = 3,
        batch_size: int = 1,
        lr: float = 1e-5,
        gradient_accumulation: int = 4,
        max_grad_norm: float = 1.0,
        save_dir: str = None,
        name: str = None,
        resume: str = None,
        max_samples: int = None,
        num_workers: int = 4,
        pin_memory: bool = True,
    ) -> Dict[str, Any]:
        """
        Train VLM with QLoRA.

        Args:
            data: Directory containing train.jsonl and images/
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            gradient_accumulation: Gradient accumulation steps
            max_grad_norm: Max gradient norm for clipping
            save_dir: Output directory
            name: Experiment name
            resume: Path to adapter to resume from
            max_samples: Maximum training samples (None = use all)

        Returns:
            Training results
        """
        data_path = Path(data)
        train_jsonl = data_path / 'train.jsonl'

        if not train_jsonl.exists():
            raise FileNotFoundError(
                f"{train_jsonl} not found. "
                "Run generate_vlm_dataset first."
            )

        # Set output directory
        if save_dir is None:
            save_dir = Path(__file__).parent.parent.parent / "runs" / "vlm"
        else:
            save_dir = Path(save_dir)

        if name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"train_{timestamp}"

        output_dir = save_dir / name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load VLM config for system_prompt
        config_json = data_path / 'config.json'
        system_prompt = None
        if config_json.exists():
            with open(config_json) as f:
                vlm_config = json.load(f)
            system_prompt = vlm_config.get('system_prompt')
            print(f"Loaded system_prompt from config: {'yes' if system_prompt else 'no'}")

        # Load VLM first — the dataset needs to know the model name and
        # pixel limits so worker processes can instantiate matching
        # processors without re-inventing config.
        _clear_gpu_memory()
        self._load_vlm()

        # Dataset + DataLoader.  num_workers > 0 is a ~5-8x speedup: the
        # Qwen processor (apply_chat_template, process_vision_info,
        # image tokenization) is CPU-heavy and was previously blocking
        # the training step on the main thread.
        train_dataset = VLMDataset(
            jsonl_path=str(train_jsonl),
            image_root=str(data_path),
            model_name=self.model_name,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
            default_system_prompt=system_prompt,
            max_samples=max_samples,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=_vlm_collate_single if batch_size == 1 else None,
            persistent_workers=(num_workers > 0),
        )

        # Resume from adapter
        if resume:
            self.vlm.load_adapter(resume)

        # Optimizer
        trainable_params = self.vlm.get_trainable_parameters()
        optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)

        # Training loop
        global_step = 0
        best_loss = float('inf')
        loss_history = []

        for epoch in range(epochs):
            epoch_loss = 0
            num_steps = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

            for batch_idx, batch in enumerate(pbar):
                try:
                    # Dataset workers have already tokenized the image + text
                    # (with the sample's per-sample system prompt applied).
                    # Main thread only needs to move tensors to GPU.
                    inputs = batch['inputs']
                    device = self.vlm.model.device
                    inputs = {
                        k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                        for k, v in inputs.items()
                    }

                    # Labels
                    labels = inputs['input_ids'].clone()

                    # Forward pass
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        outputs = self.vlm.forward(
                            input_ids=inputs['input_ids'],
                            attention_mask=inputs['attention_mask'],
                            pixel_values=inputs.get('pixel_values'),
                            image_grid_thw=inputs.get('image_grid_thw'),
                            labels=labels,
                        )

                    if outputs['loss'] is not None:
                        loss = outputs['loss'] / gradient_accumulation
                        loss.backward()

                        # Gradient accumulation
                        if (batch_idx + 1) % gradient_accumulation == 0:
                            torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                            optimizer.step()
                            optimizer.zero_grad()
                            global_step += 1

                        epoch_loss += loss.item() * gradient_accumulation
                        num_steps += 1
                        loss_history.append(loss.item() * gradient_accumulation)

                        pbar.set_postfix({
                            'loss': f'{epoch_loss/num_steps:.4f}',
                            'step': global_step,
                        })

                    # Clear cache periodically
                    if batch_idx % 10 == 0:
                        torch.cuda.empty_cache()

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        torch.cuda.empty_cache()
                        gc.collect()
                        optimizer.zero_grad()
                        continue
                    raise e

            # Epoch summary
            if num_steps > 0:
                avg_loss = epoch_loss / num_steps

                # Save checkpoint
                checkpoint_dir = output_dir / f"checkpoint-epoch{epoch+1}"
                self.vlm.save_adapter(checkpoint_dir)

                # Save best
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_dir = output_dir / "best"
                    self.vlm.save_adapter(best_dir)
                    self.best_adapter = best_dir

        # Save final
        final_dir = output_dir / "final"
        self.vlm.save_adapter(final_dir)

        # Cleanup
        del self.vlm
        self.vlm = None
        _clear_gpu_memory()

        return {
            "best_adapter": str(self.best_adapter) if self.best_adapter else str(final_dir),
            "final_adapter": str(final_dir),
            "best_loss": best_loss,
            "loss_history": loss_history,
            "output_dir": str(output_dir),
        }

    def save(self, path: str):
        """Save current adapter."""
        if self.vlm is not None:
            self.vlm.save_adapter(path)


def train_vlm(
    data: str,
    model: str = "Qwen/Qwen3-VL-4B-Instruct",
    epochs: int = 3,
    precision: str = "4bit",
    lora_r: int = 64,
    lr: float = 1e-5,
    **kwargs,
) -> Dict[str, Any]:
    """
    Train VLM (convenience function).

    Supported models:
        Qwen 2.5 VL: Qwen/Qwen2.5-VL-3B-Instruct, Qwen/Qwen2.5-VL-7B-Instruct
        Qwen 3 VL: Qwen/Qwen3-VL-2B-Instruct, Qwen/Qwen3-VL-4B-Instruct, Qwen/Qwen3-VL-8B-Instruct

    Args:
        data: Path to VLM data directory
        model: VLM model name
        epochs: Number of epochs
        precision: Quantization precision
        lora_r: LoRA rank
        lr: Learning rate
        **kwargs: Additional training arguments

    Returns:
        Training results
    """
    trainer = VLMTrainer(
        model=model,
        precision=precision,
        lora_r=lora_r,
    )
    return trainer.train(
        data=data,
        epochs=epochs,
        lr=lr,
        **kwargs,
    )
