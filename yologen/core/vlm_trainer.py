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
    """Dataset for VLM training with Q&A pairs."""

    def __init__(self, jsonl_path: str, image_root: str, max_samples: int = None):
        """
        Args:
            jsonl_path: Path to JSONL file with Q&A samples
            image_root: Root directory for images
            max_samples: Maximum number of samples to use (None = use all)
        """
        import random

        self.image_root = Path(image_root)
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'image_path': str(self.image_root / sample['image']),
            'question': sample['question'],
            'answer': sample['answer'],
            'class': sample.get('class', ''),
            'bbox': sample.get('bbox', []),  # Empty list instead of None
        }


def _clear_gpu_memory():
    """Clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class VLMTrainer:
    """
    VLM Trainer with QLoRA support.

    Example:
        trainer = VLMTrainer(model="Qwen/Qwen2.5-VL-7B-Instruct")
        trainer.train(data="vlm_data/", epochs=3)
        trainer.save("adapter/")
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        precision: str = "4bit",
        lora_r: int = 64,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        gradient_checkpointing: bool = True,
        device: str = "",
    ):
        """
        Initialize VLM trainer.

        Args:
            model: HuggingFace model name
            precision: Quantization (4bit, 8bit, fp16)
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
            lora_dropout: LoRA dropout
            gradient_checkpointing: Enable gradient checkpointing
            device: Device to use
        """
        self.model_name = model
        self.precision = precision
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.gradient_checkpointing = gradient_checkpointing
        self.device = device

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

        # Dataset
        train_dataset = VLMDataset(str(train_jsonl), str(data_path), max_samples=max_samples)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )

        # Clear memory and load VLM
        _clear_gpu_memory()
        self._load_vlm()

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
                    # Get sample
                    image_path = batch['image_path'][0] if isinstance(batch['image_path'], list) else batch['image_path']
                    question = batch['question'][0] if isinstance(batch['question'], list) else batch['question']
                    answer = batch['answer'][0] if isinstance(batch['answer'], list) else batch['answer']

                    # Prepare input (with system_prompt for consistent training)
                    inputs = self.vlm.prepare_input(
                        image=image_path,
                        question=question,
                        bbox=None,
                        system_prompt=system_prompt,
                    )

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
    model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    epochs: int = 3,
    precision: str = "4bit",
    lora_r: int = 64,
    lr: float = 1e-5,
    **kwargs,
) -> Dict[str, Any]:
    """
    Train VLM (convenience function).

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
