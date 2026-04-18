"""VLM Dataset Generator.

Two generation modes:

* ``qa_format: "descriptive"`` (default, backwards-compatible)
    Template-based Q&A pairs describing what is inside the red bbox.

* ``qa_format: "binary_multiclass"``
    For each GT bbox, emit one Yes/No sample per class defined in the
    dataset. Matching class → "Yes", others → "No". Produces free
    cross-class hard negatives with zero extra effort.

Optional: **hard negative mining** — detector-free spatial similarity
mining that adds out-of-box "No" samples from regions of the image that
visually resemble positives but do not contain any target. Requires
``qa_format == "binary_multiclass"``. See :mod:`yologen.data.negative_miner`.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from yologen.data.negative_miner import GTBox, NegativeMiner


class VLMDatasetGenerator:
    """
    Generate VLM training dataset from YOLO labels.

    Creates Q&A pairs with red box visual grounding.

    Example:
        generator = VLMDatasetGenerator("dataset.yaml")
        generator.generate(output_dir="vlm_data/")

        # With external config
        generator = VLMDatasetGenerator("dataset.yaml", vlm_config={...})
    """

    def __init__(
        self,
        data_yaml: str,
        box_thickness: int = 3,
        vlm_config: Optional[Dict] = None,
    ):
        """
        Initialize generator.

        Args:
            data_yaml: Path to dataset.yaml
            box_thickness: Red box thickness in pixels
            vlm_config: Optional VLM dataset config (overrides dataset.yaml settings)
        """
        self.data_yaml = Path(data_yaml)
        self.config = self._load_config()
        self.data_path = self.config['path']

        # Class names
        class_names = self.config.get('names', {})
        if isinstance(class_names, dict):
            self.class_names = [class_names.get(i, f'class_{i}') for i in range(len(class_names))]
        else:
            self.class_names = list(class_names)

        # VLM dataset config - prefer external config, fallback to dataset.yaml
        if vlm_config is None:
            vlm_config = self.config.get('vlm_dataset', {})
        self.vlm_config = vlm_config

        self.prompt_templates = vlm_config.get('prompts', [])
        self.class_details = vlm_config.get('details', {})
        self.class_prompts: Dict[str, str] = vlm_config.get('class_prompts', {})
        self.class_questions: Dict[str, str] = vlm_config.get('class_questions', {})
        self.qa_format = vlm_config.get('qa_format', 'descriptive')
        if self.qa_format not in ('descriptive', 'binary_multiclass'):
            raise ValueError(
                f"Unknown qa_format: {self.qa_format!r}. "
                "Expected 'descriptive' or 'binary_multiclass'."
            )
        self.box_thickness = vlm_config.get('box_thickness', box_thickness)

        # Hard-negative mining config
        self.negative_mining_config: Dict = vlm_config.get('negative_mining', {})

        self._validate_binary_multiclass()
        self._validate_negative_mining()

        # Box color - config uses BGR (OpenCV), we store RGB for PIL
        box_color_bgr = vlm_config.get('box_color', [0, 0, 255])  # Default red in BGR
        self.box_color_bgr = tuple(box_color_bgr)  # For OpenCV (training)
        self.box_color_rgb = (box_color_bgr[2], box_color_bgr[1], box_color_bgr[0])  # For PIL (inference)

        # System prompt
        self.system_prompt = vlm_config.get('system_prompt',
            "You are an object detection assistant. "
            "Identify objects in red marked areas clearly and confidently."
        )

    def _active_class_names(self) -> List[str]:
        """Class names filtered to those used in training (``unused`` excluded)."""
        return [c for c in self.class_names if c.lower() != "unused"]

    def _validate_binary_multiclass(self) -> None:
        """Check that binary_multiclass config is complete."""
        if self.qa_format != "binary_multiclass":
            return
        active = self._active_class_names()
        if not active:
            raise ValueError(
                "binary_multiclass requires at least one non-'unused' class in "
                "dataset names."
            )
        missing_prompts = [c for c in active if c not in self.class_prompts]
        if missing_prompts:
            raise ValueError(
                f"binary_multiclass requires `class_prompts` for every class. "
                f"Missing: {missing_prompts}. "
                f"Add `vlm_dataset.class_prompts: {{<class>: <system prompt>}}` "
                f"to your config."
            )
        # Auto-fill class_questions with a sensible default when missing.
        for c in active:
            if c not in self.class_questions:
                self.class_questions[c] = (
                    f"Is there a {c} in the red bounding box? Answer Yes or No."
                )

    def _validate_negative_mining(self) -> None:
        """Ensure negative_mining is only enabled alongside binary_multiclass."""
        if not self.negative_mining_config.get("enabled"):
            return
        if self.qa_format != "binary_multiclass":
            raise ValueError(
                "vlm_dataset.negative_mining.enabled=true requires "
                "vlm_dataset.qa_format='binary_multiclass'. "
                "Spatial hard negatives only make sense with binary Yes/No "
                "supervision; descriptive captions cannot be a 'No' label."
            )

    def _load_config(self) -> Dict:
        """Load dataset configuration."""
        with open(self.data_yaml, 'r') as f:
            config = yaml.safe_load(f)

        base_path = self.data_yaml.parent
        if 'path' in config:
            data_path = Path(config['path'])
            if not data_path.is_absolute():
                config['path'] = (base_path / data_path).resolve()
            else:
                config['path'] = data_path
        else:
            config['path'] = base_path.resolve()

        return config

    def generate(
        self,
        output_dir: str = None,
        force: bool = False,
    ) -> Dict[str, int]:
        """
        Generate VLM dataset.

        Args:
            output_dir: Output directory
            force: Force regeneration

        Returns:
            Statistics dictionary
        """
        if output_dir is None:
            output_path = self.data_path / 'vlm'
        else:
            output_path = Path(output_dir)

        # Check existing
        if not force and (output_path / 'train.jsonl').exists():
            print("VLM dataset exists. Use force=True to regenerate.")
            return {'status': 'skipped'}

        stats: Dict = {
            'train': 0,
            'val': 0,
            'images': 0,
            'qa_pairs': 0,
            'positives': 0,
            'cross_class_negatives': 0,
            'hard_negatives': 0,
        }

        # Lazy-instantiate the miner once; it loads models on first mine_image call.
        miner: Optional[NegativeMiner] = None
        hnm_stats_per_split: Dict[str, Dict] = {}
        if self.negative_mining_config.get("enabled"):
            miner = NegativeMiner(self.negative_mining_config)

        for split in ['train', 'val']:
            # Find directories
            if (self.data_path / 'images' / split).exists():
                img_dir = self.data_path / 'images' / split
                label_dir = self.data_path / 'labels' / split
            elif (self.data_path / split / 'images').exists():
                img_dir = self.data_path / split / 'images'
                label_dir = self.data_path / split / 'labels'
            else:
                continue

            out_img_dir = output_path / 'images' / split
            out_img_dir.mkdir(parents=True, exist_ok=True)

            # Find images
            img_files = []
            for ext in ['.jpg', '.jpeg', '.png']:
                img_files.extend(img_dir.glob(f'*{ext}'))
                img_files.extend(img_dir.glob(f'*{ext.upper()}'))

            samples = []
            for img_path in tqdm(img_files, desc=f"  {split}"):
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                img_h, img_w = img.shape[:2]

                # Load labels
                label_path = label_dir / f"{img_path.stem}.txt"
                boxes = self._parse_labels(label_path)

                if not boxes:
                    continue

                # Count classes
                class_counts = {}
                for box in boxes:
                    if box['class_id'] < len(self.class_names):
                        name = self.class_names[box['class_id']]
                        if name.lower() != 'unused':
                            class_counts[name] = class_counts.get(name, 0) + 1

                if not class_counts:
                    continue

                # Grounded Q&A (per box)
                for box_idx, box in enumerate(boxes):
                    class_id = box['class_id']
                    if class_id >= len(self.class_names):
                        continue

                    class_name = self.class_names[class_id]
                    if class_name.lower() == 'unused':
                        continue

                    bbox = self._xywh_to_xyxy(box, img_w, img_h)
                    img_with_box = self._draw_red_box(img, bbox)

                    out_img_name = f"{img_path.stem}_box{box_idx}.jpg"
                    cv2.imwrite(str(out_img_dir / out_img_name), img_with_box)
                    stats['images'] += 1

                    if self.qa_format == "binary_multiclass":
                        new_samples = self._binary_multiclass_samples(
                            class_name=class_name,
                            class_id=class_id,
                            bbox=bbox,
                            split=split,
                            out_img_name=out_img_name,
                        )
                        for s in new_samples:
                            samples.append(s)
                            stats['qa_pairs'] += 1
                            if s["answer"] == "Yes":
                                stats['positives'] += 1
                            else:
                                stats['cross_class_negatives'] += 1
                    else:
                        qa_pairs = self._generate_grounded_qa(class_name)
                        for qa in qa_pairs:
                            samples.append({
                                "image": f"images/{split}/{out_img_name}",
                                "question": qa["q"],
                                "answer": qa["a"],
                                "class": class_name,
                                "class_id": class_id,
                                "bbox": bbox,
                                "type": "grounded",
                            })
                            stats['qa_pairs'] += 1

            # Hard-negative mining (optional, binary_multiclass only)
            if miner is not None:
                neg_samples, hnm_stats = self._mine_and_render_negatives(
                    miner=miner,
                    split=split,
                    img_dir=img_dir,
                    label_dir=label_dir,
                    out_img_dir=out_img_dir,
                )
                hnm_stats_per_split[split] = hnm_stats
                for s in neg_samples:
                    samples.append(s)
                    stats['qa_pairs'] += 1
                    stats['hard_negatives'] += 1

            # Save
            random.shuffle(samples)
            with open(output_path / f'{split}.jsonl', 'w') as f:
                for sample in samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')

            stats[split] = len(samples)

        # Persist per-split hard-negative mining statistics for reproducibility.
        if hnm_stats_per_split:
            with open(output_path / 'hnm_stats.json', 'w') as f:
                json.dump({
                    'config': {
                        k: v for k, v in self.negative_mining_config.items()
                        if k != 'vlm_verify'
                    },
                    'vlm_verify_enabled': bool(
                        self.negative_mining_config.get('vlm_verify', {}).get('enabled')
                    ),
                    'per_split': hnm_stats_per_split,
                }, f, indent=2)

        # Save config (includes all settings for inference consistency)
        # Note: box_color is saved as RGB for PIL compatibility
        box_color_rgb = self.box_color_rgb  # Already converted to RGB
        with open(output_path / 'config.json', 'w') as f:
            json.dump({
                'qa_format': self.qa_format,
                'class_names': self.class_names,
                'prompt_templates': self.prompt_templates,
                'class_details': self.class_details,
                'class_prompts': self.class_prompts,
                'class_questions': self.class_questions,
                'box_thickness': self.box_thickness,
                'box_color': list(box_color_rgb),  # RGB format
                'system_prompt': self.system_prompt,
                'negative_mining': {
                    k: v for k, v in self.negative_mining_config.items()
                    if k not in ('vlm_verify',)
                } if self.negative_mining_config.get('enabled') else {'enabled': False},
                'stats': stats,
            }, f, indent=2)

        return stats

    def _parse_labels(self, label_path: Path) -> List[Dict]:
        """Parse YOLO label file."""
        if not label_path.exists():
            return []

        boxes = []
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('---'):
                    break
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        boxes.append({
                            'class_id': int(parts[0]),
                            'x': float(parts[1]),
                            'y': float(parts[2]),
                            'w': float(parts[3]),
                            'h': float(parts[4]),
                        })
                    except ValueError:
                        continue
        return boxes

    def _xywh_to_xyxy(self, box: Dict, img_w: int, img_h: int) -> List[int]:
        """Convert normalized xywh to pixel xyxy."""
        x, y, w, h = box['x'], box['y'], box['w'], box['h']
        x1 = int((x - w / 2) * img_w)
        y1 = int((y - h / 2) * img_h)
        x2 = int((x + w / 2) * img_w)
        y2 = int((y + h / 2) * img_h)
        return [max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)]

    def _draw_red_box(self, image: np.ndarray, bbox: List[int]) -> np.ndarray:
        """Draw colored box on image (uses BGR from config)."""
        img = image.copy()
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), self.box_color_bgr, self.box_thickness)
        return img

    def _get_detail(self, class_name: str) -> str:
        """Get class-specific detail sentence."""
        if self.class_details:
            if class_name in self.class_details:
                return random.choice(self.class_details[class_name])
            if class_name.lower() in self.class_details:
                return random.choice(self.class_details[class_name.lower()])
        return ""

    def _fill_template(self, template: str, **kwargs) -> str:
        """Fill template with placeholders and normalize whitespace.

        Empty placeholders (e.g. `{detail}` when no class details are
        configured) otherwise leave double spaces or trailing whitespace
        in the rendered answer, which shows up verbatim as supervision.
        """
        import re
        result = template
        for key, value in kwargs.items():
            result = result.replace(f"{{{key}}}", str(value))
        # Collapse runs of internal whitespace and strip edges.
        return re.sub(r"\s+", " ", result).strip()

    def _generate_grounded_qa(self, class_name: str) -> List[Dict]:
        """Generate grounded Q&A for single object using templates from config."""
        qa = []
        c = class_name.lower()
        detail = self._get_detail(class_name)

        if not self.prompt_templates:
            raise ValueError("No prompt templates defined in vlm_dataset.prompts config")

        for tmpl in self.prompt_templates:
            q_template = tmpl.get('question', '')
            a_template = tmpl.get('answer', '')

            # Skip templates that need multiple objects (global)
            if '{objects}' in q_template or '{count_text}' in q_template:
                continue

            q = self._fill_template(q_template, **{
                'class': c,
                'detail': detail,
                'yes_no': 'Yes',
                'explanation': f'there is a {c} in the marked area',
            })
            a = self._fill_template(a_template, **{
                'class': c,
                'detail': detail,
                'yes_no': 'Yes',
                'explanation': f'there is a {c} in the marked area',
            })
            if q and a and '{' not in q and '{' not in a:
                qa.append({"q": q, "a": a})

        return qa


    # ------------------------------------------------------------------
    # Binary multiclass helpers
    # ------------------------------------------------------------------

    def _binary_multiclass_samples(
        self,
        class_name: str,
        class_id: int,
        bbox: List[int],
        split: str,
        out_img_name: str,
    ) -> List[Dict]:
        """Emit one Yes/No sample per active class for a single GT bbox.

        Matching class → ``"Yes"`` (positive); others → ``"No"`` (cross-class
        hard negative, free of charge).
        """
        samples: List[Dict] = []
        for target in self._active_class_names():
            is_match = target.lower() == class_name.lower()
            samples.append({
                "image": f"images/{split}/{out_img_name}",
                "system": self.class_prompts[target],
                "question": self.class_questions[target],
                "answer": "Yes" if is_match else "No",
                "class": class_name,
                "class_id": class_id,
                "target": target,
                "bbox": bbox,
                "type": "positive" if is_match else "cross_class_negative",
            })
        return samples

    # ------------------------------------------------------------------
    # Hard-negative mining helpers
    # ------------------------------------------------------------------

    def _mine_and_render_negatives(
        self,
        miner: NegativeMiner,
        split: str,
        img_dir: Path,
        label_dir: Path,
        out_img_dir: Path,
    ) -> tuple[List[Dict], Dict]:
        """Run the miner across a split and render each mined region.

        Returns:
            A pair ``(samples, mining_stats)`` where ``samples`` is a list of
            Yes/No jsonl-ready records and ``mining_stats`` is the serialised
            :class:`yologen.data.negative_miner.MiningStats` dict for this
            split. Each mined region spawns one sample per active class
            (answer ``"No"`` across the board — spatial negatives belong to
            no class).
        """
        # Collect (image_path, [GTBox, ...]) pairs the miner can consume.
        from PIL import Image as PILImage

        pairs: List[tuple[Path, List[GTBox]]] = []
        img_files: List[Path] = []
        for ext in ['.jpg', '.jpeg', '.png']:
            img_files.extend(img_dir.glob(f'*{ext}'))
            img_files.extend(img_dir.glob(f'*{ext.upper()}'))

        for img_path in img_files:
            raw_boxes = self._parse_labels(label_dir / f"{img_path.stem}.txt")
            if not raw_boxes:
                continue
            with PILImage.open(img_path) as im:
                img_w, img_h = im.size
            gt_list: List[GTBox] = []
            for b in raw_boxes:
                if b['class_id'] >= len(self.class_names):
                    continue
                cname = self.class_names[b['class_id']]
                if cname.lower() == 'unused':
                    continue
                xyxy = tuple(self._xywh_to_xyxy(b, img_w, img_h))
                gt_list.append(GTBox(bbox=xyxy, class_id=b['class_id'], class_name=cname))
            if gt_list:
                pairs.append((img_path, gt_list))

        if not pairs:
            return [], {"images_processed": 0}

        results, mining_stats = miner.mine_dataset(pairs, show_progress=True)
        mining_stats_dict = mining_stats.to_dict()
        print(f"  [HNM/{split}] {mining_stats_dict}")

        # Render each mined region + emit Yes/No samples across classes.
        out_samples: List[Dict] = []
        for img_path, regions in results.items():
            if not regions:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            for reg_idx, region in enumerate(regions):
                rendered = self._draw_red_box(img, list(region.bbox))
                out_name = f"{img_path.stem}_neg{reg_idx}.jpg"
                cv2.imwrite(str(out_img_dir / out_name), rendered)

                for target in self._active_class_names():
                    out_samples.append({
                        "image": f"images/{split}/{out_name}",
                        "system": self.class_prompts[target],
                        "question": self.class_questions[target],
                        "answer": "No",
                        "class": None,  # not a positive class
                        "class_id": None,
                        "target": target,
                        "bbox": list(region.bbox),
                        "type": "hard_negative",
                        "hnm_metadata": {
                            "similarity": round(region.similarity, 4),
                            "ring_idx": region.ring_idx,
                            "source_gt_class": region.source_gt_class,
                        },
                    })
        return out_samples, mining_stats_dict


def generate_vlm_dataset(
    data_yaml: str,
    output_dir: str = None,
    force: bool = False,
) -> Dict[str, int]:
    """
    Generate VLM dataset (convenience function).

    Args:
        data_yaml: Path to dataset.yaml
        output_dir: Output directory
        force: Force regeneration

    Returns:
        Statistics dictionary
    """
    generator = VLMDatasetGenerator(data_yaml)
    return generator.generate(output_dir=output_dir, force=force)
