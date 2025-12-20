"""
VLM Dataset Generator

Creates Q&A pairs for VLM training from YOLO detection labels.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import yaml
from tqdm import tqdm


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
        self.class_prompts = vlm_config.get('class_prompts', {})  # Class-specific prompts
        self.qa_format = vlm_config.get('qa_format', 'descriptive')  # 'descriptive' or 'binary'
        self.box_thickness = vlm_config.get('box_thickness', box_thickness)

        # Box color - config uses BGR (OpenCV), we store RGB for PIL
        box_color_bgr = vlm_config.get('box_color', [0, 0, 255])  # Default red in BGR
        self.box_color_bgr = tuple(box_color_bgr)  # For OpenCV (training)
        self.box_color_rgb = (box_color_bgr[2], box_color_bgr[1], box_color_bgr[0])  # For PIL (inference)

        # System prompt
        self.system_prompt = vlm_config.get('system_prompt',
            "You are an object detection assistant. "
            "Identify objects in red marked areas clearly and confidently."
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

        stats = {'train': 0, 'val': 0, 'images': 0, 'qa_pairs': 0}

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

                    # Generate Q&A
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

                # Global Q&A
                img_all_boxes = img.copy()
                for box in boxes:
                    if box['class_id'] < len(self.class_names):
                        if self.class_names[box['class_id']].lower() != 'unused':
                            bbox = self._xywh_to_xyxy(box, img_w, img_h)
                            img_all_boxes = self._draw_red_box(img_all_boxes, bbox)

                out_img_name_global = f"{img_path.stem}_all.jpg"
                cv2.imwrite(str(out_img_dir / out_img_name_global), img_all_boxes)

                global_qa = self._generate_global_qa(class_counts)
                for qa in global_qa:
                    samples.append({
                        "image": f"images/{split}/{out_img_name_global}",
                        "question": qa["q"],
                        "answer": qa["a"],
                        "class_counts": class_counts,
                        "type": "global",
                    })
                    stats['qa_pairs'] += 1

            # Save
            random.shuffle(samples)
            with open(output_path / f'{split}.jsonl', 'w') as f:
                for sample in samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')

            stats[split] = len(samples)

        # Save config (includes all settings for inference consistency)
        # Note: box_color is saved as RGB for PIL compatibility
        box_color_rgb = self.box_color_rgb  # Already converted to RGB
        with open(output_path / 'config.json', 'w') as f:
            json.dump({
                'class_names': self.class_names,
                'prompt_templates': self.prompt_templates,
                'class_details': self.class_details,
                'box_thickness': self.box_thickness,
                'box_color': list(box_color_rgb),  # RGB format
                'system_prompt': self.system_prompt,
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

    def _format_objects(self, class_counts: Dict[str, int]) -> str:
        """Format class counts as text."""
        items = []
        for name, count in class_counts.items():
            if count == 1:
                items.append(f"a {name.lower()}")
            else:
                items.append(f"{count} {name.lower()}s")
        if len(items) == 0:
            return "no objects"
        elif len(items) == 1:
            return items[0]
        elif len(items) == 2:
            return f"{items[0]} and {items[1]}"
        return ", ".join(items[:-1]) + f", and {items[-1]}"

    def _get_detail(self, class_name: str) -> str:
        """Get class-specific detail sentence."""
        if self.class_details:
            if class_name in self.class_details:
                return random.choice(self.class_details[class_name])
            if class_name.lower() in self.class_details:
                return random.choice(self.class_details[class_name.lower()])
        return ""

    def _fill_template(self, template: str, **kwargs) -> str:
        """Fill template with placeholders."""
        result = template
        for key, value in kwargs.items():
            result = result.replace(f"{{{key}}}", str(value))
        return result

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

    def _generate_global_qa(self, class_counts: Dict[str, int]) -> List[Dict]:
        """Generate global/image-level Q&A using templates from config."""
        qa = []
        objects = self._format_objects(class_counts)
        total = sum(class_counts.values())
        count_text = f"{'is' if total == 1 else 'are'} {total} object{'s' if total > 1 else ''}"

        # Get detail for first class
        first_class = list(class_counts.keys())[0] if class_counts else ''
        detail = self._get_detail(first_class) if first_class else ''

        if not self.prompt_templates:
            raise ValueError("No prompt templates defined in vlm_dataset.prompts config")

        for tmpl in self.prompt_templates:
            q_template = tmpl.get('question', '')
            a_template = tmpl.get('answer', '')

            # Only use templates that work with multiple objects
            if '{class}' in q_template and '{objects}' not in q_template:
                continue

            q = self._fill_template(q_template, **{
                'class': first_class,
                'objects': objects,
                'count_text': count_text,
                'detail': detail,
                'yes_no': 'Yes',
                'explanation': f'there are {objects} in this image',
            })
            a = self._fill_template(a_template, **{
                'class': first_class,
                'objects': objects,
                'count_text': count_text,
                'detail': detail,
                'yes_no': 'Yes',
                'explanation': f'there are {objects} in this image',
            })
            if q and a and '{' not in q and '{' not in a:
                qa.append({"q": q, "a": a})

        return qa


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
