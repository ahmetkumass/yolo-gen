"""
Hard Negative Miner for VLM Verifier training.

Generic, detector-free hard negative generation via embedding-similarity mining.
Finds image regions that look similar to ground-truth positives but do NOT
contain the target object.

Core idea:
    1. For each GT bbox, compute a positive embedding using a pretrained
       self-supervised vision encoder (DINOv2 by default).
    2. Slide candidate windows within concentric rings around the GT bbox.
       (FPs typically concentrate near the positive — YOLO's failure geometry.)
    3. Embed each candidate, compute cosine similarity against the positive.
    4. Keep candidates with similarity in the "hard negative" range (default
       0.25-0.50 for DINOv2): visually similar but not identical.

Safeguards (multi-layer TP-leakage protection):
    Layer 1: IoU exclusion against ALL GT bboxes in the image
             (multi-weapon images don't produce self-contaminated negatives).
    Layer 2: Hard upper similarity cap. Anything above the upper bound is
             treated as "possibly the target itself" and dropped.
    Layer 3: Diversity filter between mined regions (drops duplicates).
    Layer 4 (opt-in): VLM zero-shot double-check. Ask a base VLM
             "Is there a <class> in this region? Yes/No". Reject if it says Yes.

Generic across domains: swap the backbone or class name and the same
algorithm works for weapon / defect / medical / vehicle-damage verification.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class MinedRegion:
    """A single mined hard-negative region for one image.

    Coordinates are in pixel space (xyxy). ``source_gt_idx`` refers to which
    GT bbox of the image spawned this mining (the positive we compared
    against); ``ring_idx`` records which ring-distance band the region was
    sampled from.
    """
    bbox: tuple[int, int, int, int]
    similarity: float
    ring_idx: int
    source_gt_idx: int
    source_gt_class: str

    def to_dict(self) -> dict:
        return {
            "bbox": list(self.bbox),
            "similarity": float(self.similarity),
            "ring_idx": int(self.ring_idx),
            "source_gt_idx": int(self.source_gt_idx),
            "source_gt_class": self.source_gt_class,
        }


@dataclass
class GTBox:
    """Parsed GT bbox in pixel xyxy form with class info."""
    bbox: tuple[int, int, int, int]
    class_id: int
    class_name: str


@dataclass
class MiningStats:
    """Aggregate statistics across a mining pass."""
    images_processed: int = 0
    gt_bboxes_seen: int = 0
    candidates_scanned: int = 0
    candidates_after_iou_filter: int = 0
    candidates_in_similarity_range: int = 0
    candidates_after_diversity: int = 0
    candidates_after_vlm_verify: int = 0
    regions_emitted: int = 0
    ring_distribution: Dict[int, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "images_processed": self.images_processed,
            "gt_bboxes_seen": self.gt_bboxes_seen,
            "candidates_scanned": self.candidates_scanned,
            "candidates_after_iou_filter": self.candidates_after_iou_filter,
            "candidates_in_similarity_range": self.candidates_in_similarity_range,
            "candidates_after_diversity": self.candidates_after_diversity,
            "candidates_after_vlm_verify": self.candidates_after_vlm_verify,
            "regions_emitted": self.regions_emitted,
            "ring_distribution": dict(self.ring_distribution),
        }


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def iou(a: Sequence[int], b: Sequence[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)


def _center(bbox: Sequence[int]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


# ---------------------------------------------------------------------------
# Core mining engine
# ---------------------------------------------------------------------------

class NegativeMiner:
    """Ring-based spatial + embedding-similarity hard-negative miner.

    Models are lazy-loaded on first use so constructing an instance is cheap.

    Example:
        miner = NegativeMiner({
            "enabled": True,
            "embedding_model": "facebook/dinov2-base",
            "rings": [1.0, 3.0, 6.0],
            "similarity_range": [0.25, 0.50],
            "max_per_image": 3,
        })
        for img_path, gt_boxes in dataset:
            regions = miner.mine_image(Image.open(img_path), gt_boxes)
    """

    DEFAULT_CONFIG: Dict = {
        "enabled": False,
        "embedding_model": "facebook/dinov2-base",
        "rings": [1.0, 3.0, 6.0],
        "stride_per_ring": [30, 50, 80],
        "similarity_range": [0.25, 0.50],
        "max_per_image": 3,
        "exclude_iou_with_any_gt": 0.1,
        "diversity_iou_threshold": 0.3,
        "batch_size": 32,
        "device": "auto",
        "random_seed": 0,
        "vlm_verify": {
            "enabled": False,
            "model": "Qwen/Qwen3-VL-4B-Instruct",
            "prompt": "Is there a {class} in this region? Answer Yes or No.",
            "reject_on_yes": True,
            "max_new_tokens": 4,
        },
    }

    def __init__(self, config: Optional[Dict] = None):
        cfg = dict(self.DEFAULT_CONFIG)
        if config:
            for k, v in config.items():
                if k == "vlm_verify" and isinstance(v, dict):
                    merged = dict(cfg["vlm_verify"])
                    merged.update(v)
                    cfg["vlm_verify"] = merged
                else:
                    cfg[k] = v
        self.config = cfg
        self._validate()

        self.rings: List[float] = list(cfg["rings"])
        self.strides: List[int] = list(cfg["stride_per_ring"])
        if len(self.strides) != len(self.rings):
            # Broadcast or repeat the last stride to match ring count.
            if len(self.strides) == 1:
                self.strides = self.strides * len(self.rings)
            else:
                tail = [self.strides[-1]] * (len(self.rings) - len(self.strides))
                self.strides = (self.strides + tail)[: len(self.rings)]

        sim_lo, sim_hi = cfg["similarity_range"]
        self.sim_lo, self.sim_hi = float(sim_lo), float(sim_hi)
        self.max_per_image: int = int(cfg["max_per_image"])
        self.exclude_iou: float = float(cfg["exclude_iou_with_any_gt"])
        self.diversity_iou: float = float(cfg["diversity_iou_threshold"])
        self.batch_size: int = int(cfg["batch_size"])
        self.rng = random.Random(int(cfg["random_seed"]))

        # Lazy-loaded
        self._torch = None
        self._model = None
        self._processor = None
        self._device: Optional[str] = None
        self._vlm_model = None
        self._vlm_processor = None

    def _validate(self) -> None:
        cfg = self.config
        if not cfg["rings"]:
            raise ValueError("negative_mining.rings must be non-empty")
        if any(r <= 0 for r in cfg["rings"]):
            raise ValueError("negative_mining.rings must be strictly positive")
        if sorted(cfg["rings"]) != list(cfg["rings"]):
            raise ValueError("negative_mining.rings must be ascending")
        lo, hi = cfg["similarity_range"]
        if not (0.0 <= lo < hi <= 1.0):
            raise ValueError(
                f"similarity_range must satisfy 0 <= lo < hi <= 1, got [{lo}, {hi}]"
            )
        if cfg["max_per_image"] < 0:
            raise ValueError("max_per_image must be >= 0")

    # ------------------------------------------------------------------
    # Model loading (lazy)
    # ------------------------------------------------------------------

    def _ensure_embedding_model(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoImageProcessor

        self._torch = torch
        if self.config["device"] == "auto":
            if torch.cuda.is_available():
                self._device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"
        else:
            self._device = self.config["device"]

        name = self.config["embedding_model"]
        self._model = AutoModel.from_pretrained(name).to(self._device).eval()
        self._processor = AutoImageProcessor.from_pretrained(name)

    def _ensure_vlm(self) -> None:
        if self._vlm_model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        name = self.config["vlm_verify"]["model"]
        self._vlm_processor = AutoProcessor.from_pretrained(name, trust_remote_code=True)
        self._vlm_model = AutoModelForCausalLM.from_pretrained(
            name, dtype=torch.bfloat16, trust_remote_code=True, device_map="auto",
        )
        self._vlm_model.eval()

    # ------------------------------------------------------------------
    # Embedding + similarity
    # ------------------------------------------------------------------

    def _embed(self, crops: List[Image.Image]):
        self._ensure_embedding_model()
        torch = self._torch
        with torch.no_grad():
            inputs = self._processor(images=crops, return_tensors="pt").to(self._device)
            out = self._model(**inputs)
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                emb = out.pooler_output
            else:
                emb = out.last_hidden_state[:, 0]
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb

    # ------------------------------------------------------------------
    # Candidate generation (ring-based sliding window)
    # ------------------------------------------------------------------

    def _candidates_for_gt(
        self,
        image_w: int,
        image_h: int,
        gt_bbox: Sequence[int],
    ) -> List[tuple[tuple[int, int, int, int], int]]:
        """Return list of (candidate_bbox, ring_idx).

        Rings are concentric annuli around the GT bbox's center, measured in
        multiples of the GT bbox's min-side. Candidates share the GT bbox's
        width/height so crops are comparable at the embedding stage.
        """
        x1, y1, x2, y2 = gt_bbox
        W_b, H_b = x2 - x1, y2 - y1
        if W_b <= 1 or H_b <= 1:
            return []

        cx, cy = _center(gt_bbox)
        min_side = float(min(W_b, H_b))

        # Ring bands in pixels: 0..r0*min_side, r0*min_side..r1*min_side, ...
        bands: List[tuple[float, float]] = []
        prev = 0.0
        for r in self.rings:
            outer = r * min_side
            bands.append((prev, outer))
            prev = outer

        out: List[tuple[tuple[int, int, int, int], int]] = []
        for ring_idx, (inner, outer) in enumerate(bands):
            stride = self.strides[ring_idx]
            # Axis-aligned search box around the center: extend by `outer`
            search_x1 = int(max(0, cx - outer))
            search_y1 = int(max(0, cy - outer))
            search_x2 = int(min(image_w - 1, cx + outer))
            search_y2 = int(min(image_h - 1, cy + outer))

            for y in range(search_y1, search_y2 - H_b + 1, stride):
                for x in range(search_x1, search_x2 - W_b + 1, stride):
                    cand = (x, y, x + W_b, y + H_b)
                    cand_cx, cand_cy = _center(cand)
                    dist = math.hypot(cand_cx - cx, cand_cy - cy)
                    # Ring membership: inner <= dist < outer
                    if dist < inner or dist >= outer:
                        continue
                    out.append((cand, ring_idx))
        return out

    # ------------------------------------------------------------------
    # Public mining API
    # ------------------------------------------------------------------

    def mine_image(
        self,
        image: Image.Image,
        gt_boxes: List[GTBox],
        stats: Optional[MiningStats] = None,
    ) -> List[MinedRegion]:
        """Mine hard negatives for a single image.

        Args:
            image: PIL image (will be converted to RGB).
            gt_boxes: list of GTBox (pixel xyxy, class info). All will be used
                for (a) IoU exclusion and (b) per-GT positive-embedding
                reference.

        Returns:
            List of MinedRegion, capped at ``max_per_image`` and sorted by
            similarity (highest first).
        """
        if not gt_boxes:
            return []
        if image.mode != "RGB":
            image = image.convert("RGB")

        image_w, image_h = image.size
        all_gt_xyxy = [g.bbox for g in gt_boxes]

        if stats is not None:
            stats.images_processed += 1
            stats.gt_bboxes_seen += len(gt_boxes)

        mined_per_gt: List[List[MinedRegion]] = []

        for gt_idx, gt in enumerate(gt_boxes):
            pos_crop = image.crop(gt.bbox)
            pos_emb = self._embed([pos_crop])  # [1, D]

            # Collect candidates across rings
            cand_info = self._candidates_for_gt(image_w, image_h, gt.bbox)
            if stats is not None:
                stats.candidates_scanned += len(cand_info)

            # IoU exclusion against ALL GT bboxes (multi-object safety)
            filtered: List[tuple[tuple[int, int, int, int], int]] = []
            for cand, ring_idx in cand_info:
                if any(iou(cand, g) > self.exclude_iou for g in all_gt_xyxy):
                    continue
                filtered.append((cand, ring_idx))
            if stats is not None:
                stats.candidates_after_iou_filter += len(filtered)

            if not filtered:
                mined_per_gt.append([])
                continue

            # Batch-embed + similarity
            gt_candidates: List[MinedRegion] = []
            for batch_start in range(0, len(filtered), self.batch_size):
                batch = filtered[batch_start:batch_start + self.batch_size]
                crops = [image.crop(b[0]) for b in batch]
                emb = self._embed(crops)
                sims = (emb @ pos_emb.T).squeeze(-1).cpu().tolist()
                for (cand, ring_idx), sim in zip(batch, sims):
                    if self.sim_lo <= sim < self.sim_hi:
                        gt_candidates.append(MinedRegion(
                            bbox=cand,
                            similarity=float(sim),
                            ring_idx=ring_idx,
                            source_gt_idx=gt_idx,
                            source_gt_class=gt.class_name,
                        ))

            if stats is not None:
                stats.candidates_in_similarity_range += len(gt_candidates)

            # Rank by similarity desc, then apply diversity filter
            gt_candidates.sort(key=lambda r: -r.similarity)
            kept = self._diversity_filter(gt_candidates)
            if stats is not None:
                stats.candidates_after_diversity += len(kept)

            mined_per_gt.append(kept)

        # Merge all GT-level results, then cap at max_per_image via diversity
        merged = [r for lst in mined_per_gt for r in lst]
        merged.sort(key=lambda r: -r.similarity)
        final = self._diversity_filter(merged)[: self.max_per_image]

        # Optional VLM verification (slow) — rejects regions the VLM
        # classifies as containing the target.
        if self.config["vlm_verify"]["enabled"] and final:
            final = self._vlm_filter(image, final)
            if stats is not None:
                stats.candidates_after_vlm_verify += len(final)

        if stats is not None:
            stats.regions_emitted += len(final)
            for r in final:
                stats.ring_distribution[r.ring_idx] = (
                    stats.ring_distribution.get(r.ring_idx, 0) + 1
                )

        return final

    def mine_dataset(
        self,
        images: List[tuple[Path, List[GTBox]]],
        show_progress: bool = True,
    ) -> tuple[Dict[Path, List[MinedRegion]], MiningStats]:
        """Run mining across a list of (image_path, gt_boxes) pairs.

        Models are loaded once up-front; per-image calls reuse them.
        """
        stats = MiningStats()
        results: Dict[Path, List[MinedRegion]] = {}
        iterator = tqdm(images, desc="Mining hard negatives") if show_progress else images

        for img_path, gt_boxes in iterator:
            try:
                img = Image.open(img_path)
            except Exception:
                continue
            results[img_path] = self.mine_image(img, gt_boxes, stats=stats)

        return results, stats

    # ------------------------------------------------------------------
    # Internal filters
    # ------------------------------------------------------------------

    def _diversity_filter(self, regions: List[MinedRegion]) -> List[MinedRegion]:
        """Greedy non-max-suppression-like filter on bbox IoU.

        Operates on similarity-sorted list; drops any region that overlaps
        a previously kept region with IoU > diversity_iou.
        """
        kept: List[MinedRegion] = []
        for r in regions:
            if any(iou(r.bbox, k.bbox) > self.diversity_iou for k in kept):
                continue
            kept.append(r)
        return kept

    def _vlm_filter(
        self,
        image: Image.Image,
        regions: List[MinedRegion],
    ) -> List[MinedRegion]:
        """Drop mined regions the VLM thinks contain the target class."""
        self._ensure_vlm()
        cfg = self.config["vlm_verify"]
        prompt_tmpl = cfg["prompt"]
        keep: List[MinedRegion] = []

        for r in regions:
            crop = image.crop(r.bbox)
            prompt = prompt_tmpl.replace("{class}", r.source_gt_class)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": crop},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text = self._vlm_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = self._vlm_processor(
                text=[text], images=[crop], return_tensors="pt", padding=True,
            ).to(self._vlm_model.device)
            out = self._vlm_model.generate(
                **inputs, max_new_tokens=cfg["max_new_tokens"], do_sample=False,
            )
            raw = self._vlm_processor.batch_decode(
                out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
            )[0].strip()
            first = (raw.split() or [""])[0].strip(".,!?\"'").lower()
            if cfg["reject_on_yes"] and first == "yes":
                continue
            keep.append(r)
        return keep
