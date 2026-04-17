"""Unit tests for :mod:`yologen.data.negative_miner`.

Tests the algorithm's geometry, filters, and safeguards without downloading
any real model. The DINOv2 embedding call is monkey-patched with a
deterministic stub that returns similarity-controlled vectors so we can
assert end-to-end behavior.
"""

from __future__ import annotations

import math

import pytest
import torch
from PIL import Image

from yologen.data.negative_miner import (
    GTBox,
    MinedRegion,
    NegativeMiner,
    iou,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def blank_image() -> Image.Image:
    return Image.new("RGB", (640, 480), (128, 128, 128))


@pytest.fixture
def single_gt() -> list[GTBox]:
    return [GTBox(bbox=(300, 220, 340, 260), class_id=0, class_name="target")]


@pytest.fixture
def two_gts() -> list[GTBox]:
    return [
        GTBox(bbox=(100, 100, 140, 140), class_id=0, class_name="a"),
        GTBox(bbox=(400, 300, 440, 340), class_id=1, class_name="b"),
    ]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

class TestIoU:
    def test_identical_bboxes(self):
        assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)

    def test_disjoint_bboxes(self):
        assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0

    def test_partial_overlap(self):
        # 10x10 boxes with 5x5 overlap → intersection=25, union=175, IoU=1/7
        assert iou((0, 0, 10, 10), (5, 5, 15, 15)) == pytest.approx(25 / 175)


# ---------------------------------------------------------------------------
# Config & validation
# ---------------------------------------------------------------------------

class TestConfig:
    def test_defaults_applied_when_missing_keys(self):
        miner = NegativeMiner({"enabled": True})
        assert miner.config["rings"] == [3.0, 6.0]
        assert miner.sim_lo == 0.25
        assert miner.sim_hi == 0.50

    def test_rings_must_be_ascending(self):
        with pytest.raises(ValueError):
            NegativeMiner({"rings": [3.0, 1.0]})

    def test_rings_must_be_positive(self):
        with pytest.raises(ValueError):
            NegativeMiner({"rings": [1.0, 0.0, 3.0]})

    def test_similarity_range_bounds(self):
        with pytest.raises(ValueError):
            NegativeMiner({"similarity_range": [0.6, 0.4]})
        with pytest.raises(ValueError):
            NegativeMiner({"similarity_range": [-0.1, 0.5]})

    def test_stride_broadcasts_to_ring_count(self):
        miner = NegativeMiner({"rings": [1.0, 2.0, 3.0], "stride_per_ring": [50]})
        assert miner.strides == [50, 50, 50]

    def test_vlm_verify_defaults_are_merged(self):
        miner = NegativeMiner({"vlm_verify": {"enabled": True}})
        assert miner.config["vlm_verify"]["enabled"] is True
        # Untouched keys keep their defaults.
        assert miner.config["vlm_verify"]["reject_on_yes"] is True


# ---------------------------------------------------------------------------
# Ring-based candidate generation (no embedding needed)
# ---------------------------------------------------------------------------

class TestCandidateGeneration:
    def test_candidates_respect_image_bounds(self, single_gt):
        miner = NegativeMiner({"rings": [1.0, 3.0], "stride_per_ring": [20, 40]})
        candidates = miner._candidates_for_gt(640, 480, single_gt[0].bbox)
        for cand, _ in candidates:
            assert 0 <= cand[0] < 640
            assert 0 <= cand[1] < 480
            assert cand[2] <= 640
            assert cand[3] <= 480

    def test_candidates_sized_like_gt(self, single_gt):
        miner = NegativeMiner({"rings": [1.0], "stride_per_ring": [20]})
        candidates = miner._candidates_for_gt(640, 480, single_gt[0].bbox)
        assert candidates, "ring=1 around a 40x40 bbox should yield candidates"
        gt_w = single_gt[0].bbox[2] - single_gt[0].bbox[0]
        gt_h = single_gt[0].bbox[3] - single_gt[0].bbox[1]
        for cand, _ in candidates:
            assert cand[2] - cand[0] == gt_w
            assert cand[3] - cand[1] == gt_h

    def test_ring_distance_annulus(self, single_gt):
        """A candidate's center distance falls inside its ring's [inner, outer)."""
        miner = NegativeMiner({"rings": [1.0, 3.0], "stride_per_ring": [10, 10]})
        cands = miner._candidates_for_gt(640, 480, single_gt[0].bbox)
        gt_cx = (single_gt[0].bbox[0] + single_gt[0].bbox[2]) / 2
        gt_cy = (single_gt[0].bbox[1] + single_gt[0].bbox[3]) / 2
        min_side = min(
            single_gt[0].bbox[2] - single_gt[0].bbox[0],
            single_gt[0].bbox[3] - single_gt[0].bbox[1],
        )
        bands = [(0.0, 1.0 * min_side), (1.0 * min_side, 3.0 * min_side)]
        for cand, ring_idx in cands:
            ccx = (cand[0] + cand[2]) / 2
            ccy = (cand[1] + cand[3]) / 2
            dist = math.hypot(ccx - gt_cx, ccy - gt_cy)
            inner, outer = bands[ring_idx]
            assert inner <= dist < outer, (
                f"candidate at dist={dist:.1f} assigned to ring {ring_idx} "
                f"[{inner:.1f}, {outer:.1f})"
            )


# ---------------------------------------------------------------------------
# Full mining with stubbed embedder
# ---------------------------------------------------------------------------

def _install_stubbed_embedder(
    miner: NegativeMiner,
    similarity_map: dict[tuple[int, int, int, int], float],
    default: float = 0.05,
) -> None:
    """Monkey-patch ``_embed`` so mining runs without downloading DINOv2.

    Positive is always the embedding at (0, 0) direction; each candidate's
    similarity is looked up by bbox, so we can steer the filter deterministically.
    """
    # A 2-D vector embedding; positive = unit vector [1, 0].
    import torch

    miner._torch = torch

    def _fake_embed(crops):
        vectors = []
        for c in crops:
            # Identify the crop by its content hash to find its bbox. Simpler:
            # stash sim on the PIL crop via __dict__.
            sim = getattr(c, "_stub_sim", 1.0)
            theta = math.acos(max(-1.0, min(1.0, sim)))
            vectors.append([math.cos(theta), math.sin(theta)])
        return torch.tensor(vectors, dtype=torch.float32)

    miner._embed = _fake_embed  # type: ignore[method-assign]

    # Also patch Image.crop so each returned crop carries its target sim.
    original_crop = Image.Image.crop

    def _crop_with_sim(self, box):
        crop = original_crop(self, box)
        box_t = tuple(int(b) for b in box)
        crop._stub_sim = similarity_map.get(box_t, default)
        return crop

    Image.Image.crop = _crop_with_sim  # type: ignore[method-assign]
    miner._restore_crop = lambda: setattr(Image.Image, "crop", original_crop)


class TestMining:
    def test_empty_gt_returns_empty(self, blank_image):
        miner = NegativeMiner({"enabled": True})
        assert miner.mine_image(blank_image, []) == []

    def test_multi_gt_iou_exclusion(self, blank_image, two_gts):
        """Candidates overlapping ANY GT (not just the source GT) are dropped."""
        miner = NegativeMiner({
            "rings": [2.0, 4.0],
            "stride_per_ring": [30, 30],
            "exclude_iou_with_any_gt": 0.1,
            "similarity_range": [0.0, 1.0],  # accept everything for this test
            "max_per_image": 100,
        })
        # Stub: give every region a moderate similarity so only IoU gates matter.
        _install_stubbed_embedder(miner, {}, default=0.4)
        try:
            regions = miner.mine_image(blank_image, two_gts)
            for r in regions:
                for gt in two_gts:
                    assert iou(r.bbox, gt.bbox) <= 0.1 + 1e-6
        finally:
            miner._restore_crop()

    def test_similarity_window_filters(self, blank_image, single_gt):
        """Only candidates inside [sim_lo, sim_hi) survive."""
        miner = NegativeMiner({
            "rings": [1.0],
            "stride_per_ring": [20],
            "similarity_range": [0.3, 0.6],
            "max_per_image": 100,
            "diversity_iou_threshold": 1.0,  # don't dedupe for this test
        })
        cand_info = miner._candidates_for_gt(640, 480, single_gt[0].bbox)
        # Assign alternating sims inside/outside range.
        sim_map = {}
        for i, (cand, _) in enumerate(cand_info):
            sim_map[tuple(cand)] = 0.5 if i % 2 == 0 else 0.7  # 0.7 > 0.6 → drop
        _install_stubbed_embedder(miner, sim_map, default=0.0)
        try:
            regions = miner.mine_image(blank_image, single_gt)
            for r in regions:
                assert 0.3 <= r.similarity < 0.6
        finally:
            miner._restore_crop()

    def test_max_per_image_cap(self, blank_image, single_gt):
        miner = NegativeMiner({
            "rings": [1.0],
            "stride_per_ring": [10],
            "similarity_range": [0.2, 0.7],
            "max_per_image": 2,
        })
        _install_stubbed_embedder(miner, {}, default=0.45)
        try:
            regions = miner.mine_image(blank_image, single_gt)
            assert len(regions) <= 2
        finally:
            miner._restore_crop()

    def test_regions_sorted_by_similarity_desc(self, blank_image, single_gt):
        miner = NegativeMiner({
            "rings": [1.0],
            "stride_per_ring": [20],
            "similarity_range": [0.2, 0.8],
            "max_per_image": 5,
            "diversity_iou_threshold": 1.0,
        })
        cand_info = miner._candidates_for_gt(640, 480, single_gt[0].bbox)
        sim_map = {}
        for i, (cand, _) in enumerate(cand_info):
            sim_map[tuple(cand)] = 0.3 + (i % 5) * 0.1  # values 0.3, 0.4, 0.5, 0.6, 0.7
        _install_stubbed_embedder(miner, sim_map, default=0.0)
        try:
            regions = miner.mine_image(blank_image, single_gt)
            sims = [r.similarity for r in regions]
            assert sims == sorted(sims, reverse=True)
        finally:
            miner._restore_crop()


# ---------------------------------------------------------------------------
# MinedRegion / MiningStats dataclasses
# ---------------------------------------------------------------------------

class TestDataclasses:
    def test_mined_region_to_dict(self):
        r = MinedRegion(
            bbox=(10, 20, 30, 40),
            similarity=0.412,
            ring_idx=1,
            source_gt_idx=0,
            source_gt_class="target",
        )
        d = r.to_dict()
        assert d["bbox"] == [10, 20, 30, 40]
        assert d["similarity"] == pytest.approx(0.412)
        assert d["source_gt_class"] == "target"
