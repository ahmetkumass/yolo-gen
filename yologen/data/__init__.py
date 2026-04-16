"""YoloGen data module.

Public API:
    VLMDatasetGenerator   — generate VLM Q&A data from a YOLO dataset.
    generate_vlm_dataset  — convenience wrapper.

    NegativeMiner         — detector-free hard-negative miner (embedding
                             similarity + ring-based spatial constraint).
    MinedRegion           — a single hard-negative record produced by the
                             miner.
    GTBox                 — parsed ground-truth bbox passed into the miner.
    MiningStats           — aggregate mining statistics.
"""

from yologen.data.vlm_dataset import VLMDatasetGenerator, generate_vlm_dataset
from yologen.data.negative_miner import (
    GTBox,
    MinedRegion,
    MiningStats,
    NegativeMiner,
)

__all__ = [
    "VLMDatasetGenerator",
    "generate_vlm_dataset",
    "NegativeMiner",
    "MinedRegion",
    "GTBox",
    "MiningStats",
]
