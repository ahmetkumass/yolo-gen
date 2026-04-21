"""Unit tests for the VLM adapter factory and base contract.

These tests avoid loading any real model weights — they only exercise
the registry, pattern matching, and ABC enforcement.
"""
from __future__ import annotations

import pytest

from yologen.models.vlm import (
    InternVLM,
    QwenVLM,
    VLMBase,
    VLMWorkerPreprocessor,
    create_vlm,
    register_vlm,
    registered_adapters,
)


class TestRegistry:
    def test_qwen_is_registered(self):
        adapters = registered_adapters()
        names = [cls_name for _, cls_name in adapters]
        assert "QwenVLM" in names

    def test_internvl_is_registered(self):
        adapters = registered_adapters()
        names = [cls_name for _, cls_name in adapters]
        assert "InternVLM" in names

    def test_registered_patterns_are_strings(self):
        for pattern, cls_name in registered_adapters():
            assert isinstance(pattern, str)
            assert isinstance(cls_name, str)


class TestFactoryMatching:
    @pytest.mark.parametrize(
        "model_name",
        [
            "Qwen/Qwen3-VL-2B-Instruct",
            "Qwen/Qwen3-VL-4B-Instruct",
            "Qwen/Qwen3-VL-8B-Instruct",
            "Qwen/Qwen2.5-VL-3B-Instruct",
            "Qwen/Qwen2.5-VL-7B-Instruct",
        ],
    )
    def test_qwen_variants_route_to_qwen(self, model_name):
        v = create_vlm(model_name)
        assert isinstance(v, QwenVLM)
        assert v.model_name == model_name

    @pytest.mark.parametrize(
        "model_name",
        [
            "OpenGVLab/InternVL3_5-1B",
            "OpenGVLab/InternVL3_5-4B",
            "OpenGVLab/InternVL3_5-8B",
        ],
    )
    def test_internvl_variants_route_to_internvl(self, model_name):
        v = create_vlm(model_name)
        assert isinstance(v, InternVLM)
        assert v.model_name == model_name

    def test_qwen_and_internvl_patterns_do_not_overlap(self):
        # A Qwen id must not accidentally match the InternVL pattern
        # and vice versa.
        qwen = create_vlm("Qwen/Qwen3-VL-4B-Instruct")
        assert not isinstance(qwen, InternVLM)
        intern = create_vlm("OpenGVLab/InternVL3_5-4B")
        assert not isinstance(intern, QwenVLM)

    def test_unknown_model_raises_valueerror(self):
        with pytest.raises(ValueError) as exc:
            create_vlm("NonExistent/Model-1B")
        assert "No VLM adapter registered" in str(exc.value)
        # The error message should list registered patterns so users
        # can diagnose quickly.
        assert "patterns" in str(exc.value).lower()

    def test_factory_forwards_kwargs_to_adapter(self):
        v = create_vlm(
            "Qwen/Qwen3-VL-4B-Instruct",
            load_in_4bit=False,
            use_lora=False,
        )
        # Kwargs should have reached QwenVLM.__init__
        assert v.load_in_4bit is False
        assert v.use_lora is False


class TestRegisterDecorator:
    def test_register_rejects_non_vlmbase(self):
        with pytest.raises(TypeError) as exc:
            @register_vlm(r"foo.*")
            class NotAnAdapter:  # type: ignore[no-redef]
                pass
        assert "VLMBase" in str(exc.value)


class TestVLMBaseContract:
    def test_cannot_instantiate_abstract_base(self):
        with pytest.raises(TypeError):
            VLMBase()  # type: ignore[abstract]

    @pytest.mark.parametrize("cls", [QwenVLM, InternVLM])
    def test_adapter_provides_all_required_methods(self, cls):
        # Every abstract method in VLMBase must have a concrete
        # implementation in each adapter.
        abstract_methods = VLMBase.__abstractmethods__
        for method in abstract_methods:
            impl = getattr(cls, method, None)
            assert impl is not None, f"{cls.__name__} missing {method!r}"
            assert method not in getattr(cls, "__abstractmethods__", set()), (
                f"{cls.__name__} left {method!r} abstract"
            )


class TestWorkerPreprocessor:
    @pytest.mark.parametrize(
        "cls,model_name",
        [
            (QwenVLM, "Qwen/Qwen3-VL-4B-Instruct"),
            (InternVLM, "OpenGVLab/InternVL3_5-4B"),
        ],
    )
    def test_build_worker_preprocessor_returns_base(self, cls, model_name):
        prep = cls.build_worker_preprocessor(model_name)
        assert isinstance(prep, VLMWorkerPreprocessor)

    def test_preprocessor_is_picklable(self):
        # DataLoader with num_workers>0 forks + pickles the dataset
        # (which holds the preprocessor). Constructing must not load
        # anything heavy — only the model name and config should be in
        # the pickle.
        import pickle

        for cls, name in [
            (QwenVLM, "Qwen/Qwen3-VL-4B-Instruct"),
            (InternVLM, "OpenGVLab/InternVL3_5-4B"),
        ]:
            prep = cls.build_worker_preprocessor(name)
            restored = pickle.loads(pickle.dumps(prep))
            assert isinstance(restored, type(prep))
            assert restored.model_name == name

    def test_preprocessor_kwargs_flow_through(self):
        # Trainer passes min_pixels/max_pixels; adapters translate as
        # needed. Verify the raw values reach the preprocessor for
        # Qwen and the tile count for InternVL.
        qwen_prep = QwenVLM.build_worker_preprocessor(
            "Qwen/Qwen3-VL-4B-Instruct",
            min_pixels=100,
            max_pixels=999,
        )
        assert qwen_prep.min_pixels == 100
        assert qwen_prep.max_pixels == 999

        intern_prep = InternVLM.build_worker_preprocessor(
            "OpenGVLab/InternVL3_5-4B",
            max_pixels=448 * 448 * 6,  # -> 6 tiles
        )
        assert intern_prep.max_num_tiles == 6
