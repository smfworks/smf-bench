"""
Model Registry — capability manifests and capability-gating logic.

Each model gets a YAML manifest declaring its input/output modalities and capabilities.
The runner uses this to gate tests: applicable tests execute, inapplicable tests
are flagged N/A (not a zero, not a skip — an explicit "this model can't do this").
"""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class Modality(str, Enum):
    """I/O modalities a model can accept or produce."""
    TEXT = "text"
    IMAGE_IN = "image_in"      # accepts image input
    VIDEO_IN = "video_in"      # accepts video input
    AUDIO_IN = "audio_in"     # accepts audio input
    IMAGE_OUT = "image_out"   # produces image output (generation)
    AUDIO_OUT = "audio_out"   # produces audio output (generation)
    VIDEO_OUT = "video_out"   # produces video output (generation)
    EMBED = "embed"            # produces embeddings


class Capability(str, Enum):
    """Higher-level capabilities a model may have, beyond raw I/O modalities."""
    VISION = "vision"                # image understanding
    VIDEO = "video"                  # video understanding
    AUDIO = "audio"                  # audio understanding
    OCR = "ocr"                      # text extraction from images
    CHART = "chart"                 # chart/diagram interpretation
    REASONING = "reasoning"         # logical/mathematical reasoning
    CODING = "coding"               # code generation
    WRITING = "writing"             # prose/creative writing
    TOOL_CALLING = "tool_calling"   # function/tool calling
    AGENTIC = "agentic"            # multi-step agentic tasks
    IMAGE_GEN = "image_gen"         # image generation
    AUDIO_GEN = "audio_gen"         # audio/music generation
    VIDEO_GEN = "video_gen"         # video generation
    STT = "stt"                     # speech-to-text
    CONTEXT_LONG = "context_long"   # long-context handling (>32K)
    STREAMING = "streaming"        # supports streaming responses


# Maps capabilities to the modalities they require
CAPABILITY_REQUIREMENTS: dict[Capability, set[Modality]] = {
    Capability.VISION: {Modality.IMAGE_IN},
    Capability.VIDEO: {Modality.VIDEO_IN},
    Capability.AUDIO: {Modality.AUDIO_IN},
    Capability.OCR: {Modality.IMAGE_IN},
    Capability.CHART: {Modality.IMAGE_IN},
    Capability.REASONING: {Modality.TEXT},
    Capability.CODING: {Modality.TEXT},
    Capability.WRITING: {Modality.TEXT},
    Capability.TOOL_CALLING: {Modality.TEXT},
    Capability.AGENTIC: {Modality.TEXT},
    Capability.IMAGE_GEN: {Modality.IMAGE_OUT},
    Capability.AUDIO_GEN: {Modality.AUDIO_OUT},
    Capability.VIDEO_GEN: {Modality.VIDEO_OUT},
    Capability.STT: {Modality.AUDIO_IN, Modality.TEXT},
    Capability.CONTEXT_LONG: {Modality.TEXT},
    Capability.STREAMING: {Modality.TEXT},
}


@dataclass
class ModelManifest:
    """Capability manifest for a model under test."""
    model_id: str
    name: str
    provider: str
    input_modalities: set[Modality]
    output_modalities: set[Modality]
    capabilities: set[Capability]
    context_length: int = 32768
    max_output_tokens: int = 4096
    served_name: str = ""   # the model name the serving endpoint expects (e.g. "nvidia/Qwen3.6-35B-A3B-NVFP4")
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def api_model_name(self) -> str:
        """The model name to pass to the API. Falls back to model_id if served_name unset."""
        return self.served_name or self.model_id

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelManifest":
        with open(path) as f:
            data = yaml.safe_load(f)

        input_mods = {Modality(m) for m in data.get("input_modalities", [])}
        output_mods = {Modality(m) for m in data.get("output_modalities", [])}
        caps = {Capability(c) for c in data.get("capabilities", [])}

        return cls(
            model_id=data["model_id"],
            name=data["name"],
            provider=data.get("provider", "unknown"),
            input_modalities=input_mods,
            output_modalities=output_mods,
            capabilities=caps,
            context_length=data.get("context_length", 32768),
            max_output_tokens=data.get("max_output_tokens", 4096),
            served_name=data.get("served_name", ""),
            metadata=data.get("metadata", {}),
        )

    def can_run(self, required_modalities: set[Modality], required_capabilities: set[Capability]) -> bool:
        """Check if this model can run a test requiring the given modalities/capabilities."""
        # All required input modalities must be in the model's input set
        for mod in required_modalities:
            if mod not in self.input_modalities and mod not in self.output_modalities:
                return False
        # All required capabilities must be declared
        for cap in required_capabilities:
            if cap not in self.capabilities:
                return False
        return True


class ModelRegistry:
    """Registry of model manifests, keyed by model_id."""
    def __init__(self) -> None:
        self._models: dict[str, ModelManifest] = {}

    def register(self, manifest: ModelManifest) -> None:
        self._models[manifest.model_id] = manifest

    def load_dir(self, dir_path: str | Path) -> int:
        """Load all YAML manifests from a directory."""
        count = 0
        for p in sorted(Path(dir_path).glob("*.yaml")):
            manifest = ModelManifest.from_yaml(p)
            self.register(manifest)
            count += 1
        return count

    def get(self, model_id: str) -> ModelManifest | None:
        return self._models.get(model_id)

    def list_models(self) -> list[str]:
        return sorted(self._models.keys())

    def applicable_tests(self, test_registry, model_id: str) -> tuple[list, list]:
        """Partition tests into (applicable, not_applicable) for a given model.

        Not-applicable tests are flagged N/A — not skipped, not failed.
        """
        model = self._models.get(model_id)
        if not model:
            raise KeyError(f"Model '{model_id}' not registered")

        applicable = []
        not_applicable = []

        for test in test_registry.all_tests():
            if model.can_run(test.required_modalities, test.required_capabilities):
                applicable.append(test)
            else:
                not_applicable.append(test)

        return applicable, not_applicable