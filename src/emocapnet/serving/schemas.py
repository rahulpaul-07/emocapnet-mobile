"""Pydantic response schemas for the inference API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Health(BaseModel):
    status: str = "ok"


class ModelInfo(BaseModel):
    decoder: str
    encoder_type: str
    params_millions: float
    vocab_size: int
    emotions: list[str]
    device: str
    quantized: bool


class VAD(BaseModel):
    valence: float = Field(ge=0.0, le=1.0)
    arousal: float = Field(ge=0.0, le=1.0)
    dominance: float = Field(ge=0.0, le=1.0)


class CaptionResponse(BaseModel):
    caption: str
    emotion: str
    vad: VAD
    latency_ms: float


class MultiCaptionResponse(BaseModel):
    captions: dict[str, str]
    latency_ms: float
