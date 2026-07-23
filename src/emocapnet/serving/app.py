"""FastAPI inference service for the mobile student model.

Endpoints:
    GET  /health            liveness probe
    GET  /info              model metadata
    POST /caption           image (+ emotion, optional VAD) -> caption
    POST /caption/emotions  image -> one caption per emotion

Run::

    emocapnet serve --config configs/default.yaml --ckpt runs/default/checkpoints/best.pt

Note: no ``from __future__ import annotations`` here — FastAPI/pydantic v2 need
concrete (non-deferred) annotations on endpoint signatures.
"""

import io
import logging
import time
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image

from emocapnet.config import Config, load_config
from emocapnet.constants import EMOTION_LABELS, EMOTION_VAD
from emocapnet.serving.schemas import VAD, CaptionResponse, Health, ModelInfo, MultiCaptionResponse
from emocapnet.tokenization import TokenizerBundle, build_tokenizer, strip_caption

log = logging.getLogger(__name__)


class CaptionService:
    """Owns the loaded model + tokenizer + preprocessing; does the actual inference."""

    def __init__(self, cfg: Config, ckpt_path: str | None, quantized: bool = False) -> None:
        from emocapnet.data.dataset import build_processors
        from emocapnet.models.student import EmoCapNetMobile
        from emocapnet.training.checkpoint import load_student_weights

        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() and not quantized else "cpu")
        self.tok: TokenizerBundle = build_tokenizer(cfg.tokenizer_name)
        self.transform = build_processors(cfg, need_teacher=False).student
        self.model = EmoCapNetMobile(cfg, vocab_size=self.tok.vocab_size)
        if ckpt_path and Path(ckpt_path).exists():
            load_student_weights(self.model, ckpt_path, "cpu")
        else:
            log.warning("No checkpoint provided/found (%s) — serving untrained weights", ckpt_path)
        # capture before quantization: quantized Linear modules hide their weights
        # from .parameters(), which would misreport the model size in /info
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.quantized = quantized
        if quantized:
            from emocapnet.compression.quantize import quantize_int8

            self.model = quantize_int8(self.model)
            log.info("Serving INT8 dynamic-quantized model")
        self.model = self.model.to(self.device).eval()

    # ------------------------------------------------------------------ inference
    def _prepare(self, image_bytes: bytes) -> torch.Tensor:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return self.transform(img).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def caption(self, image_bytes: bytes, emotion: str, vad) -> "tuple[str, VAD]":
        if emotion not in EMOTION_LABELS:
            raise ValueError(f"Unknown emotion {emotion!r}; choose from {EMOTION_LABELS}")
        v = vad if vad is not None else EMOTION_VAD[emotion]
        pv = self._prepare(image_bytes)
        vad_t = torch.tensor(v, dtype=torch.float32).view(1, 3).to(self.device)
        emo_ids = torch.tensor([self.tok.emo_token_ids[emotion]], device=self.device)
        gen = self.model.generate(pv, vad_t, emo_ids, self.tok.eos_id, self.cfg)
        text = strip_caption(self.tok.tokenizer.decode(gen[0], skip_special_tokens=False))
        return text, VAD(valence=v[0], arousal=v[1], dominance=v[2])

    def info(self) -> ModelInfo:
        return ModelInfo(
            decoder=getattr(self.model, "dec_name", "unknown"),
            encoder_type=self.cfg.student_encoder_type,
            params_millions=round(self.n_params / 1e6, 1),
            vocab_size=self.tok.vocab_size,
            emotions=EMOTION_LABELS,
            device=str(self.device),
            quantized=self.quantized,
        )


def create_app(
    config_path: str | None = None,
    ckpt_path: str | None = None,
    quantized: bool = False,
    service: CaptionService | None = None,
) -> FastAPI:
    """App factory. Pass a prebuilt ``service`` in tests; otherwise built from config."""
    if service is None:
        cfg = load_config(config_path) if config_path else Config()
        service = CaptionService(cfg, ckpt_path, quantized=quantized)

    app = FastAPI(
        title="EmoCapNet-Mobile",
        description="Emotion-controllable image captioning (distilled + quantized).",
        version="1.0.0",
    )
    app.state.service = service

    @app.get("/health", response_model=Health)
    def health() -> Health:
        return Health()

    @app.get("/info", response_model=ModelInfo)
    def info() -> ModelInfo:
        return service.info()

    @app.post("/caption", response_model=CaptionResponse)
    async def caption(
        image: UploadFile = File(...),
        emotion: str = Form("factual"),
        valence: float | None = Form(None),
        arousal: float | None = Form(None),
        dominance: float | None = Form(None),
    ) -> CaptionResponse:
        vad = None
        if any(x is not None for x in (valence, arousal, dominance)):
            if any(x is None for x in (valence, arousal, dominance)):
                raise HTTPException(422, "Provide all of valence/arousal/dominance, or none")
            if not all(0.0 <= x <= 1.0 for x in (valence, arousal, dominance)):
                raise HTTPException(422, "VAD values must be in [0, 1]")
            vad = (valence, arousal, dominance)
        data = await image.read()
        t0 = time.time()
        try:
            text, vad_out = service.caption(data, emotion, vad)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        except Exception as e:  # unreadable image etc.
            raise HTTPException(400, f"Could not process image: {e!r}") from e
        return CaptionResponse(
            caption=text, emotion=emotion, vad=vad_out, latency_ms=round((time.time() - t0) * 1000, 1)
        )

    @app.post("/caption/emotions", response_model=MultiCaptionResponse)
    async def caption_all(image: UploadFile = File(...)) -> MultiCaptionResponse:
        data = await image.read()
        t0 = time.time()
        try:
            captions = {e: service.caption(data, e, None)[0] for e in EMOTION_LABELS}
        except Exception as e:
            raise HTTPException(400, f"Could not process image: {e!r}") from e
        return MultiCaptionResponse(captions=captions, latency_ms=round((time.time() - t0) * 1000, 1))

    return app
