"""API contract tests for the FastAPI service (tiny model, untrained weights)."""

from __future__ import annotations

import io

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from emocapnet.constants import EMOTION_LABELS  # noqa: E402


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from PIL import Image

    from emocapnet.config import Config
    from emocapnet.data.synthetic import generate_synthetic_dataset
    from emocapnet.serving.app import CaptionService, create_app
    from emocapnet.tokenization import build_tokenizer, write_tiny_gpt2_tokenizer

    tmp = tmp_path_factory.mktemp("serve")
    csv_path, img_dir = generate_synthetic_dataset(tmp / "data", n_images=4, seed=0)
    try:
        build_tokenizer("gpt2")
        tok_name = "gpt2"
    except Exception:
        tok_name = write_tiny_gpt2_tokenizer(tmp / "tok")
    cfg = Config(
        captions_csv=csv_path,
        image_dirs=[img_dir],
        work_dir=str(tmp / "run"),
        student_encoder_type="tiny",
        ultralight=True,
        tokenizer_name=tok_name,
        max_length=24,
        prefix_len=4,
        hard_max_length=6,
        min_length=2,
        num_workers=0,
    )
    service = CaptionService(cfg, ckpt_path=None)
    app = create_app(service=service)

    # in-memory test image
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (120, 40, 200)).save(buf, format="JPEG")
    return TestClient(app), buf.getvalue()


def test_health(client):
    c, _ = client
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_info_lists_all_emotions(client):
    c, _ = client
    r = c.get("/info")
    assert r.status_code == 200
    body = r.json()
    assert body["emotions"] == EMOTION_LABELS
    assert body["params_millions"] > 0
    assert body["quantized"] is False


def test_caption_default_emotion(client):
    c, img = client
    r = c.post("/caption", files={"image": ("x.jpg", img, "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["emotion"] == "factual"
    assert isinstance(body["caption"], str)
    assert body["latency_ms"] > 0
    assert body["vad"] == {"valence": 0.5, "arousal": 0.5, "dominance": 0.5}


def test_caption_explicit_vad(client):
    c, img = client
    r = c.post(
        "/caption",
        files={"image": ("x.jpg", img, "image/jpeg")},
        data={"emotion": "happiness", "valence": "0.9", "arousal": "0.6", "dominance": "0.7"},
    )
    assert r.status_code == 200
    assert r.json()["vad"]["valence"] == 0.9


def test_caption_rejects_bad_emotion(client):
    c, img = client
    r = c.post("/caption", files={"image": ("x.jpg", img, "image/jpeg")}, data={"emotion": "joyful"})
    assert r.status_code == 422


def test_caption_rejects_partial_vad(client):
    c, img = client
    r = c.post("/caption", files={"image": ("x.jpg", img, "image/jpeg")}, data={"valence": "0.5"})
    assert r.status_code == 422


def test_caption_rejects_out_of_range_vad(client):
    c, img = client
    r = c.post(
        "/caption",
        files={"image": ("x.jpg", img, "image/jpeg")},
        data={"valence": "1.5", "arousal": "0.5", "dominance": "0.5"},
    )
    assert r.status_code == 422


def test_caption_rejects_non_image(client):
    c, _ = client
    r = c.post("/caption", files={"image": ("x.jpg", b"not an image", "image/jpeg")})
    assert r.status_code == 400


def test_all_emotions_endpoint(client):
    c, img = client
    r = c.post("/caption/emotions", files={"image": ("x.jpg", img, "image/jpeg")})
    assert r.status_code == 200
    assert set(r.json()["captions"]) == set(EMOTION_LABELS)
