import pytest
import yaml

from emocapnet.config import Config, load_config


def test_default_config_roundtrip(tmp_path):
    cfg = Config()
    path = tmp_path / "cfg.yaml"
    cfg.save(path)
    loaded = load_config(path)
    assert loaded.to_dict() == cfg.to_dict()


def test_overrides_are_typed():
    cfg = load_config(None, overrides=["epochs=3", "distill_alpha=0.25", "ultralight=true", "image_dirs=a,b"])
    assert cfg.epochs == 3 and isinstance(cfg.epochs, int)
    assert cfg.distill_alpha == 0.25
    assert cfg.ultralight is True
    assert cfg.image_dirs == ["a", "b"]


def test_unknown_key_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"not_a_key": 1}))
    with pytest.raises(ValueError, match="Unknown config keys"):
        load_config(path)
    with pytest.raises(ValueError, match="Unknown config key"):
        load_config(None, overrides=["nope=1"])


def test_ckpt_dir_follows_work_dir():
    cfg = load_config(None, overrides=["work_dir=/tmp/xyz"])
    assert cfg.ckpt_dir.replace("\\", "/") == "/tmp/xyz/checkpoints"
