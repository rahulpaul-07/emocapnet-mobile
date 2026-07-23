"""EmoCapNet-Mobile: emotion-controllable image captioning, distilled for on-device use."""

__version__ = "1.0.0"

from emocapnet.config import Config, load_config, seed_everything

__all__ = ["Config", "load_config", "seed_everything", "__version__"]
