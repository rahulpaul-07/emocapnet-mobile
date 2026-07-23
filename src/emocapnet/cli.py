"""Command-line interface.

Commands::

    emocapnet make-synthetic --out data/synthetic
    emocapnet train    --config configs/default.yaml [-o key=value ...]
    emocapnet evaluate --config configs/default.yaml --ckpt runs/default/checkpoints/best.pt
    emocapnet quantize --config configs/default.yaml --ckpt runs/default/checkpoints/best.pt
    emocapnet smoke    --out runs/smoke
    emocapnet serve    --config configs/default.yaml --ckpt runs/default/checkpoints/best.pt
    emocapnet export-onnx --config configs/default.yaml --ckpt runs/default/checkpoints/best.pt --out onnx/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

log = logging.getLogger("emocapnet")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _build_pipeline(cfg):
    """Shared setup: seed, tokenizer, teacher, student, datamodule, device."""
    import torch

    from emocapnet.config import seed_everything
    from emocapnet.data.datamodule import CaptionDataModule
    from emocapnet.models.student import EmoCapNetMobile
    from emocapnet.models.teacher import load_teacher
    from emocapnet.tokenization import build_tokenizer

    seed_everything(cfg.seed)
    cfg.ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = build_tokenizer(cfg.tokenizer_name)
    teacher = load_teacher(cfg, tok.vocab_size, device)
    dm = CaptionDataModule(cfg, tok, need_teacher=teacher is not None)
    student = EmoCapNetMobile(cfg, vocab_size=tok.vocab_size).to(device)
    n_params = student.num_parameters()
    log.info("Student: %s | %.1fM params | device %s", student.dec_name, n_params / 1e6, device)
    if teacher is not None:
        tp = sum(p.numel() for p in teacher.parameters())
        log.info("Teacher: %.1fM params -> student is %.1fx smaller", tp / 1e6, tp / n_params)
    return device, tok, teacher, dm, student


# ---------------------------------------------------------------------- commands
def cmd_make_synthetic(args) -> int:
    from emocapnet.data.synthetic import generate_synthetic_dataset

    csv_path, img_dir = generate_synthetic_dataset(
        args.out, n_images=args.n_images, captions_per_image=args.captions_per_image, seed=args.seed
    )
    print(f"captions: {csv_path}\nimages:   {img_dir}")
    return 0


def cmd_train(args) -> int:
    from emocapnet.config import load_config
    from emocapnet.evaluation.metrics import ensure_nltk_data
    from emocapnet.training.trainer import Trainer

    cfg = load_config(args.config, args.override)
    ensure_nltk_data()
    device, tok, teacher, dm, student = _build_pipeline(cfg)
    cfg.save(Path(cfg.work_dir) / "config_resolved.yaml")

    trainer = Trainer(cfg, student, teacher, dm.train_loader(), dm.val_loader(), device)
    trainer.pretrain_decoder(dm.pretrain_loader())
    trainer.fit()
    print(f"Best val loss: {trainer.best_val:.4f}  (checkpoints in {cfg.ckpt_dir})")
    return 0


def cmd_evaluate(args) -> int:
    from emocapnet.config import load_config
    from emocapnet.evaluation.controllability import controllability_report
    from emocapnet.evaluation.evaluator import evaluate_captions, format_metrics, vad_mse
    from emocapnet.evaluation.metrics import ensure_nltk_data
    from emocapnet.training.checkpoint import load_student_weights

    cfg = load_config(args.config, args.override)
    ensure_nltk_data()
    device, tok, teacher, dm, student = _build_pipeline(cfg)
    ckpt = args.ckpt or str(Path(cfg.ckpt_dir) / "best.pt")
    if Path(ckpt).exists():
        load_student_weights(student, ckpt, device)
    else:
        log.warning("No checkpoint at %s -> evaluating untrained weights", ckpt)

    test_loader = dm.test_loader()
    metrics = evaluate_captions(student, test_loader, cfg, tok, device, tag="student")
    metrics["vad_mse"] = vad_mse(student, test_loader, device)
    print(format_metrics(metrics, "STUDENT (mobile)"))

    if args.controllability:
        report = controllability_report(
            student, dm.test_df, dm.image_index, dm.processors.student, cfg, tok, n_images=args.n_images
        )
        for dim, r in report.items():
            print(f"  controllability {dim:10}: requested-vs-text Pearson r = {r:+.3f}")

    if args.eval_teacher and teacher is not None:
        t_metrics = evaluate_captions(
            teacher, test_loader, cfg, tok, device, use_student_pixels=False, tag="teacher"
        )
        t_metrics["vad_mse"] = vad_mse(teacher, test_loader, device, use_student_pixels=False)
        print(format_metrics(t_metrics, "TEACHER (v3)"))

    out = {k: v for k, v in metrics.items() if k not in ("hyps", "refs")}
    (Path(cfg.work_dir) / "eval_metrics.json").write_text(json.dumps(out, indent=2, default=float))
    return 0


def cmd_quantize(args) -> int:
    import torch

    from emocapnet.compression.quantize import compression_summary, quantize_int8, to_fp16
    from emocapnet.config import load_config
    from emocapnet.training.checkpoint import load_student_weights

    cfg = load_config(args.config, args.override)
    device, tok, teacher, dm, student = _build_pipeline(cfg)
    ckpt = args.ckpt or str(Path(cfg.ckpt_dir) / "best.pt")
    if Path(ckpt).exists():
        load_student_weights(student, ckpt, "cpu")

    student = student.to("cpu").eval()
    summary = compression_summary(student, teacher)
    torch.save(to_fp16(student).state_dict(), Path(cfg.work_dir) / "student_fp16.pt")
    try:
        torch.save(quantize_int8(student).state_dict(), Path(cfg.work_dir) / "student_int8.pt")
    except Exception as e:
        log.warning("INT8 export failed: %r", e)
    (Path(cfg.work_dir) / "compression_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


def cmd_smoke(args) -> int:
    """Hermetic end-to-end run on synthetic data with the tiny encoder + ultralight
    decoder: train -> evaluate -> quantize -> latency. Proves the pipeline works."""
    import torch

    from emocapnet.compression.quantize import compression_summary
    from emocapnet.config import Config
    from emocapnet.data.synthetic import generate_synthetic_dataset
    from emocapnet.evaluation.evaluator import evaluate_captions, format_metrics, vad_mse
    from emocapnet.evaluation.latency import cpu_latency_ms
    from emocapnet.evaluation.metrics import ensure_nltk_data
    from emocapnet.training.trainer import Trainer

    out = Path(args.out)
    csv_path, img_dir = generate_synthetic_dataset(out / "data", n_images=args.n_images, seed=0)
    cfg = Config(
        captions_csv=csv_path,
        image_dirs=[img_dir],
        work_dir=str(out / "run"),
        student_encoder_type="tiny",
        ultralight=True,
        tokenizer_name=args.tokenizer,
        epochs=args.epochs,
        pretrain_epochs=0,
        batch_size=8,
        grad_accum=1,
        num_workers=0,
        max_length=24,
        prefix_len=4,
        hard_max_length=10,
        min_length=2,
        distill_alpha=0.0,
        unfreeze_vit_epoch=1,
    )
    ensure_nltk_data()
    device, tok, teacher, dm, student = _build_pipeline(cfg)

    trainer = Trainer(cfg, student, teacher, dm.train_loader(), dm.val_loader(), device)
    trainer.fit()

    test_loader = dm.test_loader()
    metrics = evaluate_captions(student, test_loader, cfg, tok, device, tag="student")
    metrics["vad_mse"] = vad_mse(student, test_loader, device)
    print(format_metrics(metrics, "SMOKE STUDENT"))

    summary = compression_summary(student)
    batch = next(iter(test_loader))
    ms = cpu_latency_ms(
        student,
        batch["pv_student"][:1],
        batch["vad"][:1],
        tok.emo_token_ids["happiness"],
        cfg,
        tok,
        repeats=3,
    )
    summary["cpu_latency_ms"] = round(ms, 1)
    print(json.dumps(summary, indent=2))
    (Path(cfg.work_dir) / "smoke_summary.json").write_text(
        json.dumps(
            {
                "metrics": {k: v for k, v in metrics.items() if k not in ("hyps", "refs")},
                "compression": summary,
            },
            indent=2,
            default=float,
        )
    )
    assert torch.isfinite(torch.tensor(trainer.best_val)), "training diverged"
    print("SMOKE OK")
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    from emocapnet.serving.app import create_app

    app = create_app(config_path=args.config, ckpt_path=args.ckpt, quantized=args.int8)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def cmd_export_onnx(args) -> int:
    from emocapnet.config import load_config
    from emocapnet.export.onnx_export import OnnxCaptioner, export_onnx, verify_parity
    from emocapnet.training.checkpoint import load_student_weights

    cfg = load_config(args.config, args.override)
    device, tok, teacher, dm, student = _build_pipeline(cfg)
    ckpt = args.ckpt or str(Path(cfg.ckpt_dir) / "best.pt")
    if Path(ckpt).exists():
        load_student_weights(student, ckpt, "cpu")
    else:
        log.warning("No checkpoint at %s -> exporting untrained weights", ckpt)

    image_size = dm.processors.student_size
    meta = export_onnx(student, cfg, args.out, image_size)
    if not args.skip_verify:
        captioner = OnnxCaptioner(args.out, cfg)
        verify_parity(student, captioner, cfg, image_size, tok.emo_token_ids["happiness"], tok.eos_id)
        print("Parity verified: onnxruntime output is token-identical to torch.")
    print(json.dumps(meta, indent=2))
    return 0


# ---------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="emocapnet", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("make-synthetic", help="generate a tiny self-contained dataset")
    sp.add_argument("--out", default="data/synthetic")
    sp.add_argument("--n-images", type=int, default=40)
    sp.add_argument("--captions-per-image", type=int, default=2)
    sp.add_argument("--seed", type=int, default=0)
    sp.set_defaults(func=cmd_make_synthetic)

    for name, func in (("train", cmd_train), ("evaluate", cmd_evaluate), ("quantize", cmd_quantize)):
        sp = sub.add_parser(name)
        sp.add_argument("--config", required=True)
        sp.add_argument("-o", "--override", action="append", default=[], metavar="KEY=VALUE")
        if name in ("evaluate", "quantize"):
            sp.add_argument("--ckpt", default=None)
        if name == "evaluate":
            sp.add_argument("--eval-teacher", action="store_true")
            sp.add_argument("--controllability", action="store_true")
            sp.add_argument("--n-images", type=int, default=30)
        sp.set_defaults(func=func)

    sp = sub.add_parser("serve", help="run the FastAPI inference service")
    sp.add_argument("--config", required=True)
    sp.add_argument("--ckpt", default=None)
    sp.add_argument("--int8", action="store_true", help="serve the INT8 dynamic-quantized build")
    sp.add_argument("--host", default="0.0.0.0")
    sp.add_argument("--port", type=int, default=8000)
    sp.set_defaults(func=cmd_serve)

    sp = sub.add_parser("export-onnx", help="export encoder/decoder ONNX graphs + parity check")
    sp.add_argument("--config", required=True)
    sp.add_argument("-o", "--override", action="append", default=[], metavar="KEY=VALUE")
    sp.add_argument("--ckpt", default=None)
    sp.add_argument("--out", default="onnx")
    sp.add_argument("--skip-verify", action="store_true")
    sp.set_defaults(func=cmd_export_onnx)

    sp = sub.add_parser("smoke", help="hermetic end-to-end pipeline check on synthetic data")
    sp.add_argument("--out", default="runs/smoke")
    sp.add_argument("--epochs", type=int, default=1)
    sp.add_argument("--n-images", type=int, default=24)
    sp.add_argument("--tokenizer", default="gpt2")
    sp.set_defaults(func=cmd_smoke)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
