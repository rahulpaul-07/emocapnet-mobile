"""Distillation trainer: Stage-0 language pre-finetuning, then KD training with
staged encoder unfreeze, AMP, gradient accumulation, and best-checkpoint tracking.
"""

from __future__ import annotations

import logging
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from emocapnet.config import Config
from emocapnet.losses import combined_loss
from emocapnet.training.checkpoint import save_checkpoint
from emocapnet.training.schedule import freeze_encoder, unfreeze_encoder, unwrap, warmup_cosine
from emocapnet.utils import make_grad_scaler

log = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        cfg: Config,
        student: nn.Module,
        teacher: nn.Module | None,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.student = student
        self.teacher = teacher
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.use_amp = device.type == "cuda"
        self.history: list[dict] = []
        self.best_val = float("inf")

    # ------------------------------------------------------------------ internals
    def _autocast(self):
        return torch.autocast(device_type=self.device.type, enabled=self.use_amp)

    def _make_scaler(self):
        return make_grad_scaler(self.device.type, self.use_amp)

    def run_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler,
        scaler,
        train: bool,
        use_teacher: bool,
    ) -> dict[str, float]:
        cfg = self.cfg
        self.student.train(train)
        agg = dict.fromkeys(("tot", "lm", "kd", "vad", "emo", "con", "ft"), 0.0)
        n = 0
        optimizer.zero_grad()
        for step, batch in enumerate(tqdm(loader, leave=False, desc="train" if train else "val")):
            pv_s = batch["pv_student"].to(self.device)
            ids = batch["input_ids"].to(self.device)
            mask = batch["attention_mask"].to(self.device)
            bt = {
                "labels": batch["labels"].to(self.device),
                "vad": batch["vad"].to(self.device),
                "emotion_idx": batch["emotion_idx"].to(self.device),
            }
            t_out = None
            if use_teacher and self.teacher is not None and cfg.distill_alpha > 0:
                pv_t = batch["pv_teacher"].to(self.device)
                with torch.no_grad(), self._autocast():
                    t_out = self.teacher(pv_t, ids, mask, bt["vad"])
            with torch.set_grad_enabled(train), self._autocast():
                s_out = self.student(pv_s, ids, mask, bt["vad"])
                loss, parts = combined_loss(s_out, bt, t_out, cfg, unwrap(self.student).logit_scale)
                loss = loss / cfg.grad_accum
            if train:
                scaler.scale(loss).backward()
                if (step + 1) % cfg.grad_accum == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.student.parameters(), cfg.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
            b = pv_s.size(0)
            n += b
            agg["tot"] += float(loss) * cfg.grad_accum * b
            for k in ("lm", "kd", "vad", "emo", "con", "ft"):
                agg[k] += parts[k] * b
        return {k: v / max(n, 1) for k, v in agg.items()}

    # ------------------------------------------------------------------ stage 0
    def pretrain_decoder(self, pre_loader: DataLoader | None) -> None:
        """Stage 0: supervised LM pre-finetuning on factual captions (no teacher)."""
        cfg = self.cfg
        if cfg.pretrain_epochs <= 0 or pre_loader is None:
            log.info("Stage 0 skipped")
            return
        log.info("=== Stage 0: LM pre-finetuning (%d epochs) ===", cfg.pretrain_epochs)
        freeze_encoder(self.student)
        opt = torch.optim.AdamW(
            (p for p in self.student.parameters() if p.requires_grad),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        sched = warmup_cosine(opt, cfg.pretrain_epochs, 1, len(pre_loader))
        scaler = self._make_scaler()
        for ep in range(cfg.pretrain_epochs):
            stats = self.run_epoch(pre_loader, opt, sched, scaler, train=True, use_teacher=False)
            log.info(
                "pretrain %d/%d  lm %.3f  tot %.3f", ep + 1, cfg.pretrain_epochs, stats["lm"], stats["tot"]
            )
        torch.save(
            {"model_state": unwrap(self.student).state_dict()},
            f"{cfg.ckpt_dir}/student_pretrained.pt",
        )

    # ------------------------------------------------------------------ main loop
    def fit(self) -> list[dict]:
        cfg = self.cfg
        freeze_encoder(self.student)  # warm up mapping/decoder first
        optimizer = torch.optim.AdamW(
            (p for p in self.student.parameters() if p.requires_grad),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        scheduler = warmup_cosine(optimizer, cfg.epochs, cfg.warmup_epochs, len(self.train_loader))
        scaler = self._make_scaler()
        start = time.time()

        for epoch in range(cfg.epochs):
            if epoch == cfg.unfreeze_vit_epoch:
                unfreeze_encoder(self.student)
                inner = unwrap(self.student)
                dec_params = [
                    p
                    for nm, p in inner.named_parameters()
                    if p.requires_grad and not nm.startswith("encoder.")
                ]
                enc_params = [p for p in inner.encoder.parameters() if p.requires_grad]
                optimizer = torch.optim.AdamW(
                    [
                        {"params": dec_params, "lr": cfg.learning_rate},
                        {"params": enc_params, "lr": cfg.learning_rate / 5},
                    ],
                    weight_decay=cfg.weight_decay,
                )
                scheduler = warmup_cosine(optimizer, cfg.epochs - epoch, 0, len(self.train_loader))
                log.info("Epoch %d: encoder unfrozen (encoder LR = base/5)", epoch + 1)

            tr = self.run_epoch(self.train_loader, optimizer, scheduler, scaler, train=True, use_teacher=True)
            with torch.no_grad():
                # Teacher off in val: student-only val loss is still a consistent
                # best-checkpoint signal across epochs, at half the cost.
                vl = self.run_epoch(
                    self.val_loader, optimizer, scheduler, scaler, train=False, use_teacher=False
                )

            elapsed = (time.time() - start) / 60
            log.info(
                "Ep %02d  tr %.4f (lm %.3f kd %.3f vad %.4f emo %.3f con %.3f ft %.3f)  vl %.4f  [%.0f min]",
                epoch + 1,
                tr["tot"],
                tr["lm"],
                tr["kd"],
                tr["vad"],
                tr["emo"],
                tr["con"],
                tr["ft"],
                vl["tot"],
                elapsed,
            )
            self.history.append(
                {
                    "epoch": epoch + 1,
                    **{f"tr_{k}": v for k, v in tr.items()},
                    **{f"vl_{k}": v for k, v in vl.items()},
                    "min": round(elapsed, 1),
                }
            )
            is_best = vl["tot"] < self.best_val
            if is_best:
                self.best_val = vl["tot"]
            save_checkpoint(self.student, epoch + 1, vl["tot"], self.history, cfg, is_best=is_best)

        log.info("Done in %.1f min | best val %.4f", (time.time() - start) / 60, self.best_val)
        return self.history
