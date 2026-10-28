"""Parity fixtures for negpy_mobile: the desktop CPU engine's output, stage by stage.

Run from the repository root: `uv run python mobile/scripts/make_fixtures.py`. Writes
`mobile/negpy_mobile/tests/fixtures/<case>.json` with the flat config, the context and the
metrics, and one raw little-endian float32 file per array beside it. Synthetic negatives,
seeded, so a run is reproducible.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, is_dataclass, replace
from typing import Any

import numpy as np

from negpy.domain.interfaces import PipelineContext
from negpy.domain.models import WorkspaceConfig
from negpy.features.altprocess.models import AltProcess
from negpy.features.exposure.models import RenderIntent
from negpy.features.exposure.normalization import LogNegativeBounds
from negpy.features.geometry.processor import CropProcessor, GeometryProcessor
from negpy.features.local.models import LocalAdjustmentsConfig, LocalMask, MaskShape
from negpy.features.finish.processor import FinishProcessor, rebate_tone
from negpy.features.process.models import ProcessMode
from negpy.kernel.image.logic import working_oetf_encode
from negpy.kernel.system.config import DEFAULT_WORKSPACE_CONFIG
from negpy.services.export.print import PrintService
from negpy.services.rendering.engine import DarkroomEngine, print_stages

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "negpy_mobile", "tests", "fixtures")
H, W = 72, 108
ANALYSIS_ONLY = ("density_parade", "histogram_density", "scan_full_scale", "normalized_log")


def synthetic_scene(rng: np.random.Generator) -> np.ndarray:
    """A scene in linear light: a sky gradient, color patches, a face-like warm blob, noise."""
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    yy, xx = y / (H - 1), x / (W - 1)
    img = np.zeros((H, W, 3), np.float32)
    img[..., 0] = 0.35 + 0.25 * yy
    img[..., 1] = 0.45 + 0.20 * yy
    img[..., 2] = 0.75 - 0.35 * yy
    patches = [
        ((0.05, 0.25), (0.05, 0.30), (0.8, 0.1, 0.1)),
        ((0.05, 0.25), (0.35, 0.60), (0.1, 0.6, 0.15)),
        ((0.05, 0.25), (0.65, 0.95), (0.9, 0.85, 0.2)),
        ((0.55, 0.80), (0.05, 0.30), (0.02, 0.02, 0.03)),
        ((0.55, 0.80), (0.35, 0.60), (0.18, 0.18, 0.18)),
        ((0.55, 0.80), (0.65, 0.95), (0.95, 0.95, 0.95)),
    ]
    for (y0, y1), (x0, x1), color in patches:
        img[int(y0 * H) : int(y1 * H), int(x0 * W) : int(x1 * W)] = color
    blob = np.exp(-(((yy - 0.42) / 0.12) ** 2 + ((xx - 0.5) / 0.1) ** 2))
    img += blob[..., None] * np.array([0.5, 0.3, 0.2], np.float32)
    img *= 1.0 + 0.04 * rng.standard_normal(img.shape).astype(np.float32)
    return np.clip(img, 1e-4, 4.0)


def negative_of(scene: np.ndarray, mask: tuple[float, float, float], gamma: float = 0.6) -> np.ndarray:
    """Linear transmittance of a color negative: dye density grows with exposure under an
    orange mask (base transmittance per channel)."""
    density = gamma * np.log10(np.clip(scene, 1e-4, None) * 8.0 + 1.0)
    base = np.asarray(mask, np.float32)
    neg = base[None, None, :] * 10.0 ** (-density)
    return np.clip(neg, 1e-5, 1.0).astype(np.float32)


def jsonable(v: Any) -> Any:
    if isinstance(v, LogNegativeBounds):
        return {"floors": [float(f) for f in v.floors], "ceils": [float(c) for c in v.ceils]}
    if is_dataclass(v) and not isinstance(v, type):
        return jsonable(asdict(v))
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.bool_):
        return bool(v)
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    return str(v)


class Case:
    def __init__(self, name: str, config: WorkspaceConfig, source: np.ndarray):
        self.name = name
        self.config = config
        self.source = source


def run_case(case: Case) -> None:
    arrays: dict[str, np.ndarray] = {"source": case.source}
    engine = DarkroomEngine()
    settings = case.config
    context = PipelineContext(
        scale_factor=max(case.source.shape[:2]) / 1600.0,
        original_size=case.source.shape[:2],
        process_mode=settings.process.process_mode,
        wants_uv_grid=False,
    )
    final = engine.process(case.source, settings, source_hash=f"fixture:{case.name}", context=context)
    geometry_ctx = PipelineContext(scale_factor=context.scale_factor, original_size=case.source.shape[:2], process_mode=settings.process.process_mode)
    arrays["geometry"] = np.ascontiguousarray(GeometryProcessor(settings.geometry).process(case.source, geometry_ctx), np.float32)
    arrays["final"] = np.ascontiguousarray(final, np.float32)
    for stage in ("base", "exposure", "clahe", "lab"):
        entry = getattr(engine.cache, stage)
        if entry is not None:
            arrays[stage] = np.ascontiguousarray(entry.data, np.float32)
    flat = settings.exposure.render_intent == RenderIntent.FLAT
    if not flat:
        printed = print_stages(settings, arrays["lab"], context)
        arrays["print"] = np.ascontiguousarray(printed, np.float32)
        cropped = CropProcessor(settings.geometry).process(printed, context)
        arrays["crop"] = np.ascontiguousarray(cropped, np.float32)
        paper = PrintService.effective_paper_linear(settings.finish, settings.toning)
        tone = rebate_tone(settings, context.metrics) if settings.finish.carrier_width > 0.0 else None
        if tone is not None:
            arrays["rebate_tone"] = np.ascontiguousarray(tone, np.float32)
        finished = FinishProcessor(settings.finish, settings.export.export_print_size, paper, tone).process(cropped, context)
        arrays["finish"] = np.ascontiguousarray(finished, np.float32)
        arrays["encoded"] = np.ascontiguousarray(working_oetf_encode(finished), np.float32)
        assert np.array_equal(arrays["encoded"], arrays["final"]), f"{case.name}: recomputed tail differs from the engine"
    metrics: dict[str, Any] = {}
    for key, value in context.metrics.items():
        if isinstance(value, np.ndarray):
            # The Analysis panel's readouts are not pipeline inputs.
            if value.dtype.kind == "f" and key not in ANALYSIS_ONLY:
                arrays[f"metrics.{key}"] = np.ascontiguousarray(value, np.float32)
            continue
        if key in ("uv_grid",):
            continue
        metrics[key] = jsonable(value)
    metrics["active_roi"] = list(context.active_roi) if context.active_roi else None
    doc = {
        "config": jsonable(settings.to_dict()),
        "context": {
            "scale_factor": context.scale_factor,
            "original_size": list(context.original_size),
            "process_mode": str(settings.process.process_mode),
            "active_roi": list(context.active_roi) if context.active_roi else None,
        },
        "metrics": metrics,
        "arrays": {},
    }
    for key, arr in arrays.items():
        fname = f"{case.name}.{key}.f32"
        arr.astype("<f4").tofile(os.path.join(OUT_DIR, fname))
        doc["arrays"][key] = {"shape": list(arr.shape), "file": fname}
    with open(os.path.join(OUT_DIR, f"{case.name}.json"), "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
    print(f"{case.name}: {len(arrays)} arrays, roi={context.active_roi}")


def cases() -> list[Case]:
    rng = np.random.default_rng(7)
    scene = synthetic_scene(rng)
    c41 = negative_of(scene, (0.92, 0.55, 0.22))
    bw_scene = np.repeat(np.mean(scene, axis=2, keepdims=True), 3, axis=2)
    bw = negative_of(bw_scene, (0.85, 0.85, 0.85), gamma=0.7)
    base = WorkspaceConfig()
    manual = replace(
        base,
        exposure=replace(
            base.exposure,
            auto_exposure=False,
            auto_normalize_contrast=False,
            density=1.3,
            grade=95.0,
            toe=0.4,
            shoulder=0.3,
            wb_magenta=0.1,
            wb_yellow=-0.05,
            shadow_density=0.1,
            highlight_density=-0.05,
            midtone_gamma=0.1,
            paper_dmin=True,
            dye_separation=1.2,
        ),
        geometry=replace(base.geometry, fine_rotation=1.5, crop_rect=(0.08, 0.1, 0.9, 0.95)),
        lab=replace(base.lab, saturation=1.2, sharpen=0.5, clahe_strength=0.3),
        finish=replace(base.finish, vignette_stops=0.5),
        process=replace(base.process, hue_trim=5.0, white_point_offset=0.02),
        toning=replace(base.toning, shadow_tint_hue=200.0, shadow_tint_strength=0.3),
    )
    bw_cfg = replace(
        base,
        process=replace(base.process, process_mode=ProcessMode.BW, source_color=False),
        toning=replace(base.toning, sepia_strength=0.5, selenium_strength=0.3),
    )
    flat = replace(base, exposure=replace(base.exposure, render_intent=RenderIntent.FLAT))
    masks = replace(
        base,
        exposure=replace(base.exposure, contrast_mask_strength=0.5, diffusion=0.3),
        local=LocalAdjustmentsConfig(
            masks=(LocalMask(vertices=((0.5, 0.5), (0.8, 0.5), (0.8, 0.8)), stops=0.5, shape=MaskShape.OVAL, feather=0.05),)
        ),
    )
    lith = replace(bw_cfg, altproc=replace(base.altproc, alt_process=AltProcess.LITH))
    cyano = replace(bw_cfg, altproc=replace(base.altproc, alt_process=AltProcess.CYANOTYPE))
    sabattier = replace(bw_cfg, altproc=replace(base.altproc, alt_process=AltProcess.SABATTIER))
    carrier = replace(base, finish=replace(base.finish, carrier_width=2.0), geometry=replace(base.geometry, rotation=1, flip_horizontal=True))
    return [
        Case("c41_default", base, c41),
        Case("c41_manual", manual, c41),
        Case("c41_flat", flat, c41),
        Case("c41_masks", masks, c41),
        Case("c41_carrier", carrier, c41),
        Case("bw_default", bw_cfg, bw),
        Case("bw_lith", lith, bw),
        Case("bw_cyanotype", cyano, bw),
        Case("bw_sabattier", sabattier, bw),
    ]


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    for stale in os.listdir(OUT_DIR):
        os.remove(os.path.join(OUT_DIR, stale))
    with open(os.path.join(OUT_DIR, "defaults.json"), "w", encoding="utf-8") as f:
        json.dump({"workspace": jsonable(WorkspaceConfig().to_dict()), "app": jsonable(DEFAULT_WORKSPACE_CONFIG.to_dict())}, f, indent=1, sort_keys=True)
    for case in cases():
        run_case(case)
    return 0


if __name__ == "__main__":
    sys.exit(main())
