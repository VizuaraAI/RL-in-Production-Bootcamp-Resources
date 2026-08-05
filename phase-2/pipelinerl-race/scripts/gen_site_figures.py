"""PaperBanana concept figures for the project site + paper.

These are CONCEPT figures — they explain the architecture. Every figure containing a NUMBER
is a matplotlib plot generated from real telemetry (scripts/four_graphs.py); nothing here is
allowed to depict a measurement, because an image model cannot be trusted with data.

    python gen_site_figures.py            # all
    python gen_site_figures.py hero split # only matching
"""
import argparse
import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

ENV_PATH = "/Users/raj/Desktop/Course_Creator/.env.local"
load_dotenv(ENV_PATH)
# Supply the key via the environment or the shared .env.local — never inline it here.
# This file is published; a hardcoded key would leak on the first commit.
#     export GOOGLE_API_KEY=...
if not os.environ.get("GOOGLE_API_KEY"):
    with open(ENV_PATH) as f:
        for line in f:
            if line.startswith("GOOGLE_API_KEY="):
                os.environ["GOOGLE_API_KEY"] = line.strip().split("=", 1)[1]
                break

from paperbanana import DiagramType, GenerationInput, PaperBananaPipeline  # noqa: E402
from paperbanana.core.config import Settings  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "website" / "assets" / "pb"
OUT.mkdir(parents=True, exist_ok=True)
WORK = Path("/tmp/pb_race_site")
WORK.mkdir(parents=True, exist_ok=True)

# Same restrained palette as the lecture deck, so site / slides / paper read as one artifact.
STYLE = (
    "Clean, minimal, modern scientific figure on a WARM CREAM PAPER background, the exact "
    "colour #F7F2E8 — NOT white. Editorial 'Distill.pub' and 3Blue1Brown aesthetic: warm, "
    "analog, beautifully typeset textbook feel. Typography: elegant serif for the title, "
    "clean sans-serif for labels, monospace for any short code-like tag. Use ONLY this "
    "restrained palette: dark warm ink #1F1B16 for text and thin outlines; deep brick red "
    "#8B3A3A for TRAINING; ochre gold #8E6B2F for GENERATION; slate blue #4A5D7E for "
    "systems, GPUs, weights and interconnect; forest green #3D5A4A for good/speedup; muted "
    "plum #6B4E7E for queues and controllers; pale warm grey #C9BFAC for IDLE or wasted "
    "time. Soft rounded rectangles, thin 1px hairline rules, generous whitespace, very "
    "subtle soft shadows. Large readable short labels only. No clutter, no rainbow colours, "
    "no neon, no gradients. A single hand-tuned figure from a premium systems paper. "
    "CRITICAL: do NOT render any measurement numbers, percentages, axis values or result "
    "figures — this is a conceptual diagram only."
)
M = DiagramType.METHODOLOGY

FIGURES = [
    {
        "filename": "hero_three_arms.png",
        "context": (
            "A wide hero figure titled 'one GPU budget, three schedules'. Three horizontal "
            "lanes stacked vertically, each showing the same two GPUs over the same span of "
            "time, left to right. LANE A labelled 'CONVENTIONAL': both GPU rows alternate in "
            "wide blocks — an ochre gold block spanning both GPUs marked 'generate', then a "
            "brick red block spanning both marked 'train', repeating; between the blocks draw "
            "pale warm grey hatched gaps. LANE B labelled 'CONCURRENT': the two GPU rows are "
            "split — the upper row is a continuous ochre gold bar 'generate', the lower row a "
            "continuous brick red bar 'train'; thin slate blue arrows point from the lower bar "
            "up to the upper bar at regular intervals, each tagged 'weights'. LANE C labelled "
            "'CONCURRENT + IN-FLIGHT': identical to lane B but the slate blue arrows are more "
            "frequent and land INSIDE the ochre bar without breaking it, with a small tag "
            "'mid-sequence'. To the right of each lane, a small pale grey block whose height "
            "suggests how much idle time that schedule wastes — tallest for A, shortest for C."
        ),
    },
    {
        "filename": "the_bubble.png",
        "context": (
            "A figure titled 'where the time goes' showing WHY conventional RL idles. A "
            "horizontal time axis. Draw about eight stacked horizontal bars, one per sequence "
            "being generated, in ochre gold — the bars END AT VERY DIFFERENT TIMES, a few "
            "early, most in the middle, and one straggler running far to the right. After each "
            "bar ends, the gap until the straggler finishes is a pale warm grey hatched region. "
            "A vertical dashed ink line at the straggler's end labelled 'everyone waits for the "
            "slowest sequence'. Beneath, a small curve showing the running batch size DECAYING "
            "over time, annotated 'the batch drains'. Right side annotation: 'then the whole "
            "cluster stops generating and starts training'."
        ),
    },
    {
        "filename": "inflight_update.png",
        "context": (
            "A close-up mechanism figure titled 'an in-flight weight update'. Centre: ONE long "
            "horizontal bar representing a single sequence being generated left to right, made "
            "of small ochre gold token squares. About one third along, a narrow vertical slate "
            "blue band labelled 'engine pauses BETWEEN STEPS'. Crucially the token bar "
            "CONTINUES past the band without restarting — tokens after the band are a slightly "
            "deeper shade, showing they came from newer weights. A second such band two thirds "
            "along. Below the bar, a wide rounded card labelled 'KV CACHE — retained, not "
            "recomputed' with a small forest green check mark. A short annotation reads "
            "'generation resumes immediately; nothing is thrown away'. In pale ink at the "
            "bottom right, a contrast note: 'the alternative: stop the world, finish the batch, "
            "then swap weights'."
        ),
    },
    {
        "filename": "split_vs_colocate.png",
        "context": (
            "A two-panel comparison titled 'two ways to spend the same GPUs'. LEFT panel "
            "'COLOCATE': two GPU cards drawn as rounded rectangles, each containing BOTH an "
            "ochre gold 'generate' block and a brick red 'train' block stacked vertically with "
            "a small clock icon between them and the caption 'take turns — memory must be "
            "swapped'. A small pale grey bar underneath labelled 'idle while swapping'. RIGHT "
            "panel 'SPLIT': two GPU cards side by side — the left card entirely brick red "
            "labelled 'trains, always', the right card entirely ochre gold labelled 'generates, "
            "always' — with a bold slate blue arrow between them labelled 'weights over "
            "NVLink'. Beneath the right panel a much shorter pale grey bar. The visual message "
            "is that splitting removes the swap."
        ),
    },
    {
        "filename": "staleness.png",
        "context": (
            "A figure titled 'the price of not waiting'. Left: a single horizontal sequence bar "
            "divided into four coloured segments running from a pale washed tone on the left to "
            "a rich brick red on the right, each segment tagged with a small increasing version "
            "marker; under the left end 'earliest tokens: older weights', under the right end "
            "'latest tokens: newest weights'. Right: a simple two-box balance — an ochre gold "
            "box labelled 'more samples per second' and a brick red box labelled 'each sample "
            "slightly off-policy' sitting on either side of a small fulcrum, with the caption "
            "'learning speed = throughput x effectiveness' written underneath in clean "
            "sans-serif. No numbers anywhere."
        ),
    },
    {
        "filename": "experiment_design.png",
        "context": (
            "A methodology figure titled 'what is held fixed, and what varies'. A large rounded "
            "panel on the left labelled 'IDENTICAL ACROSS ALL ARMS' containing a neat vertical "
            "list of small tagged rows: 'model', 'dataset & prompt order', 'GRPO loss', 'batch "
            "shape', 'hyper-parameters', 'random seed', 'library versions', 'total GPUs' — each "
            "row with a small forest green check. A thin vertical divider. On the right, a "
            "smaller panel labelled 'THE ONLY VARIABLE' containing a single highlighted row in "
            "slate blue reading 'when each GPU does what'. Beneath both, a narrow strip "
            "labelled 'checked automatically after every run' with a small lock icon. Clean, "
            "spare, lots of whitespace."
        ),
    },
]


async def one(pipeline, fig: dict, sem: asyncio.Semaphore) -> tuple[str, bool]:
    """Signature matches the deck's working generator: GenerationInput needs
    source_context + communicative_intent, and the result carries image_path/image_bytes."""
    import shutil
    name = fig["filename"]
    async with sem:
        print(f"[start]  {name}", flush=True)
        try:
            res = await pipeline.generate(GenerationInput(
                source_context=fig["context"] + "\n\nSTYLE:\n" + STYLE,
                communicative_intent=fig["context"][:300],
                diagram_type=M,
            ))
        except Exception as e:
            print(f"[FAIL]   {name}: {type(e).__name__}: {e}", flush=True)
            return name, False
        src = getattr(res, "image_path", None)
        if src and os.path.exists(src):
            shutil.copy2(src, OUT / name)
            print(f"[done]   {name}", flush=True)
            return name, True
        data = getattr(res, "image_bytes", None)
        if data:
            (OUT / name).write_bytes(data)
            print(f"[done]   {name}", flush=True)
            return name, True
        print(f"[FAIL]   {name}: no image", flush=True)
        return name, False


async def main(patterns: list[str], workers: int) -> None:
    figs = [f for f in FIGURES
            if not patterns or any(p in f["filename"] for p in patterns)]
    print(f"generating {len(figs)} figures -> {OUT}", flush=True)
    settings = Settings(
        vlm_provider="gemini", vlm_model="gemini-2.5-flash",
        image_provider="google_imagen", image_model="gemini-3-pro-image-preview",
        refinement_iterations=2, output_dir=str(WORK), output_resolution="2k",
        GOOGLE_API_KEY=os.environ["GOOGLE_API_KEY"],
    )
    pipeline = PaperBananaPipeline(settings=settings)
    sem = asyncio.Semaphore(workers)
    res = await asyncio.gather(*[one(pipeline, f, sem) for f in figs])
    ok = sum(1 for _, s in res if s)
    print(f"\n=== SUMMARY ===  ok={ok}  failed={len(res) - ok}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("patterns", nargs="*")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    asyncio.run(main(a.patterns, a.workers))
