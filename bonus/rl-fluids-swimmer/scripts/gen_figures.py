"""PaperBanana concept figures for the RL-in-fluids bonus lecture.

CONCEPT figures only — anything with a measured number is matplotlib from the real
runs (scripts/make_result_figures.py). Palette matches the veRL/PipelineRL deck so
the bonus lecture reads as part of the same series.

    GOOGLE_API_KEY=... python gen_figures.py [name ...]
"""
import argparse
import asyncio
import os
import shutil
from pathlib import Path

ENV_PATH = "/Users/raj/Desktop/Course_Creator/.env.local"
if not os.environ.get("GOOGLE_API_KEY") and os.path.exists(ENV_PATH):
    for line in open(ENV_PATH):
        if line.startswith("GOOGLE_API_KEY="):
            os.environ["GOOGLE_API_KEY"] = line.strip().split("=", 1)[1]
            break

from paperbanana import DiagramType, GenerationInput, PaperBananaPipeline  # noqa: E402
from paperbanana.core.config import Settings  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "figures" / "pb"
OUT.mkdir(parents=True, exist_ok=True)
WORK = Path("/tmp/pb_fluids")
WORK.mkdir(parents=True, exist_ok=True)

STYLE = (
    "Clean, minimal, modern scientific figure on a WARM CREAM PAPER background, the exact "
    "colour #F7F2E8 — NOT white. Editorial 'Distill.pub' and 3Blue1Brown aesthetic: warm, "
    "analog, beautifully typeset textbook feel. Typography: elegant serif for the title, "
    "clean sans-serif for labels, monospace for any short code-like tag. Use ONLY this "
    "restrained palette: dark warm ink #1F1B16 for text and thin outlines; slate blue "
    "#4A5D7E for water, flow, streamlines and vortices; deep brick red #8B3A3A for the "
    "swimmer / the agent / its trajectory; ochre gold #8E6B2F for reward and highlights; "
    "forest green #3D5A4A for success/checkmarks; muted plum #6B4E7E for the naive or "
    "failing case; pale warm grey #C9BFAC for zero/wasted motion. Soft rounded shapes, "
    "thin 1.5px strokes, generous whitespace, elements on a clean grid. No gradients, no "
    "3D, no neon, no clutter. A hand-tuned figure from a premium physics lecture. "
    "CRITICAL: no measurement numbers, percentages or axis values — concepts only."
)
M = DiagramType.METHODOLOGY

FIGURES = {
    "hero_two_problems": (
        "A wide hero figure titled 'reinforcement learning meets fluid mechanics', split "
        "into two equal panels by a thin vertical rule. LEFT panel, subtitle 'learning "
        "WHERE to steer': a lattice of four soft slate-blue vortices drawn as circular "
        "arrows arranged in a 2x2 checkerboard (adjacent ones spinning opposite ways), and "
        "a small brick-red swimmer (a dot with a short direction arrow) tracing a winding "
        "red dashed path that weaves BETWEEN the vortices toward the top edge, where a "
        "small ochre sun/light icon sits. A plum-coloured second path spirals uselessly "
        "around inside one vortex, labelled quietly 'naive'. RIGHT panel, subtitle "
        "'learning HOW to move': three brick-red circles in a horizontal row joined by two "
        "thin rod links, shown twice (above: arms long; below: one arm shortened), with a "
        "small arrow to the right labelled 'net crawl'. Thin caption spanning the bottom: "
        "'two tiny brains, each a handful of numbers, both trained only by a reward'."
    ),
    "rl_loop_fluids": (
        "A figure titled 'the exact same loop, a much smaller brain', showing the classic "
        "agent-environment RL cycle as two rounded boxes joined by two curved arrows. Top "
        "box, brick red, labelled 'agent — a lookup table'. Bottom box, slate blue, "
        "labelled 'environment — a real incompressible flow field'. Right arrow going "
        "down labelled 'action: pick a compass heading'. Left arrow going up labelled "
        "'state: which map cell am I in' and, in ochre gold, 'reward: arrive, and arrive "
        "fast'. To the right, a small quiet comparison card: a large rounded rectangle "
        "labelled 'language-model policy — billions of numbers', below it a small square "
        "labelled 'the navigator — about a thousand', and below that a TINY square "
        "labelled 'the gait learner — eight'. Caption: 'the loop does not care how big "
        "the policy is'."
    ),
    "gyrotaxis_tumbling": (
        "A physics explainer titled 'why swimming up is not enough', two panels side by "
        "side. LEFT panel 'weak vortex': a small brick-red swimmer drawn as a capsule with "
        "a heavy dot at its bottom end and an arrow pointing up; a curved slate-blue arrow "
        "around it (gentle, thin); a small forest-green check and the caption 'righting "
        "torque wins — the swimmer points up and rises'. RIGHT panel 'strong vortex': the "
        "same capsule now tilted sideways inside a much bolder slate-blue circular arrow; "
        "small motion marks show it being spun; a plum cross and the caption 'the vortex "
        "spins it faster than it can right itself — it tumbles and goes nowhere'. Thin "
        "caption below both: 'the flow itself decides whether obedience works'."
    ),
    "scallop_theorem": (
        "A famous-theorem explainer titled 'the scallop theorem', two stacked rows. TOP "
        "row: a simple hinged scallop shape (two thin wedges joined at a hinge) shown "
        "opening then closing, with left-right arrows between the poses, ending in a pale "
        "grey equals-zero mark and the quiet caption 'open, close, open, close — at "
        "micro-scale, exactly zero net motion. Retrace your steps and the fluid retraces "
        "them for you.' BOTTOM row: four small shape-diagrams of a three-sphere swimmer "
        "(three circles joined by two links, arms long or short) arranged left to right in "
        "a CYCLE with arrows connecting them in sequence and a curved arrow looping back, "
        "each diagram slightly displaced rightward from the previous, ending with a "
        "forest-green arrow labelled 'net motion'. Caption: 'a loop in shape space never "
        "retraces itself — that is the entire secret'."
    ),
    "gait_cycle": (
        "A state-machine diagram titled 'the gait the agent discovered', four rounded "
        "cards arranged in a diamond, each card showing a tiny three-sphere swimmer "
        "pictogram: top card 'both arms long', right card 'left arm short', bottom card "
        "'both arms short', left card 'right arm short'. Bold brick-red arrows connect "
        "them in one consistent rotational direction around the diamond, forming a closed "
        "cycle. In the centre of the diamond, a small ochre-gold tag reading 'one full "
        "cycle = one small step of swimming'. A thin plum arrow off to the side shows a "
        "card pair connected by a double-headed arrow with a grey zero, labelled 'back and "
        "forth = nowhere'. Caption: 'the learner found this loop on its own, from reward "
        "alone'."
    ),
    "state_action_card": (
        "A quiet reference card titled 'what the robot knows', two columns. LEFT column "
        "'state — a coarse position fix': a 6x6 grid of small squares with ONE square "
        "highlighted brick red and a small robot dot inside it, caption 'which cell of the "
        "map it is in — nothing about the flow at all'. RIGHT column 'action — eight "
        "compass headings': a compass rose of eight thin arrows (N, NE, E, SE, S, SW, W, "
        "NW), one arrow bold brick red, caption 'point somewhere and swim; the current "
        "does what it wants'. Bottom strip in ochre gold: 'reward — minus one every tick, "
        "a bonus on arrival. Slow routes and drownings punish themselves.'"
    ),
    "reliability": (
        "A two-panel summary figure titled 'the current decides whether obedience works'. "
        "LEFT panel 'aim straight at the target': a vortex lattice hinted with two soft "
        "slate-blue circular arrows; a plum path leaves a start dot, bends dramatically "
        "away from the dashed straight line to a target flag, and spirals into a closed "
        "loop around a vortex, ending with a small plum cross labelled 'swept away — "
        "never arrives'. RIGHT panel 'the learned detour': same lattice, a brick-red path "
        "that curves WITH one vortex's rotation like a car taking a roundabout, exits on "
        "its far side, and lands on the target flag with a forest-green check. Thin "
        "caption beneath: 'four in ten naive robots never arrive. every trained robot "
        "does.'"
    ),
}


async def one(pipeline, key, prompt, sem):
    async with sem:
        print(f"[start]  {key}", flush=True)
        try:
            r = await pipeline.generate(GenerationInput(
                source_context=prompt + "\n\nSTYLE:\n" + STYLE,
                communicative_intent=prompt[:300],
                diagram_type=M))
        except Exception as e:
            print(f"[FAIL]   {key}: {type(e).__name__}: {e}", flush=True)
            return key, False
        src = getattr(r, "image_path", None)
        if src and os.path.exists(src):
            shutil.copy2(src, OUT / f"{key}.png")
            print(f"[done]   {key}", flush=True)
            return key, True
        print(f"[FAIL]   {key}: no image", flush=True)
        return key, False


async def main(names, workers):
    figs = {k: v for k, v in FIGURES.items() if not names or any(n in k for n in names)}
    print(f"generating {len(figs)} figures -> {OUT}", flush=True)
    settings = Settings(
        vlm_provider="gemini", vlm_model="gemini-2.5-flash",
        image_provider="google_imagen", image_model="gemini-3-pro-image-preview",
        refinement_iterations=2, output_dir=str(WORK), output_resolution="2k",
        GOOGLE_API_KEY=os.environ["GOOGLE_API_KEY"],
    )
    pipeline = PaperBananaPipeline(settings=settings)
    sem = asyncio.Semaphore(workers)
    res = await asyncio.gather(*[one(pipeline, k, v, sem) for k, v in figs.items()])
    ok = sum(1 for _, s in res if s)
    print(f"\n=== SUMMARY ===  ok={ok}  failed={len(res) - ok}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    asyncio.run(main(a.names, a.workers))
