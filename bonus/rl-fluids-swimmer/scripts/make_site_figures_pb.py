#!/usr/bin/env python3
"""White editorial figures for the project SITE + GitHub README.

Same visual register as the other Vizuara production studies (FT/Economist
explainer, pure white, sentence case, no shouting). Semantics held fixed:

    blue    #2196C9  the flow / water / the current
    magenta #D400BC  the learned agent and its routes
    grey    #9AA1AC  the naive baseline / zero motion
    amber   #E39A3E  reward and highlights

    GOOGLE_API_KEY=... python make_site_figures_pb.py [name ...]
"""
import asyncio, os, shutil, sys, traceback

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "website", "figures_pb")

STYLE = (
    "Refined editorial diagram for a serious research publication — the visual register of a "
    "Financial Times or Economist explainer, not a poster. Pure white background. "
    "Restrained use of a brand palette as clean flat fills: blue (#2196C9) for water, flow, "
    "currents and vortices; magenta (#D400BC) for the learned agent and its paths; warm "
    "amber (#E39A3E) for rewards and highlights; near-black (#1F1E1A) for text; light grey "
    "(#9AA1AC) for the naive baseline, zero motion and secondary shapes. "
    "Typography is QUIET and PROFESSIONAL: medium-weight sans-serif, sentence case only — "
    "absolutely NO ALL-CAPS words, no shouting, no oversized numbers, no heavy black banners. "
    "Labels modest in size, the way a caption sits under a chart. Thin precise strokes "
    "(1.5-2px), plenty of white space, generous margins, elements aligned to a clean grid. "
    "Flat vector, no gradients, no 3D, no drop shadows. Calm, precise and expensive — a "
    "diagram a researcher would be happy to put in a paper. Every word spelled correctly. "
    "Do not render any measurement numbers or percentages: concepts only."
)

FIGURES = {
    "hero": (
        "A wide hero figure titled 'teaching two tiny brains to swim'. Two equal panels "
        "split by a thin vertical rule. LEFT panel, subtitle 'where should I go?': a calm "
        "2x2 checkerboard of four blue circular arrows (adjacent ones spinning opposite "
        "ways) representing vortices; a grey dashed straight line from a start dot to a "
        "small flag gets bent and spirals into one vortex, ending in a small grey cross; a "
        "magenta curved path from the same start sweeps AROUND a vortex like a roundabout "
        "and reaches the flag with a small amber star. RIGHT panel, subtitle 'how should I "
        "move at all?': a tiny machine of three magenta circles joined by two thin rods, "
        "shown in two poses (arms long, then one arm short) with a modest arrow to the "
        "right labelled 'net crawl'. Thin caption spanning the bottom: 'one brain is about "
        "a thousand numbers, the other is eight — both trained only by a reward'."
    ),
    "rl_loop": (
        "A beginner explainer titled 'the whole of reinforcement learning, in one loop'. "
        "The classic two-box cycle: top rounded box in magenta labelled 'the agent — a "
        "small table of numbers', bottom rounded box in blue labelled 'the world — a fluid "
        "in motion'. A curved arrow down the right side labelled 'action: what the agent "
        "does'; a curved arrow up the left side labelled 'state: what the agent sees', with "
        "a small amber coin icon and 'reward: how well it went'. Beneath the loop, one "
        "quiet sentence: 'the agent tries things, keeps what earns reward, and repeats — "
        "that is the entire idea'. To the right a small comparison strip: a large rounded "
        "rectangle labelled 'a language model — billions of numbers', a medium square 'our "
        "navigator — about a thousand', a tiny square 'our swimmer — eight'."
    ),
    "river": (
        "A beginner explainer titled 'why aiming at the goal can fail', drawn like a river "
        "crossing seen from above. A wide horizontal band of blue flow-lines (the river) "
        "with small arrows all pointing left, MUCH bolder than the swimmer icons, "
        "indicating the current is stronger than any swimmer. Below the band, a start dot; "
        "above the band, a small flag. A grey path goes straight up from the start toward "
        "the flag but is bent hard to the left by the current, missing the flag entirely "
        "and curling back, ending with a small grey cross labelled 'swept away'. A magenta "
        "path leaves the SAME start pointing diagonally up-and-right, angled INTO the "
        "current, and lands exactly on the flag with a small amber star, labelled 'the "
        "ferryman's trick: aim upstream'. Thin caption: 'when the current is stronger than "
        "you, effort is not a strategy — a route is'."
    ),
    "gridworld": (
        "A bridge figure titled 'a game you already know, made real'. LEFT half: a simple "
        "5x5 grid of squares labelled quietly 'the windy gridworld — a classic textbook "
        "exercise', with a start square, a goal square with a flag, and three thin grey "
        "arrows rising from the middle columns labelled 'wind pushes you off course'. "
        "RIGHT half: the same 5x5 grid but the wind arrows are replaced by a beautiful "
        "blue swirl of curved streamlines forming two small vortices, labelled 'the same "
        "game — but the wind is now a real fluid, solved exactly'. A wide arrow between "
        "the halves labelled 'one change'. Thin caption: 'same states, same actions, same "
        "reward — only the wind grew teeth'."
    ),
    "honey": (
        "A beginner physics explainer titled 'swimming in honey', two stacked rows. TOP "
        "row, labelled 'your world': a swimmer figure doing a big stroke with motion "
        "streaks, caption 'push water back, glide forward — inertia carries you'. BOTTOM "
        "row, labelled 'a microbe's world': the same stroke drawn tiny inside a thick "
        "amber-tinted syrup blob, with NO motion streaks and a small grey 'no glide' tag, "
        "caption 'at micro-scale water feels like honey: stop moving and you stop "
        "instantly. Whatever a stroke pushes forward, its exact reverse pulls back.' A "
        "thin bottom strip: 'so any back-and-forth flapping adds up to exactly zero — to "
        "move at all, your stroke must never retrace itself'."
    ),
    "inchworm": (
        "A plain-language explainer titled 'the crawl the agent invented'. Four small "
        "diagrams of a three-circle machine with two arm links, arranged left to right "
        "with magenta arrows between them, each slightly displaced rightward from the "
        "last: pose 1 'both arms long', pose 2 'squeeze the left arm', pose 3 'squeeze "
        "the right arm too', pose 4 'stretch the left arm back out', then a curved arrow "
        "looping back to pose 1 labelled 'stretch the right arm — and repeat'. Under the "
        "sequence, one quiet sentence in amber: 'squeeze, squeeze, stretch, stretch — "
        "like an inchworm. The loop never retraces itself, so the honey rule cannot "
        "cancel it.' A small side note in grey: 'flapping one arm back and forth — the "
        "retraced stroke — is drawn with a double arrow and a zero'."
    ),
    "oracle": (
        "A methodology explainer titled 'when learning fails, interrogate the world "
        "first'. Three rounded panels left to right, connected by thin arrows. Panel 1, "
        "grey, 'our first environment': a tiny swimmer capsule overwhelmed inside a bold "
        "blue vortex swirl, caption 'the current could spin the swimmer sixteen times "
        "harder than its steering could fight'. Panel 2, amber, 'the oracle': a small "
        "table icon with a magnifying glass, caption 'solve the little world EXACTLY — "
        "no learning, no noise — and read off the best any policy could ever do'. Panel "
        "3, magenta, 'the verdict': a short bar barely taller than a grey reference bar, "
        "caption 'even a perfect brain gains almost nothing here. The environment, not "
        "the algorithm, was the problem.' Thin caption below: 'we kept this failure in "
        "the write-up on purpose — it is the most useful lesson in it'."
    ),
}


async def build(key, prompt):
    from paperbanana import PaperBananaPipeline, GenerationInput, DiagramType
    from paperbanana.core.config import Settings
    settings = Settings(vlm_model="gemini-3.5-flash",
                        image_model="gemini-3-pro-image-preview",
                        refinement_iterations=2, output_dir=OUT)
    r = await PaperBananaPipeline(settings=settings).generate(
        GenerationInput(source_context=prompt, communicative_intent=STYLE,
                        diagram_type=DiagramType.METHODOLOGY))
    dest = os.path.join(OUT, f"w_{key}.png")
    if r.image_path and os.path.exists(r.image_path):
        shutil.copy2(r.image_path, dest); print(f"  OK {key}", flush=True); return True
    print(f"  FAIL {key}", flush=True); return False


def main():
    os.makedirs(OUT, exist_ok=True)
    if not os.environ.get("GOOGLE_API_KEY"):
        sys.exit("set GOOGLE_API_KEY")
    ok = 0
    for k in (sys.argv[1:] or list(FIGURES)):
        try: ok += asyncio.run(build(k, FIGURES[k]))
        except Exception: traceback.print_exc()
    print(f"{ok} figures done", flush=True)


if __name__ == "__main__":
    main()
