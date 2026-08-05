"""Task + verifiable reward.

Both arms import this. The reward must be cheap and deterministic — it runs on the critical
path of every rollout, and any nondeterminism would show up as a fake difference between arms.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass

SYSTEM = (
    "You are a careful mathematician. Reason step by step, then give the final numeric "
    "answer on the last line in the exact form:\n#### <answer>"
)

_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


@dataclass(frozen=True)
class Problem:
    pid: str
    prompt: str
    answer: str


def _norm(s: str) -> str:
    """Canonicalise a numeric string so '1,024', '1024', '1024.0' compare equal."""
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return s


def extract_answer(text: str) -> str | None:
    """Strict-then-lenient extraction.

    Strict: the '#### x' form we asked for. Lenient fallback: last number in the text.
    The fallback matters — without it a 0.5B model that solves the problem but forgets the
    format scores 0, which would make the reward mostly measure format compliance rather
    than math, and both arms would look equally bad for the wrong reason.
    """
    if "####" in text:
        tail = text.rsplit("####", 1)[1]
        m = _NUM.search(tail)
        if m:
            return _norm(m.group())
    nums = _NUM.findall(text)
    return _norm(nums[-1]) if nums else None


def reward_fn(completion: str, gold: str) -> float:
    """1.0 for a correct final answer, else 0.0. Binary and verifiable — no reward model,
    so there is nothing for the policy to hack (cf. the IPR project, where that was the
    entire subject)."""
    got = extract_answer(completion)
    return 1.0 if got is not None and got == _norm(gold) else 0.0


def load_gsm8k(split: str = "train", seed: int = 0, limit: int | None = None) -> list[Problem]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split)
    probs = []
    for i, row in enumerate(ds):
        gold = row["answer"].rsplit("####", 1)[1].strip()
        probs.append(Problem(pid=f"gsm8k-{split}-{i}", prompt=row["question"].strip(), answer=gold))
    random.Random(seed).shuffle(probs)
    return probs[:limit] if limit else probs


def load_countdown(seed: int = 0, n: int = 20_000, limit: int | None = None) -> list[Problem]:
    """Synthetic countdown: reach a target from 3-4 numbers with + - * /.
    Denser reward signal than GSM8K, which matters if a 0.5B policy is too weak to move
    on GSM8K within the race's wall-clock budget (PLAN.md §6)."""
    rng = random.Random(seed)
    probs = []
    for i in range(n):
        nums = [rng.randint(1, 25) for _ in range(rng.choice([3, 4]))]
        target = nums[0]
        for x in nums[1:]:
            target = target + x if rng.random() < 0.6 else target * x
            if target > 500:
                target //= max(1, x)
        probs.append(Problem(
            pid=f"countdown-{i}",
            prompt=(f"Using each of the numbers {nums} exactly once with + - * /, "
                    f"write an expression equal to {target}."),
            answer=str(target)))
    return probs[:limit] if limit else probs


def load_task(task: str, seed: int = 0, limit: int | None = None) -> list[Problem]:
    if task == "gsm8k":
        return load_gsm8k(seed=seed, limit=limit)
    if task == "countdown":
        return load_countdown(seed=seed, limit=limit)
    raise ValueError(f"unknown task {task!r}")


def build_chat(tokenizer, problem: Problem) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": problem.prompt}],
        tokenize=False, add_generation_prompt=True)
