"""Separate grading/decision layer. Never decides from YOLO confidence alone."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class OnionInput:
    diameter_mm: float | None  # None = reference not visible -> cannot CHOOSE
    condition: str             # Healthy | Rotten | Sprouted | Damaged (YOLO whole-onion class)
    rot: bool = False
    sprout: bool = False
    damage: bool = False
    confidence: float = 0.0    # YOLO conf of winning onion box

@dataclass
class OnionResult:
    diameter_mm: float | None
    size_grade: str
    condition: str
    rot: bool
    sprout: bool
    damage: bool
    confidence: float
    quality_score: int
    decision: str              # CHOOSE | DO NOT CHOOSE
    reasons: list[str]
    quality_grade: str = "Below B"

def size_grade(diameter_mm: float | None, cfg: dict) -> str:
    if diameter_mm is None:
        return "Unknown"
    for g in cfg["size"]["grades"]:
        if diameter_mm < g["max"] or (
            g["max"] == cfg["size"]["max_accept_mm"]
            and diameter_mm <= g["max"]
        ):
            return g["name"]
    return "Oversize"

def quality_grade(score: int, cfg: dict) -> str:
    for band in cfg["grading"].get("quality_grades", []):
        if score >= band["min_score"]:
            return band["name"]
    return "Unrated"

def quality_score(inp: OnionInput, cfg: dict) -> tuple[int, list[str]]:
    w = cfg["grading"]["weights"]
    s = w["base"]
    notes: list[str] = []
    cond = inp.condition.lower()
    is_rotten = inp.rot or cond == "rotten"
    is_sprouted = inp.sprout or cond == "sprouted"
    is_damaged = inp.damage or cond == "damaged"
    if is_rotten:
        s -= w["rotten_penalty"]; notes.append("rot penalty")
    if is_sprouted:
        s -= w["sprouted_penalty"]; notes.append("sprout penalty")
    if is_damaged:
        s -= w["damaged_penalty"]; notes.append("damage penalty")
    if sum([is_rotten, is_sprouted, is_damaged]) >= 2:
        s -= w["multi_defect_penalty"]; notes.append("multi-defect penalty")
    if inp.confidence < 0.50:
        s -= w["low_conf_penalty"]; notes.append("low-confidence penalty")
    if inp.diameter_mm is not None:
        lo, hi = cfg["size"]["min_accept_mm"], cfg["size"]["max_accept_mm"]
        if inp.diameter_mm is not None and (
            abs(inp.diameter_mm - lo) <= 3 or abs(inp.diameter_mm - hi) <= 3
        ):
            # only penalize borderline if otherwise healthy (avoid double-punish)
            if not (is_rotten or is_sprouted or is_damaged):
                s -= w["borderline_size_penalty"]; notes.append("borderline-size penalty")
    return max(0, min(100, int(s))), notes

def decide(inp: OnionInput, cfg: dict) -> OnionResult:
    lo, hi = cfg["size"]["min_accept_mm"], cfg["size"]["max_accept_mm"]
    grade = size_grade(inp.diameter_mm, cfg)
    accepted_grades = cfg["size"].get("accepted_grades")
    min_q = cfg["grading"]["min_quality_score"]
    min_c = cfg["grading"].get("min_confidence", 0.0)
    score, _ = quality_score(inp, cfg)
    letter_grade = quality_grade(score, cfg)
    accepted_quality_grades = cfg["grading"].get("accepted_quality_grades")
    reasons: list[str] = []
    ok = True
    if inp.diameter_mm is None:
        ok = False; reasons.append("no scale reference visible")
    elif not (lo <= inp.diameter_mm <= hi):
        ok = False; reasons.append(f"diameter {inp.diameter_mm:.1f}mm outside {lo}-{hi}mm")
    if inp.diameter_mm is not None and accepted_grades and grade not in accepted_grades:
        ok = False; reasons.append(f"size grade {grade} is not accepted")
    cond = inp.condition.lower()
    if cond != "healthy":
        ok = False; reasons.append(f"condition={inp.condition}")
    if inp.rot or cond == "rotten":
        ok = False; reasons.append("rot detected")
    if inp.sprout or cond == "sprouted":
        ok = False; reasons.append("sprouting detected")
    if inp.damage or cond == "damaged":
        ok = False; reasons.append("damage detected")
    if score < min_q:
        ok = False; reasons.append(f"quality {score} < {min_q}")
    if accepted_quality_grades and letter_grade not in accepted_quality_grades:
        ok = False; reasons.append(f"quality grade {letter_grade} is not accepted")
    if inp.confidence < min_c:
        ok = False; reasons.append(f"confidence {inp.confidence:.2f} < {min_c}")
    return OnionResult(
        diameter_mm=inp.diameter_mm,
        size_grade=grade,
        condition=inp.condition,
        rot=inp.rot or cond == "rotten",
        sprout=inp.sprout or cond == "sprouted",
        damage=inp.damage or cond == "damaged",
        confidence=inp.confidence,
        quality_score=score,
        decision="CHOOSE" if ok else "DO NOT CHOOSE",
        reasons=reasons,
        quality_grade=letter_grade,
    )

def format_report(r: OnionResult) -> str:
    yn = lambda b: "Detected" if b else "Not detected"
    size = f"{r.diameter_mm:.0f} mm" if r.diameter_mm is not None else "Unknown (no reference)"
    return (
        "ONION QUALITY REPORT\n\n"
        f"Size: {size}\n"
        f"Size Grade: {r.size_grade}\n"
        f"Quality Grade: {r.quality_grade}\n"
        f"Condition: {r.condition}\n"
        f"Rot: {yn(r.rot)}\n"
        f"Sprouting: {yn(r.sprout)}\n"
        f"Damage: {yn(r.damage)}\n"
        f"Confidence: {r.confidence*100:.0f}%\n"
        f"Quality Score: {r.quality_score}/100\n\n"
        f"FINAL DECISION: {r.decision}"
    )
