# -*- coding: utf-8 -*-
"""
Participant diligence analysis: run after experiment completion.
Saves a report next to the participant's data file (e.g. {subId}_analysis.json).
"""
import json
import os
from collections import defaultdict


# os_type: 0 = one-shot (novel cue and reward in same row), 1 = incremental (novel cue and reward in different rows)
CONDITION_LABELS = {0: "oneshot", 1: "incremental"}
SLIDER_LIKING_RANGE = (-5, 5)   # good_rating
SLIDER_CAUSAL_RANGE = (0, 10)  # bad_rating
# Image 1/2/3 = by frequency: 1=16x, 2=8x, 3=1x (novel). One-shot hypothesis: image 3 should get highest causal rating.


def _get_condition(trial):
    """Return condition label from trial (0=oneshot, 1=incremental) or None."""
    ot = trial.get("os_type")
    if ot is None:
        return None
    return CONDITION_LABELS.get(int(ot), f"os_type_{ot}")


def _slider_value(trial):
    """Extract numeric rating from image-slider-response trial."""
    r = trial.get("response")
    if r is not None:
        try:
            return float(r)
        except (TypeError, ValueError):
            pass
    # Custom plugin may store as question = [image_name, value]
    q = trial.get("question")
    if isinstance(q, (list, tuple)) and len(q) >= 2:
        try:
            return float(q[1])
        except (TypeError, ValueError):
            pass
    # Some plugins store in question_timeline
    qt = trial.get("question_timeline")
    if isinstance(qt, list) and len(qt) >= 1:
        v = qt[0] if isinstance(qt[0], (int, float)) else None
        if v is not None:
            return float(v)
    return None


def _is_liking_trial(trial):
    """Heuristic: liking uses min -5..5, causal 0..10."""
    mn, mx = trial.get("min"), trial.get("max")
    if mn is not None and mx is not None:
        try:
            if int(mn) == -5 and int(mx) == 5:
                return True
            if int(mn) == 0 and int(mx) == 10:
                return False
        except (TypeError, ValueError):
            pass
    return None


def _max_run_same_value(values):
    """Return length of longest run of identical values (for diligence check)."""
    if not values:
        return 0
    best = 1
    cur = 1
    for i in range(1, len(values)):
        if values[i] == values[i - 1]:
            cur += 1
        else:
            best = max(best, cur)
            cur = 1
    return max(best, cur)


def _distribution(values, bins=None):
    """Count values per bin; if bins is None, count each value."""
    if bins is not None:
        out = {b: 0 for b in bins}
        for v in values:
            for b in bins:
                if v <= b if b == bins[-1] else v < b:
                    out[b] += 1
                    break
        return out
    out = defaultdict(int)
    for v in values:
        out[round(v, 2)] += 1
    return dict(out)


def _per_round_slider_by_image(sliders):
    """
    Group slider trials by (session, round); within each round order is
    good_rating_1, good_rating_2, good_rating_3, bad_rating_1, bad_rating_2, bad_rating_3
    → image 1, 2, 3 (frequency: 16x, 8x, 1x novel). Returns list of dicts per round.
    """
    from collections import defaultdict
    key = lambda t: (t.get("session"), t.get("round"))
    rounds = defaultdict(list)
    for t in sliders:
        if t.get("session") is None and t.get("round") is None:
            continue
        rounds[key(t)].append(t)
    out = []
    for (session, round_idx), group in rounds.items():
        group.sort(key=lambda t: t.get("trial_index", 0))
        if len(group) < 6:
            continue
        liking_1 = _slider_value(group[0])
        liking_2 = _slider_value(group[1])
        liking_3 = _slider_value(group[2])
        causal_1 = _slider_value(group[3])
        causal_2 = _slider_value(group[4])
        causal_3 = _slider_value(group[5])
        cond = _get_condition(group[0])
        out.append({
            "session": session,
            "round": round_idx,
            "condition": cond,
            "image_1": {"liking": liking_1, "causal": causal_1},
            "image_2": {"liking": liking_2, "causal": causal_2},
            "image_3": {"liking": liking_3, "causal": causal_3},
        })
    return out


def _rating_means_by_condition_and_image(per_round):
    """Aggregate per-round data into mean liking/causal per condition per image (1,2,3)."""
    by_cond = defaultdict(lambda: {"image_1": {"liking": [], "causal": []}, "image_2": {"liking": [], "causal": []}, "image_3": {"liking": [], "causal": []}})
    for r in per_round:
        cond = r.get("condition") or "unknown"
        for img in ("image_1", "image_2", "image_3"):
            for kind in ("liking", "causal"):
                v = r[img][kind]
                if v is not None:
                    by_cond[cond][img][kind].append(v)
    result = {}
    for cond, data in by_cond.items():
        result[cond] = {}
        for img in ("image_1", "image_2", "image_3"):
            result[cond][img] = {}
            for kind in ("liking", "causal"):
                vals = data[img][kind]
                result[cond][img][kind] = round(sum(vals) / len(vals), 4) if vals else None
                result[cond][img][f"{kind}_n"] = len(vals)
    # Overall: aggregate all rounds
    all_data = defaultdict(lambda: {"image_1": {"liking": [], "causal": []}, "image_2": {"liking": [], "causal": []}, "image_3": {"liking": [], "causal": []}})
    for r in per_round:
        for img in ("image_1", "image_2", "image_3"):
            for kind in ("liking", "causal"):
                v = r[img][kind]
                if v is not None:
                    all_data["overall"][img][kind].append(v)
    result["overall"] = {}
    for img in ("image_1", "image_2", "image_3"):
        result["overall"][img] = {}
        for kind in ("liking", "causal"):
            vals = all_data["overall"][img][kind]
            result["overall"][img][kind] = round(sum(vals) / len(vals), 4) if vals else None
            result["overall"][img][f"{kind}_n"] = len(vals)
    return result


def _check_one_shot_hypothesis(rating_means):
    """
    One-shot hypothesis: in oneshot condition, image 3 (novel) should have highest causal rating.
    Returns (met: bool, message: str). If no oneshot data, returns (None, message).
    """
    oneshot = rating_means.get("oneshot")
    if not oneshot:
        return None, "no_oneshot_condition_data"
    c1 = oneshot.get("image_1", {}).get("causal")
    c2 = oneshot.get("image_2", {}).get("causal")
    c3 = oneshot.get("image_3", {}).get("causal")
    if c1 is None or c2 is None or c3 is None:
        return None, "missing_causal_ratings"
    # Hypothesis: causal(image_3) >= causal(image_1) and causal(image_3) >= causal(image_2)
    met = c3 >= c1 and c3 >= c2
    msg = "image3_highest_causal" if met else "image3_not_highest_causal_exclude"
    return met, msg


def analyze_participant_data(trials):
    """
    Analyze jsPsych trial list for participant diligence.

    Returns a dict with:
    - by_condition: { "oneshot" | "incremental" }: slider stats, tri stats, flags
    - overall: same structure over all trials
    - summary: diligence_score, n_missing_ratings, same_rating_warnings, etc.
    """
    if not trials or not isinstance(trials, list):
        return {"error": "invalid_data", "message": "trials must be a non-empty list"}

    sliders = [t for t in trials if t.get("trial_type") == "image-slider-response"]
    tri_trials = [t for t in trials if t.get("trial_type") == "os_tri_question"]

    def build_slider_stats(subset, label):
        liking_vals = []
        causal_vals = []
        missing = 0
        rts = []
        for t in subset:
            v = _slider_value(t)
            if v is None:
                missing += 1
            else:
                kind = _is_liking_trial(t)
                if kind is True:
                    liking_vals.append(v)
                elif kind is False:
                    causal_vals.append(v)
                else:
                    if SLIDER_LIKING_RANGE[0] <= v <= SLIDER_LIKING_RANGE[1]:
                        liking_vals.append(v)
                    else:
                        causal_vals.append(v)
            rt = t.get("rt")
            if rt is not None:
                try:
                    rts.append(float(rt))
                except (TypeError, ValueError):
                    pass

        liking_dist = _distribution(liking_vals) if liking_vals else {}
        causal_dist = _distribution(causal_vals) if causal_vals else {}
        max_run_liking = _max_run_same_value(liking_vals) if liking_vals else 0
        max_run_causal = _max_run_same_value(causal_vals) if causal_vals else 0

        return {
            "n_trials": len(subset),
            "n_missing_rating": missing,
            "n_liking": len(liking_vals),
            "n_causal": len(causal_vals),
            "liking_distribution": liking_dist,
            "causal_distribution": causal_dist,
            "max_run_same_rating_liking": max_run_liking,
            "max_run_same_rating_causal": max_run_causal,
            "rt_ms_mean": sum(rts) / len(rts) if rts else None,
            "rt_ms_min": min(rts) if rts else None,
            "rt_ms_max": max(rts) if rts else None,
        }

    def build_tri_stats(subset, label):
        missing = 0
        ratings_per_round = []  # list of [v0,v1,v2]
        for t in subset:
            r = t.get("rating")
            if not r or not isinstance(r, list):
                missing += 1
                continue
            vals = []
            for item in r:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    try:
                        vals.append(float(item[1]))
                    except (TypeError, ValueError):
                        pass
            if len(vals) >= 3:
                ratings_per_round.append(vals)
            else:
                missing += 1
        flat = [v for row in ratings_per_round for v in row]
        dist = _distribution(flat) if flat else {}
        max_run = _max_run_same_value(flat) if flat else 0
        return {
            "n_trials": len(subset),
            "n_missing_rating": missing,
            "n_with_rating": len(ratings_per_round),
            "distribution": dist,
            "max_run_same_rating": max_run,
        }

    # Per-image rating means by condition (image 1=16x, 2=8x, 3=1x novel)
    per_round = _per_round_slider_by_image(sliders)
    rating_means_by_condition = _rating_means_by_condition_and_image(per_round)
    one_shot_met, one_shot_msg = _check_one_shot_hypothesis(rating_means_by_condition)
    exclude_by_one_shot_hypothesis = one_shot_met is False  # True = should exclude (did not show expected pattern)

    # By condition (when os_type is present)
    conditions = set()
    for t in sliders + tri_trials:
        c = _get_condition(t)
        if c is not None:
            conditions.add(c)

    by_condition = {}
    for cond in sorted(conditions or [None]):
        key = cond if cond else "overall"
        sub_sliders = [t for t in sliders if _get_condition(t) == cond]
        sub_tri = [t for t in tri_trials if _get_condition(t) == cond]
        by_condition[key] = {
            "slider_ratings": build_slider_stats(sub_sliders, key),
            "triangle_ratings": build_tri_stats(sub_tri, key),
            "rating_means_by_image": rating_means_by_condition.get(key) or rating_means_by_condition.get(cond),
        }

    # Overall (all trials, no condition filter)
    overall_sliders = build_slider_stats(sliders, "overall")
    overall_tri = build_tri_stats(tri_trials, "overall")

    total_missing = overall_sliders["n_missing_rating"] + overall_tri["n_missing_rating"]
    total_slider = overall_sliders["n_trials"]
    total_tri = overall_tri["n_trials"]
    same_rating_warnings = []
    if overall_sliders["max_run_same_rating_liking"] >= 5:
        same_rating_warnings.append(
            f"liking_slider: {overall_sliders['max_run_same_rating_liking']} consecutive same ratings"
        )
    if overall_sliders["max_run_same_rating_causal"] >= 5:
        same_rating_warnings.append(
            f"causal_slider: {overall_sliders['max_run_same_rating_causal']} consecutive same ratings"
        )
    if overall_tri["max_run_same_rating"] >= 5:
        same_rating_warnings.append(
            f"triangle: {overall_tri['max_run_same_rating']} consecutive same ratings"
        )

    # Simple diligence score: 1 - (missing / expected). Clamp 0..1.
    expected_ratings = total_slider + total_tri
    diligence_ratio = 1.0 - (total_missing / expected_ratings) if expected_ratings else 0.0
    diligence_ratio = max(0.0, min(1.0, diligence_ratio))

    summary = {
        "n_slider_trials": total_slider,
        "n_triangle_trials": total_tri,
        "n_missing_ratings": total_missing,
        "diligence_ratio": round(diligence_ratio, 4),
        "same_rating_warnings": same_rating_warnings,
        "conditions_found": list(conditions) if conditions else ["no_os_type_in_data"],
        "one_shot_hypothesis_met": one_shot_met,
        "one_shot_hypothesis_message": one_shot_msg,
        "exclude_by_one_shot_hypothesis": exclude_by_one_shot_hypothesis,
    }

    return {
        "by_condition": by_condition,
        "rating_means_by_condition": rating_means_by_condition,
        "overall": {
            "slider_ratings": overall_sliders,
            "triangle_ratings": overall_tri,
        },
        "summary": summary,
    }


def run_and_save_analysis(data_dir, sub_id, raw_json_string):
    """
    Parse participant JSON, run analysis, and save results next to the data file.

    Writes:
    - {data_dir}/{sub_id}_analysis.json  (full analysis dict)
    - {data_dir}/{sub_id}_report.txt     (short human-readable report)

    Returns the analysis dict, or None on error.
    """
    try:
        data = json.loads(raw_json_string) if isinstance(raw_json_string, str) else raw_json_string
    except (json.JSONDecodeError, TypeError) as e:
        return None

    trials = data if isinstance(data, list) else data.get("trials", data.get("data", []))
    if not isinstance(trials, list):
        return None

    result = analyze_participant_data(trials)
    if "error" in result:
        return result

    out_path_json = os.path.join(data_dir, f"{sub_id}_analysis.json")
    out_path_txt = os.path.join(data_dir, f"{sub_id}_report.txt")

    try:
        with open(out_path_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    except OSError:
        pass

    try:
        lines = [
            f"Participant: {sub_id}",
            f"Diligence ratio: {result['summary']['diligence_ratio']:.2%}",
            f"Missing ratings: {result['summary']['n_missing_ratings']}",
            f"Conditions: {', '.join(result['summary']['conditions_found'])}",
            f"One-shot hypothesis (image3 highest causal): {result['summary'].get('one_shot_hypothesis_met')} ({result['summary'].get('one_shot_hypothesis_message', '')})",
            f"Exclude by one-shot hypothesis: {result['summary'].get('exclude_by_one_shot_hypothesis', False)}",
            "",
        ]
        for w in result["summary"]["same_rating_warnings"]:
            lines.append(f"Warning: {w}")
        if result["summary"]["same_rating_warnings"]:
            lines.append("")
        for cond, means in result.get("rating_means_by_condition", {}).items():
            if not means or cond == "unknown":
                continue
            lines.append(f"--- Rating means ({cond}) ---")
            for img in ("image_1", "image_2", "image_3"):
                m = means.get(img, {})
                lines.append(f"  {img}: liking={m.get('liking')}, causal={m.get('causal')}")
            lines.append("")
        for cond, stats in result.get("by_condition", {}).items():
            lines.append(f"--- {cond} ---")
            s = stats["slider_ratings"]
            lines.append(f"  Slider: n={s['n_trials']}, missing={s['n_missing_rating']}, "
                        f"max_same_run(liking/causal)={s['max_run_same_rating_liking']}/{s['max_run_same_rating_causal']}")
            t = stats["triangle_ratings"]
            lines.append(f"  Triangle: n={t['n_trials']}, missing={t['n_missing_rating']}, max_same_run={t['max_run_same_rating']}")
        with open(out_path_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    except OSError:
        pass

    return result
