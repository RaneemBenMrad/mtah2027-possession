"""Possession by HMM: nearest-team evidence + dead-ball rules + Viterbi smoothing.

    python starter/improved.py            # writes evaluation/results/improved_hmm.csv
    python evaluation/validate_submission.py evaluation/results/improved_hmm.csv

Rules compliance (clarification email): every parameter below is FIXED A PRIORI from football
logic. Nothing is fitted or tuned on the public ground truth or on the video. Use run_evaluation.py
only to check that the file is valid, never to pick between parameter values.
If you want to tune, do it on an external dataset you have the rights to use.

Model
-----
States: H (home has the ball), A (away), D (dead ball).
Per-frame evidence, from the tracking only:
  * ball visible: the closer team gets evidence, softly (exp. decay with distance, capped)
  * ball outside the pitch lines          -> evidence for D
  * ball almost motionless for ~2 s       -> evidence for D (set piece, waiting)
  * ball not visible                      -> neutral: keep the current state
Transitions: long expected durations (possession ~15 s, dead ball ~10 s), so a switch must be
backed by sustained evidence. Viterbi finds the single best state path. Finally, runs shorter than
MIN_RUN are absorbed into their neighbours.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from games import GAME, GAMES, DATA_DIR, require, source_frames, half_video_blocks  # noqa: E402
from baseline import (team_of_each_track, write_submission, RESULTS_DIR,  # noqa: E402
                      FIELD_SEP, GROUP_SEP, MISSING)
from load_smart_tagging import load as load_tagging  # noqa: E402

# ---- Parameters: fixed a priori, NOT tuned on ground truth ----------------------------------
FPS = 25
LAM = 3.0            # m: distance scale of the "who is closer" evidence
CLIP = 4.0           # max |evidence| per frame (nats) so one noisy frame cannot dominate
FAR = 50.0           # m: distance used when a team has no tracked player this frame
BONUS = 0.3          # small preference for H/A over D while the ball is clearly live
STILL_WINDOW = 25    # frames: +/- 1 s window to measure ball motion
STILL_M = 1.0        # m: ball moved less than this over the window -> motionless
EV_OOB = 2.5         # nats for D when the ball is outside the pitch
EV_STILL = 2.0       # nats for D when the ball is motionless
# Smart Tagging "SET PIECES" windows end a dead-ball stretch (a restart). The window is ~5 s and the
# restart instant sits about 2/3 of the way through it: the published kickoff anchors fall at
# 0.70 and 0.66 of the two kickoff windows (see load_smart_tagging.check_time_base). Fixed from
# those anchors, not from possession ground truth.
SP_FRACTION = 0.68
EV_SETPIECE = 1.5    # nats for D before the restart instant, inside a SET PIECES window
MEAN_POSSESSION_S = 15.0
MEAN_DEAD_S = 10.0
MIN_RUN = 2 * FPS    # frames: shortest run kept (2 s)

H, A, D = 0, 1, 2
LETTERS = np.array(["H", "A", "D"])
CACHE = os.path.join(DATA_DIR, "features_cache.npz")   # inside data/, which git ignores


def extract_features(raw_path, team_of_track, n_frames, pitch):
    """Per video frame: ball x, y (m) and distance (m) to the closest home / away player.

    NaN where there is no answer (ball not visible, or the team has no tracked player).
    """
    length, width = pitch
    bx = np.full(n_frames, np.nan, dtype=np.float32)
    by = np.full(n_frames, np.nan, dtype=np.float32)
    d_home = np.full(n_frames, np.nan, dtype=np.float32)
    d_away = np.full(n_frames, np.nan, dtype=np.float32)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                continue
            ball = parts[2]
            if not ball or ball[0] == MISSING:
                continue
            i = int(parts[0]) - 1                      # frame_id counts from 1
            if not 0 <= i < n_frames:
                continue

            c1 = ball.index(",")
            c2 = ball.index(",", c1 + 1)
            x = float(ball[:c1]) * length
            y = float(ball[c1 + 1:c2]) * width
            bx[i], by[i] = x, y

            best = [np.inf, np.inf]
            for slot, group in enumerate(parts[1].split(GROUP_SEP)):
                if not group or group[0] == MISSING:
                    continue
                g1 = group.index(",")
                g2 = group.index(",", g1 + 1)
                dx = float(group[:g1]) * length - x
                dy = float(group[g1 + 1:g2]) * width - y
                dist2 = dx * dx + dy * dy
                team = team_of_track[slot]
                if dist2 < best[team]:
                    best[team] = dist2
            if best[0] < np.inf:
                d_home[i] = best[0] ** 0.5
            if best[1] < np.inf:
                d_away[i] = best[1] ** 0.5
    return bx, by, d_home, d_away


def emissions(bx, by, d_home, d_away, pitch, set_piece_dead=None):
    """(n, 3) log-evidence for H, A, D. Only differences between states matter."""
    length, width = pitch
    n = len(bx)
    visible = ~np.isnan(bx)
    e = np.zeros((n, 3), dtype=np.float64)

    idx = np.arange(n)
    lo = np.maximum(idx - STILL_WINDOW, 0)
    hi = np.minimum(idx + STILL_WINDOW, n - 1)
    with np.errstate(invalid="ignore"):
        displacement = np.hypot(bx[hi] - bx[lo], by[hi] - by[lo])
        still = visible & (displacement < STILL_M)          # NaN compares False
        outside = visible & ((bx < 0) | (bx > length) | (by < 0) | (by > width))

    dh = np.where(np.isnan(d_home), FAR, d_home)
    da = np.where(np.isnan(d_away), FAR, d_away)
    diff = np.clip((da - dh) / LAM, -CLIP, CLIP)             # > 0: home is closer
    e[:, H] = np.where(visible, diff / 2.0, 0.0)
    e[:, A] = np.where(visible, -diff / 2.0, 0.0)

    live = visible & ~outside & ~still
    e[live, H] += BONUS
    e[live, A] += BONUS
    e[outside, D] += EV_OOB
    e[still & ~outside, D] += EV_STILL
    if set_piece_dead is not None:
        e[set_piece_dead, D] += EV_SETPIECE
    return e


def set_piece_mask(n_frames):
    """True on frames inside a SET PIECES window, before the restart instant. From Smart Tagging."""
    mask = np.zeros(n_frames, dtype=bool)
    for event in load_tagging():
        if (event["code"] or "").strip().lower() != "set pieces":
            continue
        first = max(0, event["start_frame"])
        restart = first + int(round(SP_FRACTION * (event["end_frame"] - event["start_frame"])))
        mask[first:min(restart, n_frames)] = True
    return mask


def log_transitions():
    """Row = from, column = to. Expected durations set the self-transition probability."""
    leave_poss = 1.0 / (MEAN_POSSESSION_S * FPS)
    leave_dead = 1.0 / (MEAN_DEAD_S * FPS)
    p = np.zeros((3, 3))
    p[H] = [1 - leave_poss, leave_poss * 0.6, leave_poss * 0.4]
    p[A] = [leave_poss * 0.6, 1 - leave_poss, leave_poss * 0.4]
    p[D] = [leave_dead * 0.5, leave_dead * 0.5, 1 - leave_dead]
    return np.log(p)


def viterbi(e, log_trans):
    n = len(e)
    back = np.empty((n, 3), dtype=np.uint8)
    score = np.zeros(3)
    cols = np.arange(3)
    for t in range(n):
        cand = score[:, None] + log_trans
        best = cand.argmax(axis=0)
        back[t] = best
        score = cand[best, cols] + e[t]
        score -= score.max()                                  # keep numbers small
    path = np.empty(n, dtype=np.uint8)
    path[-1] = score.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t][path[t]]
    return path


def merge_short_runs(path, min_len):
    """Absorb interior runs shorter than min_len into the longer neighbouring run."""
    path = path.copy()
    for _ in range(10):
        change = np.flatnonzero(np.diff(path)) + 1
        starts = np.concatenate(([0], change))
        ends = np.concatenate((change, [len(path)]))
        lens = ends - starts
        changed = False
        for k in range(1, len(starts) - 1):
            if lens[k] < min_len:
                donor = starts[k - 1] if lens[k - 1] >= lens[k + 1] else starts[k + 1]
                path[starts[k]:ends[k]] = path[donor]
                changed = True
        if not changed:
            break
    return path


def load_features(game=GAME, refresh=False):
    if os.path.isfile(CACHE) and not refresh:
        z = np.load(CACHE)
        print("features from cache: {}".format(CACHE))
        return z["bx"], z["by"], z["d_home"], z["d_away"]
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game)
    n_frames = source_frames("tactical", game)
    print("reading {} (a minute or two) ...".format(os.path.basename(raw_path)))
    feats = extract_features(raw_path, team_of_each_track(meta_path), n_frames,
                             GAMES[game]["pitch"])
    np.savez_compressed(CACHE, bx=feats[0], by=feats[1], d_home=feats[2], d_away=feats[3])
    return feats


def n_possessions(path):
    runs = np.concatenate(([True], path[1:] != path[:-1]))
    return int(np.sum(runs & (path != D)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="improved_hmm")
    parser.add_argument("--out", default=RESULTS_DIR)
    parser.add_argument("--no-tagging", action="store_true",
                        help="do not use Smart Tagging (the first version of the pipeline)")
    parser.add_argument("--refresh", action="store_true", help="re-read the raw tracking")
    parser.add_argument("--x-outside-play", action="store_true",
                        help="label halftime / outside-play frames X instead of leaving the "
                             "model's label (check SUBMISSION.md before using this)")
    args = parser.parse_args()

    pitch = GAMES[GAME]["pitch"]
    bx, by, d_home, d_away = load_features(refresh=args.refresh)

    sp = None if args.no_tagging else set_piece_mask(len(bx))
    if sp is not None:
        print("Smart Tagging set-piece frames: {} ({:.1f}%)".format(int(sp.sum()),
                                                                100.0 * sp.mean()))
    e = emissions(bx, by, d_home, d_away, pitch, sp)
    path = viterbi(e, log_transitions())
    path = merge_short_runs(path, MIN_RUN)

    labels = LETTERS[path].astype("<U1")
    if args.x_outside_play:
        inside = np.zeros(len(labels), dtype=bool)
        for _, first, last in half_video_blocks(GAME):
            inside[first:last + 1] = True
        labels[~inside] = "X"

    out = write_submission(os.path.join(args.out, args.name + ".csv"), labels)
    counts = {l: int(np.sum(labels == l)) for l in ("H", "A", "D", "X")}
    print("H {H}  A {A}  D {D}  X {X}   possessions (H/A runs): {p}".format(
        p=n_possessions(path), **counts))
    print("-> {}".format(os.path.relpath(out, ROOT)))
    print("next: python evaluation/validate_submission.py {}".format(os.path.relpath(out, ROOT)))