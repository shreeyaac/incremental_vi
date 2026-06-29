#!/usr/bin/env python3
"""
Produce the COI evaluation table from Marta's Overview spec, one row per
action-removal experiment. Columns (algorithm-dependent ones are tuples
(bachelor, exact, approx)):

  Time | # states in cone | # states where value changed | # missed |
  Maximal value drop in cone | Minimal value drop in cone

Usage:
  python coi_table.py <model_dir> [--prop 'Pmax=? [F "goal"]'] [--num N]
"""

import argparse, sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

from incremental_vi.model_utils import (
    load_mdp, get_target_states, compute_backward_transitions,
    compute_values_and_strategy, build_modified_values, tight_environment,
    compute_mec_map,
)
from incremental_vi.coi import coi_bachelor, coi_exact, coi_approx


def n_actions(model, s):
    m = model.transition_matrix
    return m.get_row_group_end(s) - m.get_row_group_start(s)


def run(model_dir, prop, num, approx_delta=0.0):
    templ = os.path.join(model_dir, "sketch.templ")
    model, props = load_mdp(templ, prop)
    n = model.nr_states
    T = get_target_states(model)
    back = compute_backward_transitions(model)
    env = tight_environment()
    V, strat = compute_values_and_strategy(model, props[0], env=env)
    mm = compute_mec_map(model)

    candidates = [s for s in range(n)
                  if n_actions(model, s) >= 2 and s not in T and V[s] > 1e-9]

    rows = []
    for hs in candidates:
        if len(rows) >= num:
            break
        ha = strat[hs]
        Vp, ok = build_modified_values(model, props[0], hs, ha, env=env)
        if not ok:
            continue
        truly = {s for s in range(n) if abs(V[s] - Vp[s]) > 1e-7}
        if not truly:
            continue  # ineffective removal

        cones = {}
        times = {}
        for name, fn, kw in [
            ("bachelor", coi_bachelor, dict(strategy=strat, backward_trans=back)),
            ("exact", coi_exact, dict(backward_trans=back, target_states=T, mec_map=mm)),
            ("approx", coi_approx, dict(backward_trans=back, target_states=T,
                                        delta=approx_delta, mec_map=mm)),
        ]:
            t0 = time.perf_counter()
            cones[name] = fn(model, hs, ha, V, **kw)
            times[name] = time.perf_counter() - t0

        drops = [V[s] - Vp[s] for s in truly]
        rows.append({
            "hs": hs, "ha": ha,
            "time": tuple(times[m] for m in ("bachelor", "exact", "approx")),
            "cone": tuple(len(cones[m]) for m in ("bachelor", "exact", "approx")),
            "changed": tuple(len(cones[m] & truly) for m in ("bachelor", "exact", "approx")),
            "missed": tuple(len(truly - cones[m]) for m in ("bachelor", "exact", "approx")),
            "max_drop": max(drops),
            "min_drop": min(drops),
        })
    return model_dir, n, rows


def fmt_tuple(t, fmt):
    return "(" + ", ".join(fmt.format(x) for x in t) + ")"


def print_table(model_dir, n, rows):
    name = model_dir.rstrip("/").split("/")[-1]
    print(f"\n### {name}  ({n} states)   —  algorithm tuples are (bachelor, exact, approx)\n")
    hdr = ["removal (ŝ,â)", "Time s", "# in cone", "# value changed",
           "# missed", "Max drop", "Min drop"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "|".join(["---"] * len(hdr)) + "|")
    for r in rows:
        print("| {} | {} | {} | {} | {} | {:.4f} | {:.4f} |".format(
            f"({r['hs']},{r['ha']})",
            fmt_tuple(r["time"], "{:.4f}"),
            fmt_tuple(r["cone"], "{}"),
            fmt_tuple(r["changed"], "{}"),
            fmt_tuple(r["missed"], "{}"),
            r["max_drop"], r["min_drop"],
        ))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("model_dir")
    p.add_argument("--prop", default='Pmax=? [F "goal"]')
    p.add_argument("--num", type=int, default=8)
    p.add_argument("--approx-delta", type=float, default=0.0)
    a = p.parse_args()
    md, n, rows = run(a.model_dir, a.prop, a.num, a.approx_delta)
    print_table(md, n, rows)
