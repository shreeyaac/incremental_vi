#!/usr/bin/env python3
"""
Undershooting vs plain VI on PRISM MDP benchmarks.

For each effective optimal-action removal we compute the exact cone, then count
single-sided VI iterations to reconverge on M' starting from:
  - scratch        (1 on target, 0 elsewhere)        -- the baseline "full VI"
  - reset          (V_M, but cone states set to 0)
  - undershoot     (smart undershoot lower bound, Algorithm 4)

A good warm start needs far fewer iterations than full VI. Reset typically helps
little (VI must climb back up from 0 in the cone); smart undershoot lands near
V_M' and converges fast.

Usage:  python run_undershoot_prism.py consensus-coin2-K2 [--num 10] [--eps 0.1]
"""

import argparse, os, sys, statistics
sys.path.insert(0, os.path.dirname(__file__))

import run_prism_mdp as R
from incremental_vi.model_utils import (
    get_zero_states, compute_backward_transitions, compute_mec_map,
    compute_values_and_strategy, tight_environment, build_modified_values,
)
from incremental_vi.bvi import vi_mdp
from incremental_vi.coi import coi_exact
from incremental_vi.shoot import undershoot_exact


def run(name, num=10, eps=0.1, delta=1e-6):
    path, prop, consts, tspec = R.REGISTRY[name]
    model, props = R.build_model(path, prop, consts, exact=False)
    n = model.nr_states
    T = R.target_states(model, tspec)
    Z = get_zero_states(model, T)
    back = compute_backward_transitions(model)
    mm = compute_mec_map(model)
    env = tight_environment()
    V, strat = compute_values_and_strategy(model, props[0], env=env)
    V = [float(v) for v in V]

    rows = []
    for hs in range(n):
        if len(rows) >= num:
            break
        if R.n_actions(model, hs) < 2 or hs in T or V[hs] <= 1e-9:
            continue
        ha = strat[hs]
        Vp, ok = build_modified_values(model, props[0], hs, ha, env=env)
        if not ok:
            continue
        Vp = [float(x) for x in Vp]
        truly = {s for s in range(n) if abs(V[s] - Vp[s]) > 1e-7}
        if not truly:
            continue
        cone = coi_exact(model, hs, ha, V, back, T, mec_map=mm)
        removed = {hs: {ha}}

        _, it_full, _ = vi_mdp(model, T, Z, removed_actions=removed,
                               delta=delta, quiet=True)
        Vreset = [0.0 if s in cone else V[s] for s in range(n)]
        for s in T:
            Vreset[s] = 1.0
        _, it_reset, _ = vi_mdp(model, T, Z, removed_actions=removed,
                                V_init=Vreset, delta=delta, quiet=True)
        L, rounds = undershoot_exact(model, hs, ha, V, cone, eps)
        _, it_shoot, _ = vi_mdp(model, T, Z, removed_actions=removed,
                                V_init=L, delta=delta, quiet=True)
        rows.append((hs, len(cone), it_full, it_reset, it_shoot, rounds))
    return name, n, rows


def report(name, n, rows):
    print(f"\n### {name}  ({n} states)   undershoot vs VI  "
          f"(single-sided VI iterations to reconverge, delta=1e-6)")
    if not rows:
        print("  (no effective removals)")
        return
    print(f"  {'s_hat':>6}{'|cone|':>7}{'VI full':>9}{'VI reset':>10}"
          f"{'VI undershoot':>15}{'shoot rounds':>14}")
    print(f"  {'-'*61}")
    for hs, cs, f, r, s, rd in rows:
        print(f"  {hs:>6}{cs:>7}{f:>9}{r:>10}{s:>15}{rd:>14}")
    fulls = [x[2] for x in rows]; shoots = [x[4] for x in rows]; resets = [x[3] for x in rows]
    print(f"  {'-'*61}")
    print(f"  median:           full={statistics.median(fulls):.0f}  "
          f"reset={statistics.median(resets):.0f}  "
          f"undershoot={statistics.median(shoots):.0f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("names", nargs="*", default=["consensus-coin2-K2"])
    p.add_argument("--num", type=int, default=10)
    p.add_argument("--eps", type=float, default=0.1)
    a = p.parse_args()
    for nm in (a.names or ["consensus-coin2-K2"]):
        try:
            report(*run(nm, num=a.num, eps=a.eps))
        except Exception as e:
            print(f"\n### {nm}: ERROR {type(e).__name__}: {e}")
