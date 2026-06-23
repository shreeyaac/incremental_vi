#!/usr/bin/env python3
"""
Benchmark runner for the COI algorithms on the thesis POMDP benchmarks
(treated as their underlying MDPs).

For each model, computes V_M once, then for a sample of states with >=2
available actions removes the *optimal* action and:
  - runs all three COI algorithms (bachelor / exact / approx)
  - measures cone size as a fraction of |S|
  - checks correctness (missed states must be 0) by computing V_M' with BVI

Aggregates the cone-size distribution across removals -- this directly
addresses the question of whether the thesis' ~80% cone claim holds or
whether cones are typically small (<10%).

Usage:
  python run_benchmarks.py --model models/pomdp/maze/cheese/sketch.templ
  python run_benchmarks.py --all-small --num-removals 30
"""

import argparse
import glob
import os
import sys
import statistics

sys.path.insert(0, os.path.dirname(__file__))

from incremental_vi.model_utils import (
    load_mdp, get_target_states, compute_backward_transitions,
    compute_values_and_strategy, build_modified_values, tight_environment,
    compute_mec_map,
)
from incremental_vi.coi import coi_bachelor, coi_exact, coi_approx


def find_goal_prop(templ_file: str):
    """Return a Pmax reachability property string for the model's goal."""
    txt = open(templ_file).read()
    if 'label "goal"' in txt:
        return 'Pmax=? [F "goal"]'
    if 'formula goal' in txt:
        return 'Pmax=? [F goal]'
    return None


def has_holes(templ_file: str) -> bool:
    with open(templ_file) as fh:
        return any(line.strip().startswith("hole ") for line in fh)


def n_actions(model, state):
    m = model.transition_matrix
    return m.get_row_group_end(state) - m.get_row_group_start(state)


def has_unique_optimum(model, V, state, tol=1e-6):
    """
    True if `state` has a strictly unique optimal action: its best Bellman
    value exceeds the second-best by more than tol. Removing the optimal action
    at such a state is guaranteed to reduce its value (an "effective" removal),
    which is what makes a removal interesting for the COI comparison.
    """
    m = model.transition_matrix
    rs, re = m.get_row_group_start(state), m.get_row_group_end(state)
    vals = sorted(
        (sum(e.value() * V[e.column] for e in m.get_row(r)) for r in range(rs, re)),
        reverse=True,
    )
    return len(vals) >= 2 and (vals[0] - vals[1]) > tol


def benchmark_model(templ_file: str, num_removals: int = 20,
                    max_states: int = 50000,
                    check_correctness: bool = True):
    """Run the COI benchmark on one model. Returns a result dict or None.

    Values V_M / V_M' and the optimal strategy come from stormpy's model
    checker (fast, exact, MEC-aware). The three COI algorithms run on top.
    """
    prop = find_goal_prop(templ_file)
    if prop is None or has_holes(templ_file):
        return None

    model, props = load_mdp(templ_file, prop)
    n = model.nr_states
    if n > max_states:
        return {"name": templ_file, "states": n, "skipped": "too large"}

    target_states = get_target_states(model)
    backward = compute_backward_transitions(model)
    env = tight_environment()
    V_M, strategy = compute_values_and_strategy(model, props[0], env=env)
    mec_map = compute_mec_map(model)

    # Candidate states: >=2 actions, not target, positive value. We remove the
    # optimal action at each and report cone sizes over the *effective*
    # removals (those that actually change values). Prefer states with a
    # strictly unique Bellman optimum first (those are guaranteed effective),
    # then fall back to others (which may be effective via spurious-fixpoint
    # disconnection that a Bellman-tie test cannot detect).
    base = [s for s in range(n)
            if n_actions(model, s) >= 2
            and s not in target_states and V_M[s] > 1e-9]
    unique = [s for s in base if has_unique_optimum(model, V_M, s)]
    rest = [s for s in base if s not in set(unique)]
    candidates = (unique + rest)[:num_removals]

    sizes = {"bachelor": [], "exact": [], "approx": []}
    missed_total = {"bachelor": 0, "exact": 0, "approx": 0}
    eff_sizes = {"bachelor": [], "exact": [], "approx": []}  # only effective removals
    n_effective = 0

    for hat_s in candidates:
        hat_a = strategy[hat_s]  # remove the optimal action

        cones = {
            "bachelor": coi_bachelor(model, hat_s, hat_a, V_M, strategy, backward),
            "exact": coi_exact(model, hat_s, hat_a, V_M, backward, target_states,
                               mec_map=mec_map),
            "approx": coi_approx(model, hat_s, hat_a, V_M, backward, target_states,
                                 mec_map=mec_map),
        }

        effective = False
        if check_correctness:
            V_Mp, ok = build_modified_values(model, props[0], hat_s, hat_a, env=env)
            if ok:
                truly_changed = {s for s in range(n)
                                 if abs(V_M[s] - V_Mp[s]) > 1e-7}
                if truly_changed:
                    effective = True
                    n_effective += 1
                for m_name, cone in cones.items():
                    missed_total[m_name] += len(truly_changed - cone)

        for m_name, cone in cones.items():
            sizes[m_name].append(100.0 * len(cone) / n)
            if effective:
                eff_sizes[m_name].append(100.0 * len(cone) / n)

    return {
        "name": templ_file,
        "states": n,
        "removals": len(candidates),
        "n_effective": n_effective,
        "sizes": sizes,
        "eff_sizes": eff_sizes,
        "missed_total": missed_total,
        "checked": check_correctness,
    }


def print_result(r):
    if r is None:
        return
    if r.get("skipped"):
        print(f"{r['name']}  ({r['states']} states): SKIPPED ({r['skipped']})")
        return
    name = (r["name"].replace("models/pomdp/", "").replace("models/archive/", "")
            .replace("/sketch.templ", ""))
    print(f"\n### {name}  ({r['states']} states, {r['removals']} removals, "
          f"{r['n_effective']} effective)")
    if r["checked"] and r["n_effective"] == 0:
        print("  (no value-changing removals among sampled states -- "
              "underlying MDP reaches goal too robustly; nothing to measure)")
        return
    # Cone size reported over *effective* removals (those that changed values),
    # as a percentage of |S|. This is what addresses the ~80% vs <10% question.
    print(f"  {'method':<10}{'mean %':>9}{'median %':>10}{'max %':>9}{'missed':>9}")
    print(f"  {'-'*46}")
    for m in ["bachelor", "exact", "approx"]:
        s = r["eff_sizes"][m] if r["checked"] else r["sizes"][m]
        mean = statistics.mean(s) if s else 0
        med = statistics.median(s) if s else 0
        mx = max(s) if s else 0
        missed = r["missed_total"][m] if r["checked"] else "-"
        print(f"  {m:<10}{mean:>9.2f}{med:>10.2f}{mx:>9.2f}{str(missed):>9}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", help="Single model templ file")
    p.add_argument("--all-small", action="store_true",
                   help="Run on all concrete POMDP models up to --max-states")
    p.add_argument("--num-removals", type=int, default=30)
    p.add_argument("--max-states", type=int, default=50000)
    p.add_argument("--no-correctness", action="store_true",
                   help="Skip V_M' computation (faster, no missed-state check)")
    p.add_argument("--glob", default="models/pomdp/**/sketch.templ",
                   help="Glob for --all-small model discovery")
    args = p.parse_args()

    if args.model:
        models = [args.model]
    elif args.all_small:
        models = sorted(glob.glob(args.glob, recursive=True))
    else:
        print("Specify --model <file> or --all-small")
        return

    for f in models:
        try:
            r = benchmark_model(
                f, num_removals=args.num_removals,
                max_states=args.max_states,
                check_correctness=not args.no_correctness,
            )
            print_result(r)
        except Exception as e:
            print(f"\n### {f}: ERROR {e}")


if __name__ == "__main__":
    main()
