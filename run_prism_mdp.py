#!/usr/bin/env python3
"""
Run COI Algorithm 1 (exact) and Algorithm 2 (approx) -- plus the bachelor
baseline -- on genuine MDP benchmarks from the PRISM benchmark suite / QVBS.

These are real MDPs with partial reachability (traps, failure modes), unlike the
POMDP underlying MDPs (which were trivial). Values/strategy/V_M' come from
stormpy; the three COI algorithms run on top.

Usage:
  python run_prism_mdp.py                 # run the whole registry
  python run_prism_mdp.py consensus-coin2-K2
"""

import sys, os, json, statistics
sys.path.insert(0, os.path.dirname(__file__))

import stormpy
from incremental_vi.model_utils import (
    compute_backward_transitions, compute_values_and_strategy,
    build_modified_values, tight_environment, compute_mec_map,
)
from incremental_vi.coi import coi_bachelor, coi_exact, coi_approx

BM = "/Users/shreeya/storm-work/prism-benchmarks/models/mdps"

# Each entry: (model_path, prop_str, constants, target_spec)
#   target_spec = ("labels", [(label, negate), ...])   -> AND of (maybe negated) labels
#                 ("valuation", predicate_dict)         -> AND of var==value
REGISTRY = {
    "consensus-coin2-K2": (f"{BM}/consensus/coin2.nm", 'Pmax=? [F "finished"&!"agree"]',
                           "K=2", ("labels", [("finished", False), ("agree", True)])),
    "consensus-coin2-K4": (f"{BM}/consensus/coin2.nm", 'Pmax=? [F "finished"&!"agree"]',
                           "K=4", ("labels", [("finished", False), ("agree", True)])),
    "consensus-coin4-K2": (f"{BM}/consensus/coin4.nm", 'Pmax=? [F "finished"&!"agree"]',
                           "K=2", ("labels", [("finished", False), ("agree", True)])),
    "zeroconf-N1000-K1": (f"{BM}/zeroconf/zeroconf.nm", 'Pmax=? [F (l=4 & ip=1)]',
                          "reset=false,N=1000,K=1", ("valuation", {"l": 4, "ip": 1})),
    "wlan0-collisions":  (f"{BM}/wlan/wlan0.nm", 'Pmax=? [F col=COL]',
                          "COL=2", ("valuation", {"col": 2})),
}


def build_model(path, prop, consts):
    prog = stormpy.parse_prism_program(path)
    props = stormpy.parse_properties_for_prism_program(prop, prog)
    prog, props = stormpy.preprocess_symbolic_input(prog, props, consts)
    prog = prog.as_prism_program()
    opts = stormpy.BuilderOptions([p.raw_formula for p in props])
    opts.set_build_state_valuations()
    model = stormpy.build_sparse_model_with_options(prog, opts)
    return model, props


def target_states(model, spec):
    kind, data = spec
    n = model.nr_states
    if kind == "labels":
        bvs = [(model.labeling.get_states(lbl), neg) for lbl, neg in data]
        return {s for s in range(n)
                if all((bv.get(s) != neg) for bv, neg in bvs)}
    else:  # valuation
        sv = model.state_valuations
        out = set()
        for s in range(n):
            j = json.loads(str(sv.get_json(s)))
            if all(j.get(k) == v for k, v in data.items()):
                out.add(s)
        return out


def n_actions(model, s):
    m = model.transition_matrix
    return m.get_row_group_end(s) - m.get_row_group_start(s)


def run_one(name, num_removals=40):
    path, prop, consts, tspec = REGISTRY[name]
    model, props = build_model(path, prop, consts)
    n = model.nr_states
    env = tight_environment()
    T = target_states(model, tspec)
    back = compute_backward_transitions(model)
    V, strat = compute_values_and_strategy(model, props[0], env=env)
    mm = compute_mec_map(model)

    cands = [s for s in range(n) if n_actions(model, s) >= 2
             and s not in T and V[s] > 1e-9]

    sizes = {"bachelor": [], "exact": [], "approx": []}
    missed = {"bachelor": 0, "exact": 0, "approx": 0}
    n_eff = 0
    for hs in cands:
        if n_eff >= num_removals:
            break
        ha = strat[hs]
        Vp, ok = build_modified_values(model, props[0], hs, ha, env=env)
        if not ok:
            continue
        truly = {s for s in range(n) if abs(V[s] - Vp[s]) > 1e-7}
        if not truly:
            continue
        n_eff += 1
        cones = {
            "bachelor": coi_bachelor(model, hs, ha, V, strat, back),
            "exact": coi_exact(model, hs, ha, V, back, T, mec_map=mm),
            "approx": coi_approx(model, hs, ha, V, back, T, mec_map=mm),
        }
        for m_, c in cones.items():
            sizes[m_].append(100.0 * len(c) / n)
            missed[m_] += len(truly - c)

    return {"name": name, "states": n, "targets": len(T),
            "n_eff": n_eff, "sizes": sizes, "missed": missed}


def print_result(r):
    print(f"\n### {r['name']}  ({r['states']} states, {r['targets']} target, "
          f"{r['n_eff']} effective removals)")
    if r["n_eff"] == 0:
        print("  (no value-changing removals)")
        return
    print(f"  {'method':<10}{'mean %':>9}{'median %':>10}{'max %':>9}{'missed':>9}")
    print(f"  {'-'*46}")
    for m in ["bachelor", "exact", "approx"]:
        s = r["sizes"][m]
        print(f"  {m:<10}{statistics.mean(s):>9.2f}{statistics.median(s):>10.2f}"
              f"{max(s):>9.2f}{r['missed'][m]:>9}")


if __name__ == "__main__":
    num = int(os.environ.get("NUM", "40"))
    names = sys.argv[1:] if len(sys.argv) > 1 else list(REGISTRY)
    for nm in names:
        try:
            print_result(run_one(nm, num_removals=num))
        except Exception as e:
            print(f"\n### {nm}: ERROR {type(e).__name__}: {e}")
