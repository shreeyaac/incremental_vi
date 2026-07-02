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

# PRISM benchmark models. Override with PRISM_BENCHMARKS for the container/server;
# defaults to the local clone for laptop runs.
BM = os.environ.get(
    "PRISM_BENCHMARKS",
    "/Users/shreeya/storm-work/prism-benchmarks/models/mdps",
)

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


def build_model(path, prop, consts, exact=False):
    prog = stormpy.parse_prism_program(path)
    props = stormpy.parse_properties_for_prism_program(prop, prog)
    prog, props = stormpy.preprocess_symbolic_input(prog, props, consts)
    prog = prog.as_prism_program()
    if exact:
        # exact (rational) build -> exact values -> Algorithm 1 is theoretically
        # sound (best-action ties decided exactly, no delta-approximation).
        model = stormpy.build_sparse_exact_model(prog, props)
    else:
        opts = stormpy.BuilderOptions([p.raw_formula for p in props])
        opts.set_build_state_valuations()
        model = stormpy.build_sparse_model_with_options(prog, opts)
    return model, props


def exact_values_and_strategy(model, prop_obj):
    """Exact (rational) values + optimal strategy from stormpy's exact engine."""
    res = stormpy.model_checking(model, prop_obj, extract_scheduler=True)
    n = model.nr_states
    V = [res.at(s) for s in range(n)]
    sched = res.scheduler
    strat = {s: sched.get_choice(s).get_deterministic_choice() for s in range(n)}
    return V, strat


def exact_modified_values(model, prop_obj, hat_s, hat_a):
    """Exact V_M' for the action-removed MDP via an exact submodel."""
    n = model.nr_states
    nci = model.nondeterministic_choice_indices
    ks = stormpy.BitVector(n, True)
    ka = stormpy.BitVector(model.nr_choices, True)
    ka.set(nci[hat_s] + hat_a, False)
    sub = stormpy.construct_submodel(model, ks, ka)
    subm = sub.model
    res = stormpy.model_checking(subm, prop_obj)
    mapping = list(sub.new_to_old_state_mapping)
    Vp = [None] * n
    for new_s in range(subm.nr_states):
        Vp[mapping[new_s]] = res.at(new_s)
    return Vp, all(v is not None for v in Vp)


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


def run_one(name, num_removals=40, exact=True):
    import time
    path, prop, consts, tspec = REGISTRY[name]
    model, props = build_model(path, prop, consts, exact=exact)
    n = model.nr_states
    T = target_states(model, tspec)
    back = compute_backward_transitions(model)
    if exact:
        V, strat = exact_values_and_strategy(model, props[0])
        tol = 0          # exact ties -> Algorithm 1 sound
        zero = 0
    else:
        env = tight_environment()
        V, strat = compute_values_and_strategy(model, props[0], env=env)
        tol = None       # default float tolerance
        zero = 1e-9
    mm = compute_mec_map(model)

    cands = [s for s in range(n) if n_actions(model, s) >= 2
             and s not in T and V[s] > zero]

    sizes = {"bachelor": [], "exact": [], "approx": []}
    missed = {"bachelor": 0, "exact": 0, "approx": 0}
    times = {"bachelor": [], "exact": [], "approx": []}
    n_eff = 0
    for hs in cands:
        if n_eff >= num_removals:
            break
        ha = strat[hs]
        if exact:
            Vp, ok = exact_modified_values(model, props[0], hs, ha)
        else:
            Vp, ok = build_modified_values(model, props[0], hs, ha, env=env)
        if not ok:
            continue
        # exact: V[s] != Vp[s] is decided exactly; float: use a threshold
        if exact:
            truly = {s for s in range(n) if V[s] != Vp[s]}
        else:
            truly = {s for s in range(n) if abs(V[s] - Vp[s]) > 1e-7}
        if not truly:
            continue
        n_eff += 1
        for m_, fn, kw in [
            ("bachelor", coi_bachelor, dict(strategy=strat, backward_trans=back)),
            ("exact", coi_exact, dict(backward_trans=back, target_states=T,
                                      mec_map=mm, tol=tol)),
            ("approx", coi_approx, dict(backward_trans=back, target_states=T,
                                        mec_map=mm, tol=tol)),
        ]:
            t0 = time.perf_counter()
            c = fn(model, hs, ha, V, **kw)
            times[m_].append(time.perf_counter() - t0)
            sizes[m_].append(100.0 * len(c) / n)
            missed[m_] += len(truly - c)

    return {"name": name, "states": n, "targets": len(T), "exact": exact,
            "n_eff": n_eff, "sizes": sizes, "missed": missed, "times": times}


def print_result(r):
    vmode = "EXACT (rational)" if r.get("exact") else "sound VI 1e-10"
    print(f"\n### {r['name']}  ({r['states']} states, {r['targets']} target, "
          f"{r['n_eff']} effective removals)  values={vmode}")
    if r["n_eff"] == 0:
        print("  (no value-changing removals)")
        return
    print(f"  {'method':<10}{'mean %':>9}{'median %':>10}{'max %':>9}"
          f"{'missed':>8}{'mean ms':>10}{'max ms':>9}")
    print(f"  {'-'*65}")
    for m in ["bachelor", "exact", "approx"]:
        s = r["sizes"][m]
        t = [x * 1000 for x in r["times"][m]]  # ms
        print(f"  {m:<10}{statistics.mean(s):>9.2f}{statistics.median(s):>10.2f}"
              f"{max(s):>9.2f}{r['missed'][m]:>8}"
              f"{statistics.mean(t):>10.2f}{max(t):>9.2f}")


if __name__ == "__main__":
    num = int(os.environ.get("NUM", "40"))
    exact = os.environ.get("EXACT", "1") != "0"
    names = sys.argv[1:] if len(sys.argv) > 1 else list(REGISTRY)
    for nm in names:
        try:
            print_result(run_one(nm, num_removals=num, exact=exact))
        except Exception as e:
            print(f"\n### {nm}: ERROR {type(e).__name__}: {e}")
