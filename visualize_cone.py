#!/usr/bin/env python3
"""
Render an MDP as a graph with the cone-of-influence highlighted, for each of the
three algorithms (bachelor / exact / approx) -- to visually verify that
Algorithm 2 (approx) is a sound over-approximation of Algorithm 1 (exact).

Uses EXACT (rational) values so Algorithm 1 is rigorous. Emits Graphviz and
renders one image per algorithm (plus the ground-truth "value changed" set).

Usage:
  python visualize_cone.py models/test/cone_demo.prism --prop 'Pmax=? [F "goal"]'
  python visualize_cone.py models/test/cone_demo.prism --hat-s 1   # force the modified state
  # output: cone_<algo>.svg  in the current dir
"""

import argparse, os, sys, subprocess
sys.path.insert(0, os.path.dirname(__file__))

import stormpy
from incremental_vi.model_utils import compute_backward_transitions, compute_mec_map
from incremental_vi.coi import coi_bachelor, coi_exact, coi_approx


def build_exact(path, prop, consts=""):
    prog = stormpy.parse_prism_program(path)
    props = stormpy.parse_properties_for_prism_program(prop, prog)
    if consts:
        prog, props = stormpy.preprocess_symbolic_input(prog, props, consts)
        prog = prog.as_prism_program()
    m = stormpy.build_sparse_exact_model(prog, props)
    return m, props


def values_strategy(m, prop_obj):
    res = stormpy.model_checking(m, prop_obj, extract_scheduler=True)
    V = [res.at(s) for s in range(m.nr_states)]
    sched = res.scheduler
    strat = {s: sched.get_choice(s).get_deterministic_choice() for s in range(m.nr_states)}
    return V, strat


def modified_values(m, prop_obj, hs, ha):
    nci = m.nondeterministic_choice_indices
    ks = stormpy.BitVector(m.nr_states, True)
    ka = stormpy.BitVector(m.nr_choices, True); ka.set(nci[hs] + ha, False)
    sub = stormpy.construct_submodel(m, ks, ka)
    res = stormpy.model_checking(sub.model, prop_obj)
    mp = list(sub.new_to_old_state_mapping)
    Vp = [None] * m.nr_states
    for ns in range(sub.model.nr_states):
        Vp[mp[ns]] = res.at(ns)
    return Vp


def n_actions(m, s):
    return m.transition_matrix.get_row_group_end(s) - m.transition_matrix.get_row_group_start(s)


def pick_removal(m, V, strat, T, back, mm, forced_hs=None):
    """Find an (s,a) removal where approx strictly over-approximates exact
    (so the figure is informative). If forced_hs given, use it."""
    cands = ([forced_hs] if forced_hs is not None else
             [s for s in range(m.nr_states) if n_actions(m, s) >= 2 and s not in T and V[s] > 0])
    best = None
    for hs in cands:
        ha = strat[hs]
        Vp = modified_values(m, prop_obj=PROP, hs=hs, ha=ha)
        truly = {s for s in range(m.nr_states) if V[s] != Vp[s]}
        if not truly:
            continue
        ce = coi_exact(m, hs, ha, V, back, T, mec_map=mm, tol=0)
        ca = coi_approx(m, hs, ha, V, back, T, mec_map=mm, tol=0)
        if forced_hs is not None or len(ca) > len(ce):
            return hs, ha, truly
        if best is None:
            best = (hs, ha, truly)
    return best


def render(m, hs, ha, truly, cones, outdir="."):
    tm = m.transition_matrix
    nci = m.nondeterministic_choice_indices
    for algo, cone in cones.items():
        lines = ["digraph cone {", '  rankdir=LR;', '  node [style=filled, fontname="Helvetica"];']
        for s in range(m.nr_states):
            if s in cone:
                fill = "#ff8800"           # in this algorithm's cone
            else:
                fill = "#ffffff"
            border = ', color="#1f77b4", penwidth=3' if s in truly else ''  # ground truth
            shape = "diamond" if s == hs else "ellipse"
            label = f"{s}"
            lines.append(f'  {s} [label="{label}", fillcolor="{fill}", shape={shape}{border}];')
        # edges: optimal-strategy choice solid, others light; removed action dashed red
        for s in range(m.nr_states):
            rs = tm.get_row_group_start(s)
            for ai in range(n_actions(m, s)):
                succs = {e.column for e in tm.get_row(rs + ai) if e.value() > 0}
                removed = (s == hs and ai == ha)
                for d in succs:
                    if removed:
                        style = ' [color=red, style=dashed, label="removed"]'
                    else:
                        style = ' [color="#999999"]'
                    lines.append(f"  {s} -> {d}{style};")
        lines.append('  labelloc="t";')
        lines.append(f'  label="{algo}  (orange = cone, blue border = value actually changed, '
                     f'diamond = modified state {hs})";')
        lines.append("}")
        dot = "\n".join(lines)
        dotfile = os.path.join(outdir, f"cone_{algo}.dot")
        svgfile = os.path.join(outdir, f"cone_{algo}.svg")
        open(dotfile, "w").write(dot)
        engine = "dot" if m.nr_states <= 60 else "sfdp"
        subprocess.run([engine, "-Tsvg", dotfile, "-o", svgfile], check=True)
        print(f"  wrote {svgfile}  (cone {algo}: {sorted(cone)})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--prop", default='Pmax=? [F "goal"]')
    p.add_argument("--hat-s", type=int, default=None)
    p.add_argument("--consts", default="", help="e.g. K=2 for consensus")
    p.add_argument("--outdir", default=".")
    a = p.parse_args()

    PROP = a.prop
    m, props = build_exact(a.model, a.prop, a.consts)
    PROP = props[0]
    print(f"model: {a.model}  ({m.nr_states} states)")
    V, strat = values_strategy(m, props[0])
    T = {s for s in range(m.nr_states) if m.labeling.get_states("goal").get(s)} \
        if m.labeling.contains_label("goal") else \
        {s for s in range(m.nr_states) for lbl in m.labeling.get_labels()
         if lbl not in ("init", "deadlock") and m.labeling.get_states(lbl).get(s)}
    back = compute_backward_transitions(m)
    mm = compute_mec_map(m)

    hs, ha, truly = pick_removal(m, V, strat, T, back, mm, forced_hs=a.hat_s)
    cones = {
        "bachelor": coi_bachelor(m, hs, ha, V, strat, back),
        "exact":    coi_exact(m, hs, ha, V, back, T, mec_map=mm, tol=0),
        "approx":   coi_approx(m, hs, ha, V, back, T, mec_map=mm, tol=0),
    }
    print(f"removal: remove action {ha} at state {hs};  value actually changed: {sorted(truly)}")
    render(m, hs, ha, truly, cones, a.outdir)
