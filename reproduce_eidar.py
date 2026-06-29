#!/usr/bin/env python3
"""
Route B: reproduce the thesis's "affected states %" measurement (IDAR/EIDAR)
on our PAYNT (main / 0.4.0), without building the fork.

We port the fork's `count_affected_states` measurement faithfully:
  - get the parent family's optimal scheduler over its quotient sub-MDP;
  - mark each quotient state whose parent-optimal choice is NOT available in a
    child subfamily as "affected";
  - backward-close "affected" through quotient-MDP predecessors (only through
    states the parent scheduler reached);
  - perc_affected = affected / parent_sub_mdp.nr_states * 100, averaged over
    all subfamily measurements.

Target from Roman: ~86% on drone-4-1 at k=1.

Usage:
  python reproduce_eidar.py [model_dir] [time_budget_s]
"""

import sys, time, statistics

import stormpy
import paynt.parser.sketch
import paynt.quotient.pomdp
import paynt.quotient.quotient
import paynt.synthesizer.synthesizer_ar

MODEL = sys.argv[1] if len(sys.argv) > 1 else "models/archive/cav23-saynt/drone-4-1"
BUDGET = float(sys.argv[2]) if len(sys.argv) > 2 else 150.0
K = 1

paynt.quotient.pomdp.PomdpQuotient.initial_memory_size = K

Quotient = paynt.quotient.quotient.Quotient
_orig_split = Quotient.split

measurements = []            # perc_affected per subfamily
_state = {"preds": None, "t0": None, "stop": False}


class _Stop(Exception):
    pass


def compute_predecessors(mdp):
    n = mdp.nr_states
    tm = mdp.transition_matrix
    preds = [set() for _ in range(n)]
    for s in range(n):
        for choice in tm.get_rows_for_group(s):
            for entry in tm.get_row(choice):
                preds[entry.column].add(s)
    return [list(p) for p in preds]


def measure(quotient, family, subfamilies):
    parent_mdp = family.mdp
    result = family.analysis_result.undecided_result()
    scheduler = result.primary.result.scheduler

    if _state["preds"] is None:
        _state["preds"] = compute_predecessors(quotient.quotient_mdp)
    preds = _state["preds"]

    model = parent_mdp.model
    ndi = model.nondeterministic_choice_indices
    qcm = parent_mdp.quotient_choice_map
    qsm = parent_mdp.quotient_state_map

    # parent-optimal global quotient choice per parent-model state
    state_to_qchoice = []
    for s in range(model.nr_states):
        local = scheduler.get_choice(s).get_deterministic_choice()
        state_to_qchoice.append(qcm[ndi[s] + local])

    nq = quotient.quotient_mdp.nr_states
    parent_states = model.nr_states

    for sub in subfamilies:
        compatible = quotient.coloring.selectCompatibleChoices(sub.family)
        parent_state_choice = [-1] * nq
        affected = bytearray(nq)
        for s in range(parent_states):
            qc = state_to_qchoice[s]
            qs = qsm[s]
            parent_state_choice[qs] = qc
            if not compatible.get(qc):
                affected[qs] = 1
        # backward closure through predecessors the parent reached
        queue = [q for q in range(nq) if affected[q]]
        while queue:
            st = queue.pop()
            for pred in preds[st]:
                if not affected[pred] and parent_state_choice[pred] != -1:
                    affected[pred] = 1
                    queue.append(pred)
        num_affected = sum(affected)
        perc = num_affected / parent_states * 100.0
        measurements.append(perc)


def patched_split(self, family):
    subfamilies = _orig_split(self, family)
    if _state["t0"] is None:
        _state["t0"] = time.time()
    try:
        measure(self, family, subfamilies)
    except Exception as e:
        print("  [measure error]", repr(e))
    if len(measurements) % 200 == 0 and measurements:
        print(f"  ... {len(measurements)} subfamilies, "
              f"running avg affected = {statistics.mean(measurements):.1f}%")
    if time.time() - _state["t0"] > BUDGET:
        raise _Stop()
    return subfamilies


Quotient.split = patched_split

print(f"Model: {MODEL}  (k={K})  time budget={BUDGET}s")
quotient = paynt.parser.sketch.Sketch.load_sketch(
    f"{MODEL}/sketch.templ", f"{MODEL}/sketch.props")
print(f"quotient MDP: {quotient.quotient_mdp.nr_states} states, "
      f"{quotient.quotient_mdp.nr_choices} choices")

syn = paynt.synthesizer.synthesizer_ar.SynthesizerAR(quotient)
try:
    syn.synthesize(keep_optimum=True, print_stats=False)
except _Stop:
    print("  [stopped on time budget]")

print("\n================ RESULT ================")
print(f"subfamily measurements: {len(measurements)}")
if measurements:
    print(f"MEAN affected %%   : {statistics.mean(measurements):.1f}")
    print(f"median affected %% : {statistics.median(measurements):.1f}")
    print(f"min / max         : {min(measurements):.1f} / {max(measurements):.1f}")
    print(f"(thesis target on drone-4-1 k=1: ~86%)")
