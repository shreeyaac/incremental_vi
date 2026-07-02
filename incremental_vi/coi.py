"""
Three cone-of-influence algorithms for MDPs after action removal:
  - coi_bachelor : backward BFS through optimal strategy (EIDAR)
  - coi_exact    : Algorithm 1 from paper (all best exits must touch cone)
  - coi_approx   : Algorithm 2 from paper (any best exit within delta touches cone)

coi_exact and coi_approx are EC-aware: they use the best *exits* of each
state's maximal end component, not all best actions. This matters because a
spurious best action (e.g. a self-loop with B(V)(s,a)=V(s)) is internal to the
state's MEC and must not count -- otherwise the exact cone under-approximates
and misses states (observed on models with self-loop ECs).
"""

import time
from .model_utils import get_best_actions, get_optimal_strategy, compute_mec_map


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _successors_of_action(model, state: int, action_idx: int) -> set:
    """Return the set of successor states for (state, action_idx)."""
    matrix = model.transition_matrix
    row_idx = matrix.get_row_group_start(state) + action_idx
    return {entry.column for entry in matrix.get_row(row_idx) if entry.value() > 0}


def _n_actions(model, state: int) -> int:
    matrix = model.transition_matrix
    return matrix.get_row_group_end(state) - matrix.get_row_group_start(state)


def _best_exits(model, mec: frozenset, best_actions: dict) -> list:
    """
    Best exits of an end component `mec`: pairs (s', a) where s' in mec, a is a
    best action at s' (per best_actions), and action a leaves the MEC (has a
    successor outside mec). Returns a list of (state, action_idx, successors).
    """
    exits = []
    for s in mec:
        for a in best_actions.get(s, set()):
            succ = _successors_of_action(model, s, a)
            if not succ <= mec:  # leaves the MEC
                exits.append((s, a, succ))
    return exits


# ---------------------------------------------------------------------------
# coi_bachelor : EIDAR-style backward BFS through optimal strategy
# ---------------------------------------------------------------------------

def coi_bachelor(model, hat_s: int, hat_a: int, V: list,
                 strategy: dict, backward_trans: dict) -> set:
    """
    Compute COI by backward BFS through the optimal strategy sigma.
    A predecessor p enters the cone if sigma(p) leads to any state in the cone.

    This mirrors the EIDAR algorithm:
    "A state is affected if its optimal choice leads to an affected state."
    """
    cone = {hat_s}
    queue = [hat_s]
    while queue:
        s = queue.pop()
        for (pred, action_idx) in backward_trans.get(s, []):
            if pred in cone:
                continue
            # pred enters cone iff its optimal strategy action reaches s (∈ cone)
            if strategy.get(pred) == action_idx:
                cone.add(pred)
                queue.append(pred)
    return cone


# Tolerance for the exact best-action tie. The paper assumes exact values; in
# practice V comes from a sound solver at ~1e-10, so a tiny tolerance is needed
# to avoid excluding a genuine best exit whose Bellman value is V - 1e-11.
_EXACT_EPS = 1e-7


def _initial_cone(model, hat_s, hat_a, best_actions, mec_map, exact):
    """
    Compute the seed cone S0 for the MEC of hat_s, following the MEC
    initialization of Algorithm 1 (exact) / Algorithm 2 (approx).

    Returns (cone, abort) -- if abort is True the removal influences nothing
    and the caller should return the empty set.
    """
    mec_hs = mec_map[hat_s]
    succ_ha = _successors_of_action(model, hat_s, hat_a)
    is_exit = not (succ_ha <= mec_hs)
    exits_hs = _best_exits(model, mec_hs, best_actions)
    hs_is_best_exit = any(s == hat_s and a == hat_a for (s, a, _) in exits_hs)

    if is_exit:
        if exact:
            # Exact: whole MEC influenced only if (hat_s,hat_a) is the UNIQUE
            # best exit; if it is a non-unique best exit, nothing changes.
            if hs_is_best_exit and len(exits_hs) == 1:
                return set(mec_hs), False
            if hs_is_best_exit:
                return {hat_s}, False  # tie remains for the MEC, but hat_s may drop
            return {hat_s}, False
        else:
            # Approx: whole MEC influenced if (hat_s,hat_a) is A best exit.
            if hs_is_best_exit:
                return set(mec_hs), False
            return {hat_s}, False
    else:
        # Internal action of the MEC. If the MEC is non-trivial, removing an
        # internal action may split it so that some MEC states can no longer
        # reach the best exits and lose value. Detecting exactly which states
        # requires the prob-1-path analysis of Algorithm 1; we conservatively
        # seed the whole MEC (sound, may over-approximate within the MEC).
        # A singleton self-loop MEC cannot split, so seed just {hat_s}.
        if len(mec_hs) > 1:
            return set(mec_hs), False
        return {hat_s}, False


def _propagate_cone(model, cone, backward_trans, target_states,
                    best_actions, mec_map, require_all):
    """
    Backward fixpoint: add a predecessor's whole MEC if its best exits point
    into the cone (require_all=True for exact, False=any for approx).
    """
    changed = True
    while changed:
        changed = False
        candidates = set()
        for s in cone:
            for (pred, _) in backward_trans.get(s, []):
                if pred not in cone:
                    candidates.add(pred)

        for pred in candidates:
            if pred in target_states:
                continue
            mec = mec_map[pred]
            exits = _best_exits(model, mec, best_actions)
            if not exits:
                continue
            hits = [bool(succ & cone) for (_s, _a, succ) in exits]
            ok = all(hits) if require_all else any(hits)
            if ok:
                if not mec <= cone:
                    cone |= mec
                    changed = True
    return cone


# ---------------------------------------------------------------------------
# coi_exact : Algorithm 1 from paper
# ---------------------------------------------------------------------------

def coi_exact(model, hat_s: int, hat_a: int, V: list,
              backward_trans: dict, target_states: set,
              delta: float = 0.0, mec_map: dict = None, tol=None) -> set:
    """
    Compute exact COI (Algorithm 1), EC-aware. A predecessor p enters the cone
    iff ALL best exits of MEC(p) have a successor in the current cone.

    tol: best-action tie tolerance. Defaults to _EXACT_EPS (absorbs float
    solver noise). Pass tol=0 when V is EXACT (rational) -- then ties are
    decided exactly and Algorithm 1 is theoretically sound.
    """
    if mec_map is None:
        mec_map = compute_mec_map(model)
    best_actions = get_best_actions(model, V, delta=(_EXACT_EPS if tol is None else tol))

    cone, abort = _initial_cone(model, hat_s, hat_a, best_actions, mec_map,
                                exact=True)
    if abort:
        return set()
    return _propagate_cone(model, cone, backward_trans, target_states,
                           best_actions, mec_map, require_all=True)


# ---------------------------------------------------------------------------
# coi_approx : Algorithm 2 from paper
# ---------------------------------------------------------------------------

def coi_approx(model, hat_s: int, hat_a: int, V: list,
               backward_trans: dict, target_states: set,
               delta: float = 0.0, mec_map: dict = None, tol=None) -> set:
    """
    Compute approximate COI (Algorithm 2), EC-aware. A predecessor p enters the
    cone iff ANY best exit of MEC(p) (best actions within +/-delta of optimal)
    has a successor in the current cone.

    tol: base tie tolerance (default _EXACT_EPS; pass 0 for exact V).
    """
    if mec_map is None:
        mec_map = compute_mec_map(model)
    base_tol = _EXACT_EPS if tol is None else tol
    best_actions = get_best_actions(model, V, delta=max(delta, base_tol))

    cone, abort = _initial_cone(model, hat_s, hat_a, best_actions, mec_map,
                                exact=False)
    if abort:
        return set()
    return _propagate_cone(model, cone, backward_trans, target_states,
                           best_actions, mec_map, require_all=False)


# ---------------------------------------------------------------------------
# evaluate_coi : run all three and produce the comparison table
# ---------------------------------------------------------------------------

def evaluate_coi(model, hat_s: int, hat_a: int,
                 V_orig: list, V_prime: list,
                 backward_trans: dict, strategy: dict,
                 target_states: set,
                 delta: float = 0.0) -> dict:
    """
    Run all three COI algorithms and return metrics dict.

    V_orig  : values of original MDP M
    V_prime : values of modified MDP M' (action hat_a removed from hat_s)
    """
    # Ground truth: states where value actually changed
    n = model.nr_states
    truly_changed = {s for s in range(n) if abs(V_orig[s] - V_prime[s]) > 1e-9}

    mec_map = compute_mec_map(model)
    results = {}

    for name, fn, kwargs in [
        ("bachelor", coi_bachelor,
         dict(strategy=strategy, backward_trans=backward_trans)),
        ("exact", coi_exact,
         dict(backward_trans=backward_trans, target_states=target_states,
              delta=0.0, mec_map=mec_map)),
        ("approx", coi_approx,
         dict(backward_trans=backward_trans, target_states=target_states,
              delta=delta, mec_map=mec_map)),
    ]:
        t0 = time.time()
        cone = fn(model, hat_s, hat_a, V_orig, **kwargs)
        elapsed = time.time() - t0

        missed = truly_changed - cone
        value_drops = [V_orig[s] - V_prime[s] for s in cone if s in truly_changed]
        results[name] = {
            "time": elapsed,
            "cone_size": len(cone),
            "truly_changed": len(truly_changed),
            "missed": len(missed),
            "max_drop": max(value_drops) if value_drops else 0.0,
            "min_drop": min(value_drops) if value_drops else 0.0,
            "cone": cone,
        }

    return results


def print_coi_table(results: dict):
    """Pretty-print the COI evaluation table."""
    methods = ["bachelor", "exact", "approx"]
    header = f"{'Metric':<25}" + "".join(f"{m:>12}" for m in methods)
    print(header)
    print("-" * len(header))
    for key, label in [
        ("time",          "Time (s)"),
        ("cone_size",     "# states in cone"),
        ("truly_changed", "# truly changed"),
        ("missed",        "# missed (must=0)"),
        ("max_drop",      "Max value drop"),
        ("min_drop",      "Min value drop"),
    ]:
        row = f"{label:<25}"
        for m in methods:
            v = results[m][key]
            row += f"{v:>12.4f}" if isinstance(v, float) else f"{v:>12}"
        print(row)
