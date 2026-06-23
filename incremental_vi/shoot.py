"""
Undershooting for MDPs (Algorithms 3, 4, 5 from the paper).

Instead of zeroing the cone of influence, we *gradually decrease* the values
of states in the cone until we obtain a valid lower bound for the modified
MDP M'. The decrease at each state s is scaled by E(s) ~= eps * P^s(<>hat_s),
the (eps-scaled) maximal probability of eventually reaching hat_s within the
cone. This undershoots far less than uniform decrease, so the resulting lower
bound is tighter and VI converges in fewer iterations.

MEC structure is ignored for now (each state is its own singleton MEC), so the
validity check max_{(s,a) in Best(MEC(hat_s))} B(L)(s,a) >= L(hat_s) reduces to
a single Bellman update at hat_s in M': B_{M'}(L)(hat_s) >= L(hat_s).
Full MEC handling is a TODO.
"""

import math


def _bellman_at_state(model, V, state, removed_actions=None):
    """
    Max over available actions a (in M') of sum_s' delta(state,a,s') * V[s'].
    Returns -1.0 if no action is available.
    """
    matrix = model.transition_matrix
    row_start = matrix.get_row_group_start(state)
    row_end = matrix.get_row_group_end(state)
    best = -1.0
    for action_idx in range(row_end - row_start):
        if removed_actions and state in removed_actions \
                and action_idx in removed_actions[state]:
            continue
        row_idx = row_start + action_idx
        val = sum(entry.value() * V[entry.column]
                  for entry in matrix.get_row(row_idx))
        if val > best:
            best = val
    return best


# ---------------------------------------------------------------------------
# Algorithm 3: propagate the undershoot ratio E through the cone
# ---------------------------------------------------------------------------

def propagate_epsilon(model, cone: set, hat_s: int, epsilon: float,
                      removed_actions: dict = None,
                      iters: int = None, tol: float = 1e-12,
                      max_iters: int = 100_000) -> dict:
    """
    Algorithm 3. Compute E(s) for every state s.

    E(s) = epsilon       for s in cone (initially)
    E(s) = 0             for s not in cone
    Then repeatedly Bellman-update E on M' for all s in cone minus {hat_s},
    holding E(hat_s) = epsilon fixed.

    On convergence E(s) -> epsilon * P^s_cone(<>hat_s).

    iters : number of Bellman iterations. If None, run to convergence
            (a higher count yields a tighter, "better" undershoot).
    """
    E = {s: (epsilon if s in cone else 0.0) for s in range(model.nr_states)}

    interior = [s for s in cone if s != hat_s]
    n_iters = iters if iters is not None else max_iters

    for it in range(n_iters):
        new_E = dict(E)
        max_change = 0.0
        for s in interior:
            val = _bellman_at_state(model, E, s, removed_actions)
            if val < 0:
                val = 0.0
            new_E[s] = val
            max_change = max(max_change, abs(val - E[s]))
        E = new_E
        E[hat_s] = epsilon  # held fixed
        if iters is None and max_change <= tol:
            break

    return E


# ---------------------------------------------------------------------------
# Algorithm 4: undershoot with exact values
# ---------------------------------------------------------------------------

def undershoot_exact(model, hat_s: int, hat_a: int, V_M: list,
                     cone: set, epsilon: float,
                     propagate_iters: int = None,
                     max_rounds: int = 1_000_000):
    """
    Algorithm 4. Given exact values V_M, return a valid lower bound L for the
    modified MDP M' (action hat_a removed from hat_s).

    Returns (L, rounds): the lower-bound valuation and the number of
    undershoot rounds performed.
    """
    removed = {hat_s: {hat_a}}
    L = list(V_M)
    E = propagate_epsilon(model, cone, hat_s, epsilon, removed,
                          iters=propagate_iters)

    rounds = 0
    while L[hat_s] > 0 and rounds < max_rounds:
        for s in cone:
            L[s] = max(0.0, L[s] - E[s])
        rounds += 1
        # validity check: a Bellman update at hat_s in M' does not decrease
        if _bellman_at_state(model, L, hat_s, removed) >= L[hat_s]:
            return L, rounds
    return L, rounds


# ---------------------------------------------------------------------------
# Algorithm 5: undershoot with approximate (delta-lower-bound) values
# ---------------------------------------------------------------------------

def undershoot_approx(model, hat_s: int, hat_a: int, V_tilde: list,
                      cone: set, epsilon: float, delta: float,
                      propagate_iters: int = None,
                      max_rounds: int = 1_000_000):
    """
    Algorithm 5. Given a delta-approximate lower bound V_tilde on the values of
    M, return a valid lower bound L for the modified MDP M'.

    Bumps hat_s up by delta first (making it a delta-upper-bound there), then
    requires at least ceil(delta/eps) rounds before accepting.

    Returns (L, rounds).
    """
    removed = {hat_s: {hat_a}}
    L = list(V_tilde)
    L[hat_s] = V_tilde[hat_s] + delta

    E = propagate_epsilon(model, cone, hat_s, epsilon, removed,
                          iters=propagate_iters)

    min_rounds = math.ceil(delta / epsilon) if epsilon > 0 else 0

    rounds = 0
    while L[hat_s] > 0 and rounds < max_rounds:
        for s in cone:
            L[s] = max(0.0, L[s] - E[s])
        rounds += 1
        if (_bellman_at_state(model, L, hat_s, removed) >= L[hat_s]
                and rounds >= min_rounds):
            return L, rounds
    return L, rounds


# ---------------------------------------------------------------------------
# Uniform undershoot (Section 3.1.1) -- baseline, intentionally "bad"
# ---------------------------------------------------------------------------

def undershoot_uniform(model, hat_s: int, hat_a: int, V_M: list,
                       cone: set, epsilon: float,
                       max_rounds: int = 1_000_000):
    """
    Uniform undershoot baseline (Section 3.1.1): decrease every cone state by
    the same epsilon each round. Included for comparison -- typically worse
    than smart undershooting because it over-decreases states far from hat_s.

    Returns (L, rounds).
    """
    removed = {hat_s: {hat_a}}
    L = list(V_M)
    rounds = 0
    while L[hat_s] > 0 and rounds < max_rounds:
        for s in cone:
            L[s] = max(0.0, L[s] - epsilon)
        rounds += 1
        if _bellman_at_state(model, L, hat_s, removed) >= L[hat_s]:
            return L, rounds
    return L, rounds
