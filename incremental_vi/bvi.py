"""Value iteration for MDPs (max reachability probability).

bvi_mdp : two-sided Bounded VI, used to obtain accurate values V_M / V_M'.
vi_mdp  : single-sided (lower-bound) VI, used for the warm-start iteration-count
          comparisons (this is the "VI" referred to in the paper / Overview,
          converging when L^k - L^{k-1} < delta).
"""

import time
from .model_utils import get_best_actions


def vi_mdp(model, target_states: set, zero_states: set,
           removed_actions: dict = None,
           V_init: list = None,
           delta: float = 1e-6, max_iter: int = 100_000,
           quiet: bool = False):
    """
    Single-sided value iteration for the maximal-reachability lower bound.

    Converges when the max change between successive iterates drops below delta
    (i.e. L^k - L^{k-1} < delta), matching the paper's VI stopping criterion.

    V_init : optional warm-start valuation. If None, starts from
             V = 1 on targets, 0 elsewhere.
    Returns (V, iterations, elapsed_seconds).
    """
    n = model.nr_states
    matrix = model.transition_matrix

    if V_init is not None:
        V = list(V_init)
    else:
        V = [1.0 if s in target_states else 0.0 for s in range(n)]

    def bellman_step(cur):
        new_V = list(cur)
        for state in range(n):
            if state in target_states:
                new_V[state] = 1.0
                continue
            if state in zero_states:
                new_V[state] = 0.0
                continue
            row_start = matrix.get_row_group_start(state)
            row_end = matrix.get_row_group_end(state)
            best = -1.0
            for action_idx in range(row_end - row_start):
                if removed_actions and state in removed_actions \
                        and action_idx in removed_actions[state]:
                    continue
                row_idx = row_start + action_idx
                val = sum(entry.value() * cur[entry.column]
                          for entry in matrix.get_row(row_idx))
                if val > best:
                    best = val
            if best >= 0:
                new_V[state] = best
        return new_V

    t0 = time.time()
    iters = 0
    while iters < max_iter:
        new_V = bellman_step(V)
        iters += 1
        change = max(abs(new_V[s] - V[s]) for s in range(n))
        V = new_V
        if not quiet and iters % 1000 == 0:
            print(f"  VI iter {iters}: change={change:.2e}")
        if change <= delta:
            break

    elapsed = time.time() - t0
    return V, iters, elapsed


def bvi_mdp(model, target_states: set, zero_states: set,
            removed_actions: dict = None,
            L_init: list = None, U_init: list = None,
            delta: float = 1e-6, max_iter: int = 100_000,
            quiet: bool = False):
    """
    Run BVI on model (optionally with removed_actions) and return
    (L, U, iterations, elapsed_seconds).

    L_init / U_init: optional warm-start valuations. If None, uses
    standard BVI initialization (L=0 everywhere, U=0 on zero-states
    and 1 elsewhere).
    removed_actions: dict {state: set_of_action_indices} to skip.
    """
    n = model.nr_states
    matrix = model.transition_matrix

    # Initialize lower bound L
    if L_init is not None:
        L = list(L_init)
    else:
        L = [1.0 if s in target_states else 0.0 for s in range(n)]

    # Initialize upper bound U
    if U_init is not None:
        U = list(U_init)
    else:
        U = [0.0 if s in zero_states else
             (1.0 if s in target_states else 1.0)
             for s in range(n)]
        for s in target_states:
            U[s] = 1.0

    def bellman_step(V):
        new_V = list(V)
        for state in range(n):
            if state in target_states:
                new_V[state] = 1.0
                continue
            if state in zero_states:
                new_V[state] = 0.0
                continue
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
            if best >= 0:
                new_V[state] = best
            # if no action available (all removed), value stays 0
        return new_V

    t0 = time.time()
    iters = 0
    while iters < max_iter:
        L = bellman_step(L)
        U = bellman_step(U)
        iters += 1
        gap = max(U[s] - L[s] for s in range(n))
        if not quiet and iters % 1000 == 0:
            print(f"  BVI iter {iters}: gap={gap:.2e}")
        if gap <= delta:
            break

    elapsed = time.time() - t0
    return L, U, iters, elapsed
