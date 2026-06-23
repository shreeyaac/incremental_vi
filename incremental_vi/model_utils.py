"""Shared utilities for loading and inspecting stormpy MDP models."""

import stormpy


def load_mdp(prism_file: str, prop: str = 'Pmax=? [F "goal"]'):
    """
    Load a PRISM model and return (model, properties).

    POMDPs are converted to their underlying (fully observable) MDP so that
    construct_submodel and standard model checking work on them. The
    transitions are identical -- only the observation function is dropped.
    """
    program = stormpy.parse_prism_program(prism_file)
    properties = stormpy.parse_properties_for_prism_program(prop, program, None)
    model = stormpy.build_model(program, properties)
    if model.is_partially_observable:
        comp = stormpy.SparseModelComponents(
            transition_matrix=model.transition_matrix,
            state_labeling=model.labeling,
            reward_models=model.reward_models,
        )
        model = stormpy.SparseMdp(comp)
    return model, properties


def tight_environment(precision: str = "1e-10"):
    """
    A stormpy solver environment with sound value iteration and tight
    precision. Needed so that V_M and V_M' agree to well below the
    truly-changed threshold -- the default iterative solver only converges
    to ~1e-6, which otherwise shows up as spurious value changes.
    """
    env = stormpy.Environment()
    env.solver_environment.minmax_solver_environment.precision = \
        stormpy.Rational(precision)
    env.solver_environment.set_force_sound()
    return env


def compute_values_and_strategy(model, prop_obj, env=None):
    """
    Use stormpy's model checker to compute exact values V_M and the optimal
    (memoryless deterministic) strategy. Returns (V, strategy) where V is a
    list and strategy maps state -> chosen action index (0-based within the
    state's row group).

    This is far faster and MEC-aware than the pure-Python bvi_mdp, which is
    reserved for the warm-start iteration-count experiments.
    """
    if env is None:
        env = tight_environment()
    res = stormpy.model_checking(
        model, prop_obj, extract_scheduler=True,
        force_fully_observable=True, environment=env,
    )
    n = model.nr_states
    V = [res.at(s) for s in range(n)]
    sched = res.scheduler
    strategy = {s: sched.get_choice(s).get_deterministic_choice()
                for s in range(n)}  # local index within row group (0-based)
    return V, strategy


def build_modified_values(model, prop_obj, hat_s: int, hat_a: int, env=None):
    """
    Compute V_M' for the MDP with action hat_a removed from state hat_s, using
    stormpy on a submodel. Returns (V_prime, ok) where V_prime is indexed by
    ORIGINAL state index and ok is False if some original state was dropped.
    """
    if env is None:
        env = tight_environment()
    n = model.nr_states
    nci = model.nondeterministic_choice_indices
    keep_states = stormpy.BitVector(n, True)
    keep_actions = stormpy.BitVector(model.nr_choices, True)
    keep_actions.set(nci[hat_s] + hat_a, False)
    sub = stormpy.construct_submodel(model, keep_states, keep_actions)
    subm = sub.model
    res = stormpy.model_checking(
        subm, prop_obj, force_fully_observable=True, environment=env,
    )
    mapping = list(sub.new_to_old_state_mapping)
    V_prime = [None] * n
    for new_s in range(subm.nr_states):
        V_prime[mapping[new_s]] = res.at(new_s)
    ok = all(v is not None for v in V_prime)
    return V_prime, ok


def get_target_states(model, label: str = None) -> set:
    """
    Return the set of target/goal state indices.
    If label is given, use it directly. Otherwise, try common names,
    then fall back to the first non-system label.
    """
    all_labels = model.labeling.get_labels()
    system_labels = {"init", "deadlock"}

    candidates = ([label] if label else []) + ["goal", "target", "done"]
    for lbl in candidates:
        if model.labeling.contains_label(lbl):
            bv = model.labeling.get_states(lbl)
            return {s for s in range(model.nr_states) if bv.get(s)}

    # Fall back to any non-system label
    user_labels = [l for l in all_labels if l not in system_labels]
    if user_labels:
        lbl = user_labels[0]
        bv = model.labeling.get_states(lbl)
        return {s for s in range(model.nr_states) if bv.get(s)}

    raise ValueError(
        f"Cannot find target states. Available labels: {all_labels}. "
        "Pass --target-label explicitly."
    )


def compute_mec_map(model) -> dict:
    """
    Return a dict mapping each state s to the frozenset of states in its
    maximal end component. States not in any reported MEC map to the
    singleton {s}.

    Note: stormpy reports even trivial single-state self-loop MECs (a state
    with a self-loop action), which is exactly what we need -- the self-loop
    is an internal action and the genuine progressing action is a best *exit*.
    """
    n = model.nr_states
    mec_map = {s: frozenset({s}) for s in range(n)}
    for mec in stormpy.get_maximal_end_components(model):
        states = frozenset(st for st, _ in mec)
        for st in states:
            mec_map[st] = states
    return mec_map


def get_zero_states(model, target_states: set) -> set:
    """
    Compute zero states: states from which the target is unreachable.
    Uses backward reachability from target states.
    """
    backward = compute_backward_transitions(model)
    reachable = set(target_states)
    queue = list(target_states)
    while queue:
        s = queue.pop()
        for (pred, _action_idx) in backward.get(s, []):
            if pred not in reachable:
                reachable.add(pred)
                queue.append(pred)
    return set(range(model.nr_states)) - reachable


def compute_backward_transitions(model) -> dict:
    """
    Return a dict mapping each state s to a list of (predecessor_state, action_idx).
    action_idx is 0-based within the predecessor's row group.
    removed_actions: optional {state: set_of_action_indices} to skip.
    """
    matrix = model.transition_matrix
    backward = {s: [] for s in range(model.nr_states)}
    for state in range(model.nr_states):
        row_start = matrix.get_row_group_start(state)
        row_end = matrix.get_row_group_end(state)
        for action_idx, row_idx in enumerate(range(row_start, row_end)):
            for entry in matrix.get_row(row_idx):
                succ = entry.column
                if entry.value() > 0:
                    backward[succ].append((state, action_idx))
    return backward


def bellman_value(model, V: list, state: int, action_idx: int,
                  removed_actions: dict = None) -> float:
    """Compute the one-step Bellman value for (state, action_idx)."""
    if removed_actions and state in removed_actions:
        if action_idx in removed_actions[state]:
            return -1.0  # sentinel: removed action
    matrix = model.transition_matrix
    row_start = matrix.get_row_group_start(state)
    row_idx = row_start + action_idx
    return sum(entry.value() * V[entry.column]
               for entry in matrix.get_row(row_idx))


def get_best_actions(model, V: list, delta: float = 0.0,
                     removed_actions: dict = None) -> dict:
    """
    Return dict: state -> set of action indices that achieve
    within delta of the best Bellman value at that state.
    """
    matrix = model.transition_matrix
    best = {}
    for state in range(model.nr_states):
        row_start = matrix.get_row_group_start(state)
        row_end = matrix.get_row_group_end(state)
        n_actions = row_end - row_start
        vals = []
        for action_idx in range(n_actions):
            if removed_actions and state in removed_actions and action_idx in removed_actions[state]:
                vals.append(-1.0)
            else:
                row_idx = row_start + action_idx
                vals.append(sum(entry.value() * V[entry.column]
                                for entry in matrix.get_row(row_idx)))
        best_val = max(v for v in vals if v >= 0)
        best[state] = {i for i, v in enumerate(vals)
                       if v >= 0 and v >= best_val - delta}
    return best


def get_optimal_strategy(model, V: list,
                         removed_actions: dict = None) -> dict:
    """
    Return dict: state -> single best action index (argmax over Bellman values).
    Ties broken by smallest index.
    """
    best = get_best_actions(model, V, delta=0.0, removed_actions=removed_actions)
    return {s: min(actions) for s, actions in best.items()}
