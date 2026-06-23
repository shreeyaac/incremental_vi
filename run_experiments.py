#!/usr/bin/env python3
"""
Incremental VI experiments runner.

Usage examples:
  python run_experiments.py --prism models/test/test_mdp.prism --hat-s 1 --hat-a 1
  python run_experiments.py --prism models/test/test_mdp.prism --hat-s 1 --hat-a 1 --coi-method all
  python run_experiments.py --prism models/test/test_mdp.prism --hat-s 1 --hat-a 1 --reset-coi-mdp
"""

import argparse
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

from incremental_vi.model_utils import (
    load_mdp, get_target_states, get_zero_states,
    compute_backward_transitions, get_optimal_strategy,
)
from incremental_vi.bvi import bvi_mdp, vi_mdp
from incremental_vi.coi import evaluate_coi, print_coi_table
from incremental_vi.shoot import undershoot_exact, undershoot_uniform


def parse_args():
    p = argparse.ArgumentParser(description="Incremental VI experiments for MDPs")
    p.add_argument("--prism", required=True, help="Path to PRISM model file")
    p.add_argument("--prop", default='Pmax=? [F "goal"]',
                   help="PCTL property string")
    p.add_argument("--hat-s", type=int, required=True,
                   help="State index from which to remove action")
    p.add_argument("--hat-a", type=int, required=True,
                   help="Action index (0-based within state) to remove")
    p.add_argument("--coi-method", choices=["bachelor", "exact", "approx", "all"],
                   default="all", help="Which COI algorithm(s) to run")
    p.add_argument("--delta", type=float, default=1e-6,
                   help="BVI convergence threshold")
    p.add_argument("--approx-delta", type=float, default=0.0,
                   help="Extra tolerance for coi_approx best-action check")
    p.add_argument("--reset-coi-mdp", action="store_true",
                   help="Run warm-start reset experiment (compare to full VI)")
    p.add_argument("--shoot-mdp", action="store_true",
                   help="Run undershooting experiment (Algorithm 4)")
    p.add_argument("--epsilon", type=float, default=0.1,
                   help="Undershoot step size epsilon (0 < eps < 1)")
    p.add_argument("--certificates-mdp", action="store_true",
                   help="[TODO] Run certificate-based lower bound experiment")
    p.add_argument("--target-label", default=None,
                   help="Model label for target states (auto-detected if omitted)")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress BVI progress output")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"Loading model: {args.prism}")
    model, properties = load_mdp(args.prism, args.prop)
    print(f"States: {model.nr_states}  Transitions: {model.nr_transitions}")

    target_states = get_target_states(model, label=args.target_label)
    zero_states = get_zero_states(model, target_states)
    backward_trans = compute_backward_transitions(model)

    print(f"Target states: {sorted(target_states)}")
    print(f"Zero states:   {sorted(zero_states)}")
    print(f"Removing action {args.hat_a} from state {args.hat_s}")

    # ----------------------------------------------------------------
    # Step 1: BVI on original MDP M
    # ----------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Running BVI on original MDP M ...")
    L_M, U_M, iters_M, time_M = bvi_mdp(
        model, target_states, zero_states,
        delta=args.delta, quiet=args.quiet,
    )
    V_M = [(l + u) / 2 for l, u in zip(L_M, U_M)]
    print(f"  Converged in {iters_M} iterations, {time_M:.3f}s")
    print(f"  Values V_M: {[round(v, 4) for v in V_M]}")

    # ----------------------------------------------------------------
    # Step 2: BVI on modified MDP M' (action hat_a removed from hat_s)
    # ----------------------------------------------------------------
    removed = {args.hat_s: {args.hat_a}}
    print(f"\nRunning BVI on modified MDP M' ...")
    L_Mprime, U_Mprime, iters_Mprime, time_Mprime = bvi_mdp(
        model, target_states, zero_states,
        removed_actions=removed,
        delta=args.delta, quiet=args.quiet,
    )
    V_Mprime = [(l + u) / 2 for l, u in zip(L_Mprime, U_Mprime)]
    print(f"  Converged in {iters_Mprime} iterations, {time_Mprime:.3f}s")
    print(f"  Values V_M': {[round(v, 4) for v in V_Mprime]}")

    # ----------------------------------------------------------------
    # Step 3: COI evaluation
    # ----------------------------------------------------------------
    print(f"\n{'='*60}")
    print("COI Evaluation")
    strategy = get_optimal_strategy(model, V_M)
    results = evaluate_coi(
        model, args.hat_s, args.hat_a,
        V_M, V_Mprime,
        backward_trans, strategy,
        target_states,
        delta=args.approx_delta,
    )
    print_coi_table(results)

    # Baseline: single-sided VI on M' from scratch (for warm-start comparisons)
    _, iters_full_vi, time_full_vi = vi_mdp(
        model, target_states, zero_states,
        removed_actions=removed, delta=args.delta, quiet=args.quiet,
    )

    # ----------------------------------------------------------------
    # Step 4 (optional): Reset warm-start experiment
    # ----------------------------------------------------------------
    if args.reset_coi_mdp:
        print(f"\n{'='*60}")
        print("Reset warm-start experiment")

        # Use exact COI (most principled choice for reset)
        cone = results["exact"]["cone"]
        print(f"  Exact COI size: {len(cone)}")

        # Warm start: reset COI states to 0, keep others at V_M
        V_warm = [0.0 if s in cone else V_M[s] for s in range(model.nr_states)]
        for s in target_states:
            V_warm[s] = 1.0

        print("  Running VI with reset warm-start ...")
        _, iters_ws, time_ws = vi_mdp(
            model, target_states, zero_states,
            removed_actions=removed,
            V_init=V_warm,
            delta=args.delta, quiet=args.quiet,
        )

        print(f"\n  {'Metric':<30} {'Full VI':>10} {'Reset warm':>12}")
        print(f"  {'-'*52}")
        print(f"  {'# iterations':<30} {iters_full_vi:>10} {iters_ws:>12}")
        print(f"  {'Time (s)':<30} {time_full_vi:>10.4f} {time_ws:>12.4f}")
        print(f"  {'Time for finding COI':<30} {'N/A':>10} {results['exact']['time']:>12.4f}")

    # ----------------------------------------------------------------
    # Step 5 (optional): Undershooting experiment
    # ----------------------------------------------------------------
    if args.shoot_mdp:
        print(f"\n{'='*60}")
        print("Undershooting experiment")

        cone = results["exact"]["cone"]
        print(f"  Exact COI size: {len(cone)}")
        print(f"  epsilon = {args.epsilon}")

        # Smart undershoot (Algorithm 4) on exact values V_M
        L_shoot, shoot_rounds = undershoot_exact(
            model, args.hat_s, args.hat_a, V_M, cone, args.epsilon,
        )
        print(f"  Smart undershoot: {shoot_rounds} rounds")

        # Sanity: L_shoot must be a valid lower bound (L_shoot[s] <= V_M'[s])
        violations = [s for s in range(model.nr_states)
                      if L_shoot[s] > V_Mprime[s] + 1e-9]
        if violations:
            print(f"  WARNING: lower-bound violated at states {violations[:10]}")
        else:
            print(f"  Lower-bound check OK (L <= V_M' everywhere)")

        # Use the undershoot result as a warm-start for single-sided VI on M'.
        print("  Running VI with undershoot warm-start ...")
        _, iters_us, time_us = vi_mdp(
            model, target_states, zero_states,
            removed_actions=removed,
            V_init=L_shoot,
            delta=args.delta, quiet=args.quiet,
        )

        # Uniform undershoot baseline (Section 3.1.1)
        L_uni, uni_rounds = undershoot_uniform(
            model, args.hat_s, args.hat_a, V_M, cone, args.epsilon,
        )

        print(f"\n  {'Metric':<30} {'Full VI':>10} {'Undershoot':>12}")
        print(f"  {'-'*52}")
        print(f"  {'# iterations':<30} {iters_full_vi:>10} {iters_us:>12}")
        print(f"  {'Time (s)':<30} {time_full_vi:>10.4f} {time_us:>12.4f}")
        print(f"  {'Time for finding COI':<30} {'N/A':>10} {results['exact']['time']:>12.4f}")
        print(f"  {'Undershoot rounds (smart)':<30} {'N/A':>10} {shoot_rounds:>12}")
        print(f"  {'Undershoot rounds (uniform)':<30} {'N/A':>10} {uni_rounds:>12}")

    if args.certificates_mdp:
        print("\n[TODO] Certificate check not yet implemented.")

    print(f"\n{'='*60}")
    print("Done.")


if __name__ == "__main__":
    main()
