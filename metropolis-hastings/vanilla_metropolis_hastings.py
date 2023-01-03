import numpy as np
from scipy import stats


def main():
    true_loc = 65.4
    true_scale = 3.21

    target_dist = stats.norm(loc=true_loc, scale=true_scale)

    n_iters = 10000

    proposal_dist = stats.norm
    initial_state = target_dist.rvs()

    trajectory = []
    current_state = initial_state
    for it in range(n_iters):
        print(f"Iteration: {it + 1}")
        proposed_state = proposal_dist.rvs(loc=current_state)
        p_current = target_dist.pdf(current_state)
        p_proposed = target_dist.pdf(proposed_state)

        q_current = proposal_dist.pdf(current_state, loc=proposed_state)
        q_proposed = proposal_dist.pdf(proposed_state, loc=current_state)

        ratio_p_proposed_over_current = p_proposed / p_current
        ratio_q_current_over_proposed = q_current / q_proposed
        alpha = min(1, ratio_p_proposed_over_current * ratio_q_current_over_proposed)

        u = np.random.random()
        current_state = proposed_state if u <= alpha else current_state
        trajectory.append(current_state)

    burn_in = 1000
    trajectory = np.array(trajectory[burn_in:])
    approx_loc = np.mean(trajectory)
    approx_scale = np.std(trajectory)

    print(f"true_loc: {true_loc}; true_scale: {true_scale}")
    print(f"approx_loc: {approx_loc}; approx_scale: {approx_scale}")


if __name__ == "__main__":
    main()
