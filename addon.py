import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

# Global MDP parameters
gamma = 0.9
threshold = 1e-4
actions = ['up', 'down', 'left', 'right']
arrow_map = {'up': '↑', 'down': '↓', 'left': '←', 'right': '→', 'G': 'G'}

# Get next state after taking action
def get_next_state(state, action, rows, cols):
    i, j = state
    if action == 'up' and i > 0: return (i - 1, j)
    if action == 'down' and i < rows - 1: return (i + 1, j)
    if action == 'left' and j > 0: return (i, j - 1)
    if action == 'right' and j < cols - 1: return (i, j + 1)
    return state

# Extract best policy from value function
def extract_policy(V, rewards, rows, cols, goal_state):
    policy = np.full((rows, cols), '', dtype=object)
    for i in range(rows):
        for j in range(cols):
            state = (i, j)
            if state == goal_state:
                policy[i, j] = 'G'
                continue
            values = []
            for action in actions:
                ni, nj = get_next_state(state, action, rows, cols)
                values.append((rewards[ni, nj] + gamma * V[ni, nj], action))
            best_action = max(values)[1]
            policy[i, j] = best_action
    return policy

# Plot a single grid's value + policy
def plot_policy_value(V, policy, iteration, rows, cols):
    fig, ax = plt.subplots()
    ax.set_title(f'Value Iteration - Iteration {iteration}')
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_xticks(np.arange(cols + 1))
    ax.set_yticks(np.arange(rows + 1))
    ax.grid(True)

    for i in range(rows):
        for j in range(cols):
            y = rows - 1 - i
            ax.add_patch(patches.Rectangle((j, y), 1, 1, fill=False))
            value = round(V[i, j], 1)
            policy_symbol = arrow_map[policy[i, j]]
            ax.text(j + 0.5, y + 0.65, f'{value}', ha='center', va='center', fontsize=10)
            ax.text(j + 0.5, y + 0.25, policy_symbol, ha='center', va='center', fontsize=16)
    plt.show()

# Run value iteration on a single grid
def value_iteration(rows, cols, visualize=False):
    V = np.zeros((rows, cols))
    rewards = np.zeros((rows, cols))
    goal_state = (rows - 1, cols - 1)
    rewards[goal_state] = 100

    iteration = 0
    while True:
        delta = 0
        new_V = np.copy(V)
        for i in range(rows):
            for j in range(cols):
                state = (i, j)
                if state == goal_state:
                    continue
                values = []
                for action in actions:
                    ni, nj = get_next_state(state, action, rows, cols)
                    values.append(rewards[ni, nj] + gamma * V[ni, nj])
                best_value = max(values)
                new_V[i, j] = best_value
                delta = max(delta, abs(V[i, j] - best_value))
        V = new_V
        iteration += 1
        if delta < threshold:
            break

    policy = extract_policy(V, rewards, rows, cols, goal_state)
    if visualize:
        plot_policy_value(V, policy, iteration, rows, cols)

    return iteration

# Run across increasing grid sizes and plot convergence
def run_experiment_and_plot(max_size=25):
    sizes = list(range(2, max_size + 1))
    iterations_needed = []

    for n in sizes:
        print(f"Running for {n}x{n} grid...")
        iters = value_iteration(n, n)
        iterations_needed.append(iters)

    # Plotting results
    plt.figure(figsize=(8, 5))
    plt.plot(sizes, iterations_needed, marker='o')
    plt.title("Grid Size vs. Iterations to Converge (Value Iteration)")
    plt.xlabel("Grid Size (n x n)")
    plt.ylabel("Iterations")
    plt.grid(True)
    plt.show()

# Optional: visualize a single case
# value_iteration(5, 5, visualize=True)

run_experiment_and_plot(max_size=100)
