import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Grid and MDP settings
rows, cols = 4, 4
goal_state = (3, 3)
actions = ['up', 'down', 'left', 'right']
gamma = 0.9
threshold = 1e-4
rewards = np.zeros((rows, cols))
rewards[goal_state] = 100

V = np.zeros((rows, cols))
arrow_map = {
    'up': '↑',
    'down': '↓',
    'left': '←',
    'right': '→',
    'G': 'G'
}

def get_next_state(state, action):
    i, j = state
    if action == 'up' and i > 0:
        return (i - 1, j)
    elif action == 'down' and i < rows - 1:
        return (i + 1, j)
    elif action == 'left' and j > 0:
        return (i, j - 1)
    elif action == 'right' and j < cols - 1:
        return (i, j + 1)
    return state

def extract_policy(V):
    policy = np.full((rows, cols), '', dtype=object)
    for i in range(rows):
        for j in range(cols):
            state = (i, j)
            if state == goal_state:
                policy[i, j] = 'G'
                continue
            values = []
            for action in actions:
                ni, nj = get_next_state(state, action)
                values.append((rewards[ni, nj] + gamma * V[ni, nj], action))
            best_action = max(values)[1]
            policy[i, j] = best_action
    return policy

# Set up persistent figure
fig, ax = plt.subplots()

def plot_policy_value(V, policy, iteration):
    ax.clear()
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

    plt.draw()
    plt.pause(5)  # pause for 5 seconds before next iteration

def value_iteration_visualized():
    global V
    iteration = 0
    plt.ion()  # turn on interactive mode
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
                    ni, nj = get_next_state(state, action)
                    values.append(rewards[ni, nj] + gamma * V[ni, nj])
                best_value = max(values)
                new_V[i, j] = best_value
                delta = max(delta, abs(V[i, j] - best_value))
        V = new_V
        policy = extract_policy(V)
        plot_policy_value(V, policy, iteration)
        iteration += 1
        if delta < threshold:
            break
    print(f"Converged in {iteration} iterations.")
    plt.ioff()
    plt.show()

value_iteration_visualized()
