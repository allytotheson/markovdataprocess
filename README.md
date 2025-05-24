# 🧠 Value Iteration Visualization on a 4x4 Grid

This project demonstrates **Value Iteration**, a fundamental algorithm in **Reinforcement Learning**, by solving a shortest-path problem in a 4x4 grid world using **Markov Decision Process (MDP)** principles. The agent starts at `(0, 0)` and aims to reach the goal at `(3, 3)` with the highest possible reward.

---

## 🚀 Features

- Visualizes the **value function** of each grid cell after every iteration.
- Displays the **optimal policy** using directional arrows (`↑ ↓ ← →`) in each cell.
- Uses **Matplotlib** to dynamically update the visualization in the **same window** every 5 seconds.
- Automatically stops when the value function converges.

---

## 📜 How It Works

- **Grid**: A 4×4 grid with the bottom-right corner `(3, 3)` as the **goal state**, providing a reward of `100`.
- **Actions**: Agent can move `up`, `down`, `left`, or `right` (no diagonal movement).
- **Rewards**:
  - All cells have a reward of `0`.
  - The goal cell `(3, 3)` has a reward of `100`.
- **Bellman Update**:
  For each state \( s \), the value function is updated as:

  \[
  V(s) = \max_a \left[ R(s, a) + \gamma \cdot V(s') \right]
  \]

- **Policy Extraction**:
  After each update, the optimal action (direction) is selected for each state by choosing the action that maximizes the expected return.

---

## 📦 Requirements

- Python 3.6 or higher
- Libraries:
  - `numpy`
  - `matplotlib`

Install dependencies using pip:

```bash
pip install numpy matplotlib
