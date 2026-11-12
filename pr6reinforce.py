
import numpy as np
import matplotlib.pyplot as plt
import random

# Step 1: Create the maze
# 0 = Free path, 1 = Wall

maze = np.array([
    [0, 0, 0, 1, 0],
    [1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
])

start = (0, 0)
goal = (4, 4)

rows, cols = maze.shape
n_states = rows * cols
n_actions = 4

# Actions: Up, Down, Left, Right
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Step 2: Initialize Q-table

Q = np.zeros((n_states, n_actions))

alpha = 0.1
gamma = 0.9
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995
episodes = 3000
max_steps = 100

def state_id(pos):
    return pos[0] * cols + pos[1]

def get_next_state(pos, action):
    dr, dc = actions[action]
    nr, nc = pos[0] + dr, pos[1] + dc

    # Check boundaries and walls
    if (nr < 0 or nr >= rows or
        nc < 0 or nc >= cols or
        maze[nr, nc] == 1):
        return pos, -5, False

    if (nr, nc) == goal:
        return (nr, nc), 100, True

    return (nr, nc), -1, False

# Step 3: Train the agent

for episode in range(episodes):
    pos = start

    for step in range(max_steps):
        s = state_id(pos)

        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = random.randint(0, n_actions - 1)
        else:
            action = np.argmax(Q[s])

        next_pos, reward, done = get_next_state(pos, action)
        ns = state_id(next_pos)

        # Q-Learning update
        Q[s, action] += alpha * (
            reward + gamma * np.max(Q[ns]) - Q[s, action]
        )

        pos = next_pos

        if done:
            break

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

# Step 4: Find the learned path

pos = start
path = [pos]
visited = {pos}

for _ in range(max_steps):
    s = state_id(pos)
    action = np.argmax(Q[s])

    next_pos, reward, done = get_next_state(pos, action)

    if next_pos == pos or next_pos in visited:
        print("Agent could not find a new path.")
        break

    pos = next_pos
    path.append(pos)
    visited.add(pos)

    if done:
        print("Goal reached!")
        break

print("Learned path:", path)
print("Number of steps:", len(path) - 1)

# Step 5: Visualize the maze and path

display_maze = maze.astype(float)

for r, c in path:
    display_maze[r, c] = 2

display_maze[start] = 3
display_maze[goal] = 4

plt.figure(figsize=(7, 7))
plt.imshow(display_maze, cmap="viridis")

for r in range(rows):
    for c in range(cols):
        if maze[r, c] == 1:
            label = "Wall"
        elif (r, c) == start:
            label = "Start"
        elif (r, c) == goal:
            label = "Goal"
        elif (r, c) in path:
            label = "Path"
        else:
            label = ""

        if label:
            plt.text(
                c, r, label,
                ha="center",
                va="center",
                color="white",
                fontsize=8
            )

plt.xticks(range(cols))
plt.yticks(range(rows))
plt.title("Q-Learning: Maze Navigation")
plt.grid(True)
plt.show()
