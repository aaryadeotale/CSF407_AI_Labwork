# Search Agents — Solved AI Laboratory

This folder contains the completed Python implementation for the **Artificial Intelligence Laboratory: Search and A\***.

The implementation follows the warehouse robot problem and the required A*, BFS, testing, and heuristic experiments in the supplied laboratory sheet. The lab defines states, actions, transitions, initial state, goals and costs, and uses `f(n) = g(n) + h(n)` for A*. fileciteturn1file0L16-L25 fileciteturn1file0L26-L48

## Files

```text
search_agents/
├── search_agents.py
└── README.md
```

## Run

Requirements:

- Python 3
- No specialised AI/ML library is required.

Run:

```bash
python search_agents.py
```

The program performs the complete set of computational experiments.

## Included

### 1. Warehouse A*

The supplied warehouse map is implemented exactly, with:

- `#` = obstacle
- `.` = free cell
- `S` = start
- `G` = goal

The robot can move:

```text
Up
Down
Left
Right
```

and every movement costs `1`. fileciteturn1file0L67-L93

The A* implementation explicitly maintains:

```text
g(n) = cost from start
h(n) = heuristic estimate
f(n) = g(n) + h(n)
```

and reconstructs the final path.

Manhattan distance is:

```text
h(n) = |x - xG| + |y - yG|
```

as specified in the lab. fileciteturn1file0L147-L165

### 2. Required validation tests

The code includes:

- Original warehouse
- Trivial one-step case
- No-solution case
- Alternative-path case

These correspond to the testing requirements in Task 3. fileciteturn1file0L168-L205

### 3. BFS vs A*

The same warehouse is solved with:

- BFS
- A*

and the program reports:

```text
Solution found
Path length
States expanded
```

matching Task 5. fileciteturn1file0L230-L247

### 4. Heuristic investigation

The implementation tests:

```text
Manhattan distance
h(n) = 0
Euclidean distance
2 × Manhattan distance
```

and reports solution status, path length, and states expanded for each version, as required by Task 6. fileciteturn1file0L253-L265

The `2 × Manhattan` experiment is deliberately retained even though it may become non-admissible. The lab introduces admissibility as:

```text
h(n) ≤ h*(n)
```

and asks students to investigate what happens when the heuristic becomes too aggressive. fileciteturn1file0L269-L279

## Search-agent design

### State

A state is represented as:

```python
(row, column)
```

because the robot's current grid position completely determines its available movements.

### Actions

The four actions are represented as row/column changes:

```python
(-1, 0)  # Up
(1, 0)   # Down
(0, -1)  # Left
(0, 1)   # Right
```

### Frontier

A* uses Python's `heapq` priority queue.

Each frontier entry contains the information needed to select the smallest current `f(n)`:

```text
(f, g, counter, state)
```

### Path reconstruction

`came_from` stores the predecessor of every discovered state. Once the goal is reached, the program follows these predecessor links backwards and reverses the resulting sequence.

### Repeated exploration

The implementation maintains the best known `g(n)` for each state and ignores stale/worse frontier entries. This prevents unnecessary repeated expansion.

## Expected conceptual interpretation

A* is an informed search because it uses a heuristic estimate of remaining cost when deciding which state to explore next. The lab contrasts this with blind search such as BFS. fileciteturn1file0L37-L48

For this warehouse:

- Every movement has cost `1`.
- Manhattan distance never overestimates the number of horizontal/vertical moves needed when obstacles can only make the true route longer.
- Therefore Manhattan distance is an admissible heuristic for this movement model.
- `h(n)=0` removes heuristic information and makes A* behave like uniform-cost search, which is equivalent to BFS when every action has equal cost.
- Euclidean distance is also a lower-bound estimate for four-direction movement, although it can provide different search ordering.
- `2 × Manhattan` can overestimate the true remaining cost and therefore loses the admissibility guarantee.

The actual state-expansion counts are produced by the program rather than hard-coded.
