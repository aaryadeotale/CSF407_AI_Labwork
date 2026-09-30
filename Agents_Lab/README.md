# Goal-Based Agent: Warehouse Navigation

A goal-based agent that finds a collision-free path from `S` to `G` on the
warehouse grid using breadth-first search (BFS). Standard library only,
Python 3.x.

## Run

```bash
python agents.py
```

This prints the map, the path found (as moves, coordinates and drawn on the
map with `*`), and then runs four tests.

Expected result on the lab map: **20 moves**, 59 cells expanded.

## Files

| File | Purpose |
|------|---------|
| `agents.py` | `Warehouse` (environment), `GoalBasedAgent` (state, goal, search, plan execution), path validator and tests |
| `README.md` | This file |

## Agent design (Task 2)

| Component | In this program |
|-----------|-----------------|
| Environment | 2-D grid: `#` obstacle, `.` free, `S` start, `G` goal |
| Current state | Vehicle position `(row, col)` |
| Goal | State equals the cell marked `G` (`goal_test`) |
| Actions | Up, Down, Left, Right; one square each, never into `#` or off the grid |
| Decision-making | BFS over states builds a plan (list of actions); the agent executes it |

```
          +-----------------------------------------+
          |               GOAL-BASED AGENT          |
          |                                         |
 percept  |   current state --+                     |  action
 (position)-->  (row, col)    |                     |--> Up / Down /
          |                   v                     |    Left / Right
          |   goal (G) ---> SEARCH (BFS) ---> plan  |
          |                   ^                     |
          |   world model ----+                     |
          |   (grid, legal moves)                   |
          +-----------------------------------------+
                      ^                  |
                      |   Warehouse      |
                      +------------------+
                         (environment)
```

## Task 1: Understanding the problem

1. **Environment:** the warehouse grid with fixed obstacles (static, fully
   observable, deterministic, discrete).
2. **Goal:** reach cell `G` from `S`.
3. **Actions:** Up, Down, Left, Right.
4. **Information to maintain:** current position, the goal position, the map
   (which cells are blocked), and during search the visited set and the
   parent of each cell so the path can be rebuilt.
5. **Why goal-based, not simple reflex:** a reflex agent maps the current
   percept straight to an action (e.g. "if free on right, move right") and
   would get stuck in dead ends. This agent has an explicit goal and
   considers the consequences of sequences of actions before acting.

**Think about it (warehouse twice as large):** BFS still works and stays
optimal, but it expands cells in all directions, so work grows with the area
(doubling each side gives about 4x the cells). Extra difficulties: memory for
the frontier and visited set, slower planning, and, in a real warehouse,
moving obstacles, other vehicles, partial sensing and different move costs.
A* with a Manhattan-distance heuristic would usually be the better choice.

## Task 3: Prompt engineering and search choice

**Algorithm chosen:** breadth-first search.

**Why:** all moves cost the same, so BFS guarantees the shortest path, is
complete (finds a path if one exists and terminates with "no path" if not),
and is simple to implement and verify.

**Testing:** `agents.py` validates the path independently of the search (starts
at `S`, ends at `G`, only free cells, each step adjacent, no repeated cell)
and checks four cases: the lab map, an unreachable goal, adjacent start and
goal, and an open room where the result must equal the Manhattan distance.

**Questions to record in your lab report** (answer from your own experience
with the LLM you used):

1. *Did the LLM generate a working program on the first attempt?* Record what
   actually happened when you ran it.
2. *How can the prompt be improved?* Examples: name the algorithm and
   justify it, require an "unreachable goal" message, ask for the path as both
   moves and coordinates, and ask for tests.
3. *What algorithm did the LLM choose?* BFS here; note whichever one yours
   chose.
4. *Why?* Uniform move cost, so BFS gives shortest paths; DFS would not
   guarantee that, and A* or Dijkstra add complexity the problem doesn't need.
