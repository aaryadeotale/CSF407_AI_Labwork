"""
Goal-based agent for the Warehouse Navigation Problem.

Agent design
------------
Environment : a 2-D grid warehouse ('#' obstacle, '.' free, 'S' start, 'G' goal)
State       : the vehicle's (row, col) position
Goal        : reach the cell marked 'G'
Actions     : Up, Down, Left, Right (one grid square each, cannot enter '#')
Decision    : breadth-first search (BFS) builds a complete plan (a list of
              actions) from the current state to the goal; the agent then
              executes the plan one action at a time.

Why BFS?
--------
Every move costs the same (one grid square), so BFS is
  * complete  : it finds a path whenever one exists, and reports
                failure when none does;
  * optimal   : it returns a path with the fewest moves;
  * simple    : it only needs a queue and a 'visited' set.
Time and memory are O(R*C) for an R x C grid, which is fine here. For much
larger warehouses, A* with a Manhattan-distance heuristic would expand far
fewer cells (see the README).

Run:  python agents.py
"""

from collections import deque

WAREHOUSE_MAP = [
    "#####################",
    "#S....#............G#",
    "#.##....##########..#",
    "#....##.............#",
    "#.######.###.#.###..#",
    "#........#..........#",
    "#####################",
]

# action name -> (row change, column change)
ACTIONS = {
    "Up": (-1, 0),
    "Down": (1, 0),
    "Left": (0, -1),
    "Right": (0, 1),
}


# ----------------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------------
class Warehouse:
    """The environment: a 2-D grid that knows where the walls are."""

    def __init__(self, rows):
        self.grid = []
        for row in rows:
            self.grid.append(list(row))
        self.n_rows = len(self.grid)
        self.n_cols = len(self.grid[0])
        self.start = self._find("S")
        self.goal = self._find("G")

    def _find(self, symbol):
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                if self.grid[r][c] == symbol:
                    return (r, c)
        raise ValueError("Map has no '%s' cell" % symbol)

    def is_free(self, position):
        r, c = position
        inside = 0 <= r < self.n_rows and 0 <= c < self.n_cols
        return inside and self.grid[r][c] != "#"

    def successors(self, position):
        """Yield (action, next_position) for every legal move."""
        for action, (dr, dc) in ACTIONS.items():
            nxt = (position[0] + dr, position[1] + dc)
            if self.is_free(nxt):
                yield action, nxt

    def render(self, path=None):
        """Return the map as text, with the path drawn as '*' if given."""
        canvas = []
        for row in self.grid:
            canvas.append(list(row))
        if path:
            for (r, c) in path:
                if canvas[r][c] == ".":
                    canvas[r][c] = "*"
        lines = []
        for row in canvas:
            lines.append("".join(row))
        return "\n".join(lines)


# ----------------------------------------------------------------------------
# Agent
# ----------------------------------------------------------------------------
class GoalBasedAgent:
    """Perceive state -> know goal -> search for a plan -> execute actions."""

    def __init__(self, warehouse):
        self.warehouse = warehouse
        self.state = warehouse.start      # current state
        self.goal = warehouse.goal        # explicit goal
        self.plan = []                    # list of actions to execute
        self.cells_expanded = 0           # search effort (for analysis)

    def goal_test(self, state):
        return state == self.goal

    def search(self):
        """
        Breadth-first search from self.state to the goal.
        Returns a list of (action, position) steps, or None if unreachable.
        """
        start = self.state
        frontier = deque([start])
        came_from = {start: None}         # position -> (previous position, action)
        self.cells_expanded = 0

        while frontier:
            current = frontier.popleft()
            self.cells_expanded += 1
            if self.goal_test(current):
                return self._reconstruct(came_from, current)
            for action, nxt in self.warehouse.successors(current):
                if nxt not in came_from:  # never revisit a cell
                    came_from[nxt] = (current, action)
                    frontier.append(nxt)
        return None

    @staticmethod
    def _reconstruct(came_from, end):
        steps = []
        node = end
        while came_from[node] is not None:
            previous, action = came_from[node]
            steps.append((action, node))
            node = previous
        steps.reverse()
        return steps

    def formulate_plan(self):
        steps = self.search()
        if steps is None:
            self.plan = None
        else:
            self.plan = steps
        return self.plan

    def act(self):
        """Execute the next action of the plan; return (action, new_state)."""
        action, new_state = self.plan.pop(0)
        self.state = new_state
        return action, new_state


# ----------------------------------------------------------------------------
# Running and testing
# ----------------------------------------------------------------------------
def run_agent(rows, verbose=True):
    """Build the environment and agent, plan, and execute. Returns the path."""
    warehouse = Warehouse(rows)
    agent = GoalBasedAgent(warehouse)

    if verbose:
        print("Start:", warehouse.start, " Goal:", warehouse.goal)
        print(warehouse.render())
        print()

    plan = agent.formulate_plan()
    if plan is None:
        if verbose:
            print("No path exists from S to G.")
            print("Cells expanded:", agent.cells_expanded)
        return warehouse, None

    actions = []
    for action, _ in plan:
        actions.append(action)
    path = [warehouse.start]
    for _, position in plan:
        path.append(position)

    if verbose:
        print("Path found with %d moves (cells expanded: %d)"
              % (len(actions), agent.cells_expanded))
        print("Moves:", " ".join(actions))
        print("Cells:", path)
        print()
        print(warehouse.render(path))

    # execute the plan step by step as the agent would in the world
    while agent.plan:
        agent.act()
    if verbose:
        print("\nAgent finished at", agent.state, "- goal reached:",
              agent.goal_test(agent.state))
    return warehouse, path


def validate_path(warehouse, path):
    """Independent checks that the returned path is a legal S -> G route."""
    assert path[0] == warehouse.start, "path must begin at S"
    assert path[-1] == warehouse.goal, "path must end at G"
    for position in path:
        assert warehouse.is_free(position), "path enters an obstacle: %s" % (position,)
    for a, b in zip(path[:-1], path[1:]):
        step = abs(a[0] - b[0]) + abs(a[1] - b[1])
        assert step == 1, "non-adjacent step %s -> %s" % (a, b)
    assert len(set(path)) == len(path), "path revisits a cell"


def run_tests():
    print("\n" + "=" * 60)
    print("TESTS")
    print("=" * 60)

    # 1. the lab map: path is valid
    warehouse, path = run_agent(WAREHOUSE_MAP, verbose=False)
    assert path is not None
    validate_path(warehouse, path)
    print("1. Lab map: valid path, %d moves ........ PASS" % (len(path) - 1))

    # 2. blocked goal: agent must report failure, not crash or loop
    blocked = [
        "#####",
        "#S#G#",
        "#####",
    ]
    _, path = run_agent(blocked, verbose=False)
    assert path is None
    print("2. Walled-off goal: reports no path ...... PASS")

    # 3. trivial map: start next to goal, shortest path is 1 move
    tiny = [
        "####",
        "#SG#",
        "####",
    ]
    warehouse, path = run_agent(tiny, verbose=False)
    validate_path(warehouse, path)
    assert len(path) - 1 == 1
    print("3. Adjacent start/goal: 1 move ........... PASS")

    # 4. open room: BFS must return the Manhattan-distance optimum
    room = [
        "#######",
        "#S....#",
        "#.....#",
        "#....G#",
        "#######",
    ]
    warehouse, path = run_agent(room, verbose=False)
    validate_path(warehouse, path)
    manhattan = abs(warehouse.start[0] - warehouse.goal[0]) + \
        abs(warehouse.start[1] - warehouse.goal[1])
    assert len(path) - 1 == manhattan
    print("4. Open room: optimal (%d = Manhattan) .... PASS" % manhattan)

    print("\nAll tests passed.")


def main():
    run_agent(WAREHOUSE_MAP)
    run_tests()


if __name__ == "__main__":
    main()
