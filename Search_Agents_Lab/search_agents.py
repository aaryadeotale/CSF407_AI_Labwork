"""
Solved implementation for:
Artificial Intelligence Laboratory: Search and A*
Warehouse Robot Navigation

Includes:
- A* search with Manhattan / zero / Euclidean / scaled heuristics
- BFS comparison
- Path reconstruction
- State expansion counting
- Original, trivial, no-solution, and alternative-path tests
- Heuristic investigation
"""

from collections import deque
import heapq
import math


WAREHOUSE = [
    "#################",
    "#S....#.........#",
    "#.###.#.#######.#",
    "#...#.#.......#.#",
    "###.#.#######.#.#",
    "#...#.........#.#",
    "#.###########.#.#",
    "#.............#G#",
    "#################",
]

TRIVIAL = [
    "#####",
    "#SG##",
    "#####",
]

NO_SOLUTION = [
    "#######",
    "#S....#",
    "###.###",
    "#...#G#",
    "#######",
]

ALTERNATIVE = [
    "#######",
    "#S...G#",
    "#.....#",
    "#.....#",
    "#######",
]

MOVES = [
    (-1, 0),  # Up
    (1, 0),   # Down
    (0, -1),  # Left
    (0, 1),   # Right
]


def parse_grid(grid):
    """Return start and goal coordinates."""
    start = None
    goal = None

    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell == "S":
                start = (r, c)
            elif cell == "G":
                goal = (r, c)

    if start is None or goal is None:
        raise ValueError("Grid must contain both S and G.")

    return start, goal


def valid_neighbors(grid, state):
    """Generate valid states after Up/Down/Left/Right moves."""
    r, c = state

    for dr, dc in MOVES:
        nr, nc = r + dr, c + dc

        if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]):
            if grid[nr][nc] != "#":
                yield (nr, nc)


def reconstruct_path(came_from, start, goal):
    """Reconstruct path from goal back to start."""
    if goal not in came_from and goal != start:
        return None

    path = []
    current = goal

    while current != start:
        path.append(current)
        current = came_from[current]

    path.append(start)
    path.reverse()
    return path


def manhattan(state, goal):
    return abs(state[0] - goal[0]) + abs(state[1] - goal[1])


def euclidean(state, goal):
    return math.sqrt(
        (state[0] - goal[0]) ** 2 +
        (state[1] - goal[1]) ** 2
    )


def zero_heuristic(state, goal):
    return 0.0


def scaled_manhattan(state, goal):
    return 2.0 * manhattan(state, goal)


HEURISTICS = {
    "manhattan": manhattan,
    "zero": zero_heuristic,
    "euclidean": euclidean,
    "2x_manhattan": scaled_manhattan,
}


def astar(grid, heuristic=manhattan):
    """
    A* search.

    f(n) = g(n) + h(n)

    Returns a dictionary containing:
    - found
    - path
    - path_length
    - states_expanded
    """
    start, goal = parse_grid(grid)

    frontier = []
    counter = 0

    # Heap entries: (f, g, counter, state)
    heapq.heappush(
        frontier,
        (heuristic(start, goal), 0, counter, start)
    )

    came_from = {}
    g_cost = {start: 0}

    # Best-known expanded cost. A state may appear in the heap more than once
    # if a cheaper route to it is discovered later.
    expanded_g = {}

    states_expanded = 0

    while frontier:
        f, current_g, _, current = heapq.heappop(frontier)

        if current_g != g_cost.get(current, math.inf):
            continue

        if current in expanded_g and current_g >= expanded_g[current]:
            continue

        expanded_g[current] = current_g
        states_expanded += 1

        if current == goal:
            path = reconstruct_path(came_from, start, goal)
            return {
                "found": True,
                "path": path,
                "path_length": len(path) - 1,
                "states_expanded": states_expanded,
            }

        for neighbor in valid_neighbors(grid, current):
            tentative_g = current_g + 1

            if tentative_g < g_cost.get(neighbor, math.inf):
                g_cost[neighbor] = tentative_g
                came_from[neighbor] = current

                counter += 1
                h = heuristic(neighbor, goal)
                new_f = tentative_g + h

                heapq.heappush(
                    frontier,
                    (new_f, tentative_g, counter, neighbor)
                )

    return {
        "found": False,
        "path": None,
        "path_length": None,
        "states_expanded": states_expanded,
    }


def bfs(grid):
    """
    Breadth-first search for unit-cost movements.

    BFS is optimal here because every movement has cost 1.
    """
    start, goal = parse_grid(grid)

    frontier = deque([start])
    visited = {start}
    came_from = {}

    states_expanded = 0

    while frontier:
        current = frontier.popleft()
        states_expanded += 1

        if current == goal:
            path = reconstruct_path(came_from, start, goal)
            return {
                "found": True,
                "path": path,
                "path_length": len(path) - 1,
                "states_expanded": states_expanded,
            }

        for neighbor in valid_neighbors(grid, current):
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                frontier.append(neighbor)

    return {
        "found": False,
        "path": None,
        "path_length": None,
        "states_expanded": states_expanded,
    }


def mark_path(grid, path):
    """Return a printable grid with the solution path marked by *."""
    result = [list(row) for row in grid]

    if path:
        for r, c in path:
            if result[r][c] not in ("S", "G"):
                result[r][c] = "*"

    return "\n".join("".join(row) for row in result)


def print_result(name, result):
    print(f"\n{name}")
    print("-" * 60)
    print("Solution found :", result["found"])
    print("Path length    :", result["path_length"])
    print("States expanded:", result["states_expanded"])

    if result["path"] is not None:
        print("Path:")
        print(result["path"])


def run_basic_tests():
    print("=" * 72)
    print("SEARCH LAB — SOLVED IMPLEMENTATION")
    print("=" * 72)

    print("\nORIGINAL WAREHOUSE")
    print("-" * 60)
    print("\n".join(WAREHOUSE))

    result = astar(WAREHOUSE, manhattan)
    print_result("A* with Manhattan distance", result)

    if result["path"]:
        print("\nWarehouse with A* path:")
        print(mark_path(WAREHOUSE, result["path"]))

    print("\nTEST 2: TRIVIAL ONE-STEP CASE")
    trivial_result = astar(TRIVIAL, manhattan)
    print_result("Trivial A*", trivial_result)
    assert trivial_result["found"]
    assert trivial_result["path_length"] == 1

    print("\nTEST 3: NO-SOLUTION CASE")
    no_solution_result = astar(NO_SOLUTION, manhattan)
    print_result("No-solution A*", no_solution_result)
    assert not no_solution_result["found"]

    print("\nTEST 4: ALTERNATIVE PATHS")
    alternative_result = astar(ALTERNATIVE, manhattan)
    print_result("Alternative-path A*", alternative_result)
    assert alternative_result["found"]

    print("\nBasic tests passed.")


def compare_bfs_astar():
    print("\n" + "=" * 72)
    print("TASK 5: BFS VS A*")
    print("=" * 72)

    bfs_result = bfs(WAREHOUSE)
    astar_result = astar(WAREHOUSE, manhattan)

    print(f"\n{'Measure':<22} {'BFS':>15} {'A*':>15}")
    print("-" * 54)
    print(
        f"{'Solution found':<22} "
        f"{str(bfs_result['found']):>15} "
        f"{str(astar_result['found']):>15}"
    )
    print(
        f"{'Path length':<22} "
        f"{str(bfs_result['path_length']):>15} "
        f"{str(astar_result['path_length']):>15}"
    )
    print(
        f"{'States expanded':<22} "
        f"{str(bfs_result['states_expanded']):>15} "
        f"{str(astar_result['states_expanded']):>15}"
    )

    # Since every action costs 1, BFS and A* with an admissible Manhattan
    # heuristic should return an optimal path.
    assert bfs_result["found"] == astar_result["found"]
    assert bfs_result["path_length"] == astar_result["path_length"]


def heuristic_investigation():
    print("\n" + "=" * 72)
    print("TASK 6: HEURISTIC INVESTIGATION")
    print("=" * 72)

    print(
        f"\n{'Heuristic':<20} {'Found':>10} "
        f"{'Path length':>15} {'States expanded':>18}"
    )
    print("-" * 68)

    for name, heuristic in HEURISTICS.items():
        result = astar(WAREHOUSE, heuristic)
        print(
            f"{name:<20} "
            f"{str(result['found']):>10} "
            f"{str(result['path_length']):>15} "
            f"{str(result['states_expanded']):>18}"
        )


def explain_design():
    print("\n" + "=" * 72)
    print("SEARCH-PROBLEM FORMULATION")
    print("=" * 72)

    print(
        """
State S:
    A grid position (row, column) of the robot.

Actions A:
    Up, Down, Left, Right.

Transition T:
    Applying a valid action moves the robot to the adjacent free cell.

Initial state s0:
    The cell containing S.

Goal G:
    The cell containing G.

Cost c:
    Every movement costs 1.

State information:
    The robot's current row and column are sufficient.

Invalid action:
    An action is invalid if it leaves the grid or enters an obstacle (#).

Determinism:
    Yes. A state and valid action determine exactly one next state.

Solution:
    A sequence of valid moves that takes the robot from S to G.
"""
    )


def main():
    explain_design()
    run_basic_tests()
    compare_bfs_astar()
    heuristic_investigation()

    print("\n" + "=" * 72)
    print("ALL EXPERIMENTS COMPLETED")
    print("=" * 72)
    print(
        "\nNote: A* with 2x Manhattan is intentionally included as an "
        "experiment. Unlike Manhattan distance, multiplying the heuristic "
        "by 2 can make it non-admissible, so optimality is not guaranteed."
    )


if __name__ == "__main__":
    main()
