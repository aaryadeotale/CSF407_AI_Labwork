"""
Solved Logical Planning Laboratory
Warehouse Robot Planning using BFS.

Problem:
    Locations: A, B, C
    Initial state:
        At(Robot, A)
        At(Package, A)
    Goal:
        At(Package, C)

The planner represents states as sets of logical propositions and actions
with positive/negative preconditions and positive/negative effects.
"""

from collections import deque
from dataclasses import dataclass
from typing import FrozenSet, Iterable, Optional


# ---------------------------------------------------------------------------
# Action representation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Action:
    name: str
    positive_preconditions: FrozenSet[str] = frozenset()
    negative_preconditions: FrozenSet[str] = frozenset()
    add_effects: FrozenSet[str] = frozenset()
    delete_effects: FrozenSet[str] = frozenset()

    def is_applicable(self, state: FrozenSet[str]) -> bool:
        """An action is applicable iff all its preconditions hold."""
        return (
            self.positive_preconditions <= state
            and self.negative_preconditions.isdisjoint(state)
        )

    def apply(self, state: FrozenSet[str]) -> FrozenSet[str]:
        """Apply delete effects, then add effects."""
        if not self.is_applicable(state):
            raise ValueError(f"Action is not applicable: {self.name}")

        new_state = set(state)
        new_state.difference_update(self.delete_effects)
        new_state.update(self.add_effects)
        return frozenset(new_state)


# ---------------------------------------------------------------------------
# Warehouse domain
# ---------------------------------------------------------------------------

def move_action(source: str, destination: str) -> Action:
    return Action(
        name=f"Move({source},{destination})",
        positive_preconditions=frozenset({f"At(Robot,{source})"}),
        add_effects=frozenset({f"At(Robot,{destination})"}),
        delete_effects=frozenset({f"At(Robot,{source})"}),
    )


def pickup_action(location: str) -> Action:
    return Action(
        name=f"PickUp(Package,{location})",
        positive_preconditions=frozenset({
            f"At(Robot,{location})",
            f"At(Package,{location})",
        }),
        add_effects=frozenset({"Holding(Package)"}),
        delete_effects=frozenset({f"At(Package,{location})"}),
    )


def drop_action(location: str) -> Action:
    return Action(
        name=f"Drop(Package,{location})",
        positive_preconditions=frozenset({
            f"At(Robot,{location})",
            "Holding(Package)",
        }),
        add_effects=frozenset({f"At(Package,{location})"}),
        delete_effects=frozenset({"Holding(Package)"}),
    )


def make_actions(include_pickup: bool = True) -> list[Action]:
    actions = [
        move_action("A", "B"),
        move_action("B", "A"),
        move_action("B", "C"),
        move_action("C", "B"),
    ]

    if include_pickup:
        actions.append(pickup_action("A"))
        actions.append(pickup_action("B"))
        actions.append(pickup_action("C"))

    actions.extend([
        drop_action("A"),
        drop_action("B"),
        drop_action("C"),
    ])

    return actions


INITIAL_STATE = frozenset({
    "At(Robot,A)",
    "At(Package,A)",
})

GOAL = frozenset({
    "At(Package,C)",
})


# ---------------------------------------------------------------------------
# BFS planner
# ---------------------------------------------------------------------------

def goal_satisfied(state: FrozenSet[str], goal: FrozenSet[str]) -> bool:
    return goal <= state


def bfs_plan(
    initial_state: FrozenSet[str],
    goal: FrozenSet[str],
    actions: Iterable[Action],
):
    """
    Breadth-first search over logical states.

    Returns:
        None if no plan exists.

        Otherwise:
        {
            "actions": [...],
            "states": [initial_state, state_after_action_1, ...]
        }
    """
    actions = list(actions)

    queue = deque([initial_state])
    visited = {initial_state}

    # State -> (previous state, action used)
    parent = {}

    if goal_satisfied(initial_state, goal):
        return {
            "actions": [],
            "states": [initial_state],
        }

    while queue:
        current = queue.popleft()

        for action in actions:
            if not action.is_applicable(current):
                continue

            successor = action.apply(current)

            if successor in visited:
                continue

            visited.add(successor)
            parent[successor] = (current, action)
            queue.append(successor)

            if goal_satisfied(successor, goal):
                return reconstruct_plan(
                    initial_state, successor, parent
                )

    return None


def reconstruct_plan(initial_state, goal_state, parent):
    actions = []
    states = [goal_state]

    current = goal_state

    while current != initial_state:
        previous, action = parent[current]
        actions.append(action)
        current = previous
        states.append(current)

    actions.reverse()
    states.reverse()

    return {
        "actions": actions,
        "states": states,
    }


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------

def verify_plan(initial_state, goal, plan) -> bool:
    """Independently replay every action and verify all preconditions."""
    if plan is None:
        return False

    current = initial_state

    if plan["states"][0] != initial_state:
        return False

    for index, action in enumerate(plan["actions"], start=1):
        if not action.is_applicable(current):
            return False

        current = action.apply(current)

        if current != plan["states"][index]:
            return False

    return goal_satisfied(current, goal)


def format_state(state):
    return ", ".join(sorted(state))


def print_plan(plan):
    if plan is None:
        print("No plan found")
        return

    print("Plan found:")
    for i, action in enumerate(plan["actions"], start=1):
        print(f"  {i}. {action.name}")

    print("\nStates:")
    for i, state in enumerate(plan["states"]):
        print(f"  S{i}: {format_state(state)}")


def run_case(name, initial, goal, actions):
    print("\n" + "=" * 72)
    print(name)
    print("=" * 72)

    print("Initial:", format_state(initial))
    print("Goal:   ", format_state(goal))

    plan = bfs_plan(initial, goal, actions)

    if plan is None:
        print("\nNo plan found.")
        return None

    print()
    print_plan(plan)

    valid = verify_plan(initial, goal, plan)
    print("\nIndependent plan verification:", valid)

    if not valid:
        raise AssertionError("Planner returned an invalid plan.")

    return plan


# ---------------------------------------------------------------------------
# Tests required by the laboratory
# ---------------------------------------------------------------------------

def test_a_solvable_problem():
    """Original warehouse problem."""
    actions = make_actions(include_pickup=True)

    plan = run_case(
        "TEST A — SOLVABLE WAREHOUSE",
        INITIAL_STATE,
        GOAL,
        actions,
    )

    assert plan is not None
    assert verify_plan(INITIAL_STATE, GOAL, plan)
    assert "At(Package,C)" in plan["states"][-1]

    # A shortest valid plan is:
    # Move(A,B), PickUp(Package,B), Move(B,C), Drop(Package,C)
    assert len(plan["actions"]) == 4


def test_b_impossible_problem():
    """Remove all pickup actions, making the goal unreachable."""
    actions = [
        move_action("A", "B"),
        move_action("B", "A"),
        move_action("B", "C"),
        move_action("C", "B"),
        drop_action("A"),
        drop_action("B"),
        drop_action("C"),
    ]

    plan = run_case(
        "TEST B — IMPOSSIBLE PROBLEM (NO PICKUP)",
        INITIAL_STATE,
        GOAL,
        actions,
    )

    assert plan is None


def test_c_irrelevant_actions():
    """
    Add robot movement actions. The planner must not confuse the robot being
    at C with the package being at C.
    """
    actions = make_actions(include_pickup=True)

    # Extra movement that does not transport the package.
    actions.append(
        Action(
            name="IrrelevantMove(A,C)",
            positive_preconditions=frozenset({"At(Robot,A)"}),
            add_effects=frozenset({"At(Robot,C)"}),
            delete_effects=frozenset({"At(Robot,A)"}),
        )
    )

    plan = run_case(
        "TEST C — IRRELEVANT ACTIONS",
        INITIAL_STATE,
        GOAL,
        actions,
    )

    assert plan is not None
    assert verify_plan(INITIAL_STATE, GOAL, plan)
    assert "At(Package,C)" in plan["states"][-1]


def test_initial_goal():
    """If the goal already holds, the empty plan is correct."""
    initial = frozenset({"At(Robot,C)", "At(Package,C)"})
    goal = frozenset({"At(Package,C)"})

    plan = bfs_plan(initial, goal, make_actions())

    assert plan is not None
    assert plan["actions"] == []


def main():
    print("=" * 72)
    print("LOGICAL PLANNING LAB — SOLVED IMPLEMENTATION")
    print("=" * 72)

    print("\nPLANNING PROBLEM")
    print("Initial state:")
    print("  {At(Robot,A), At(Package,A)}")
    print("Goal:")
    print("  {At(Package,C)}")

    test_a_solvable_problem()
    test_b_impossible_problem()
    test_c_irrelevant_actions()
    test_initial_goal()

    print("\n" + "=" * 72)
    print("ALL TESTS PASSED")
    print("=" * 72)


if __name__ == "__main__":
    main()
