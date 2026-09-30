# Logical Planning Lab — Complete Solution

This folder contains:

```text
planner_submission/
├── planner.py
├── planner.pl
└── README.md
```

The implementation follows the supplied Logical Planning laboratory. The lab defines planning as a combination of logical reasoning and search, with actions represented by preconditions and effects. fileciteturn2file0L19-L40

## 1. `planner.py`

`planner.py` implements the warehouse planning problem using:

- logical states represented as sets of propositions;
- actions with positive/negative preconditions;
- positive/negative effects;
- applicability checking;
- state transition/application;
- breadth-first search;
- plan reconstruction;
- independent plan verification.

The lab specifically asks for a Python planner using a set of logical propositions, action preconditions/effects, BFS, no-plan detection, and printing the resulting states. fileciteturn2file0L165-L184

Run:

```bash
python planner.py
```

No external Python package is required.

---

# 2. Planning Problem

## Initial state

```text
At(Robot,A)
At(Package,A)
```

or:

```python
{
    "At(Robot,A)",
    "At(Package,A)"
}
```

## Goal

```text
At(Package,C)
```

The warehouse has three locations:

```text
A — B — C
```

The robot begins at A and the package begins at A. The goal is to deliver the package to C. fileciteturn2file0L56-L66

---

# 3. Actions

## Move

For example:

```text
Move(A,B)
```

Precondition:

```text
At(Robot,A)
```

Effects:

```text
¬At(Robot,A)
At(Robot,B)
```

The other direct movements are also implemented:

```text
Move(B,A)
Move(B,C)
Move(C,B)
```

These correspond to the connections specified in the laboratory. fileciteturn2file0L67-L79

## PickUp

For example:

```text
PickUp(Package,B)
```

Preconditions:

```text
At(Robot,B)
At(Package,B)
```

Effects:

```text
¬At(Package,B)
Holding(Package)
```

This follows the pickup definition in the lab. fileciteturn2file0L80-L91

## Drop

For example:

```text
Drop(Package,C)
```

Preconditions:

```text
At(Robot,C)
Holding(Package)
```

Effects:

```text
¬Holding(Package)
At(Package,C)
```

This follows the drop action specified in the lab. fileciteturn2file0L92-L106

---

# 4. Expected shortest plan

The planner should find:

```text
1. Move(A,B)
2. PickUp(Package,B)
3. Move(B,C)
4. Drop(Package,C)
```

State sequence:

```text
S0:
At(Robot,A)
At(Package,A)

S1:
At(Robot,B)
At(Package,A)

S2:
At(Robot,B)
Holding(Package)

S3:
At(Robot,C)
Holding(Package)

S4:
At(Robot,C)
At(Package,C)
```

The final state satisfies:

```text
At(Package,C)
```

The laboratory gives the same action sequence as an example of a valid plan. fileciteturn2file0L129-L150

---

# 5. How the planner works

The planner follows:

```text
Current state
      ↓
Check action preconditions
      ↓
Apply applicable action
      ↓
Generate successor state
      ↓
Search over alternatives
      ↓
Goal?
```

The logical component determines whether an action can be executed:

```text
S |= Preconditions(action)
```

The search component determines which sequence of applicable actions should be explored.

The key idea from the laboratory is:

```text
Logic determines what is possible;
search determines what to try.
```

fileciteturn2file0L224-L260

---

# 6. BFS

Breadth-first search is used because the laboratory explicitly requests BFS.

The planner stores:

```text
queue
visited states
parent state
action used to reach each state
```

The `parent` information is used to reconstruct the final plan after the goal is discovered.

---

# 7. Tests implemented

## Test A — Solvable problem

The original warehouse problem is executed.

Expected:

```text
Plan found
```

The returned plan is independently replayed and checked.

This corresponds to the lab's Test A. fileciteturn2file0L192-L203

## Test B — Impossible problem

All pickup actions are removed.

Therefore:

```text
At(Package,A)
```

can never become:

```text
Holding(Package)
```

and consequently the package cannot reach C.

Expected:

```text
No plan found
```

This corresponds to Test B. fileciteturn2file0L204-L209

## Test C — Irrelevant actions

An additional robot movement action is introduced.

The planner must not confuse:

```text
At(Robot,C)
```

with:

```text
At(Package,C)
```

The goal is specifically about the package.

This corresponds to the laboratory's irrelevant-action test. fileciteturn2file0L210-L223

## Additional test — Goal already satisfied

If the initial state already contains:

```text
At(Package,C)
```

the correct plan is the empty sequence.

---

# 8. Independent verification

The program does not simply trust the plan produced by BFS.

`verify_plan()` starts from the initial state and independently:

1. checks every action's preconditions;
2. applies its effects;
3. compares the generated state with the planner's state sequence;
4. checks that the final state satisfies the goal.

This implements the laboratory's distinction between an LLM/generated candidate and independent verification. fileciteturn2file0L261-L280

---

# 9. `planner.pl`

`planner.pl` implements the Prolog portion of the laboratory.

It contains:

```prolog
connected(a,b).
connected(b,a).
connected(b,c).
connected(c,b).
```

and:

```prolog
can_move(X,Y) :-
    connected(X,Y).
```

It also contains:

```prolog
valid_move(X,Y) :-
    connected(X,Y).
```

and a simple plan verifier:

```prolog
valid_plan([]).

valid_plan([move(X,Y)|Rest]) :-
    valid_move(X,Y),
    valid_plan(Rest).
```

These correspond to the optional Prolog verifier described in the laboratory. fileciteturn2file0L354-L370

## Run Prolog

If SWI-Prolog is installed:

```bash
swipl planner.pl
```

Then try:

```prolog
?- can_move(a,b).
```

Expected:

```text
true.
```

Try:

```prolog
?- can_move(a,c).
```

Expected:

```text
false.
```

For the generated movement sequence:

```prolog
?- valid_plan([move(a,b), move(b,c)]).
```

Expected:

```text
true.
```

For an unsupported direct movement:

```prolog
?- valid_plan([move(a,c)]).
```

Expected:

```text
false.
```

The lab specifies that `a -> b` and `b -> c` should succeed while `a -> c` should fail for this knowledge base. fileciteturn2file0L383-L404

---

# 10. Subjective / Reflection Questions — Answers

## Q1. Why is it useful to specify action preconditions and effects before asking an LLM to write the planner?

Specifying preconditions and effects defines exactly when an action is legal and how it changes the state. This gives the LLM a precise problem specification and prevents it from inventing incorrect action behaviour. It also makes the generated code easier to inspect and test. The laboratory places this specification before code generation for this reason. fileciteturn2file0L107-L128

## Q2. Give an example of an error that could occur if the planner failed to check an action's preconditions.

The planner could execute:

```text
Drop(Package,C)
```

when the robot is still at A. It could then incorrectly place the package at C even though the robot never reached C and was not holding the package. Checking preconditions prevents such an invalid transition.

## Q3. Why is a plan that "looks reasonable" not necessarily a valid plan?

A plan is valid only if every action is applicable in the state where it is executed and the final state satisfies the goal. A sequence can look reasonable but contain an action whose preconditions are false. Therefore, the complete state transition sequence must be verified rather than judged by appearance. fileciteturn2file0L192-L223

## Q4. What did the LLM contribute to the implementation?

The LLM can help translate the specified planning model into Python code, suggest data structures such as queues and sets, and provide implementation boilerplate. The human still specifies the problem, checks the action definitions, executes tests, and validates the resulting planner.

## Q5. What did you have to verify independently?

The action preconditions and effects, applicability checks, state transitions, BFS exploration, plan reconstruction, no-solution behaviour, and final goal satisfaction all need independent verification. The Python implementation therefore replays the generated plan rather than relying only on an explanation.

## Q6. In this laboratory, where is logical reasoning being used?

Logical reasoning is used when the planner checks whether:

```text
S |= Preconditions(action)
```

If all required facts are true and no prohibited condition is present, the action is applicable. Its effects then determine the successor state. fileciteturn2file0L53-L55

## Q7. How is planning related to the search algorithms studied in the previous module?

Planning can be viewed as search over states. Each applicable action generates a successor state, and BFS explores alternative action sequences until it reaches a state satisfying the goal. Thus, logical reasoning determines which transitions are legal while search determines which sequence of legal transitions to explore. The laboratory summarises this as:

```text
Logic + Search = Planning
```

fileciteturn2file0L27-L28

---

# Optional Prolog Reflection Questions

## Q1. What is the difference between a Prolog fact and a Prolog rule?

A fact directly states something that is true, such as:

```prolog
connected(a,b).
```

A rule defines something that follows when another condition is true:

```prolog
can_move(X,Y) :-
    connected(X,Y).
```

The rule means that `can_move(X,Y)` follows whenever `connected(X,Y)` holds.

## Q2. How does a Prolog query correspond to asking whether something follows from a knowledge base?

A query asks Prolog whether the requested proposition can be established from the available facts and rules. For example:

```prolog
?- can_move(a,b).
```

succeeds because `connected(a,b)` is a fact and the `can_move` rule derives the requested proposition.

## Q3. Why might it be useful to use a Prolog program to verify a plan generated by a Python program?

Python can generate a candidate plan while Prolog independently checks whether proposed movements are supported by the logical knowledge base. This separates generation from verification and can reveal invalid actions in the candidate plan. The lab explicitly presents this as:

```text
Generate → Independent verification
```

fileciteturn2file0L396-L408

## Q4. What advantage does an independent verifier provide when the original plan was generated with the help of an LLM?

An independent verifier does not rely on the same generated reasoning to decide whether the result is valid. It checks the candidate against explicit facts and rules, reducing the risk of accepting a plausible-looking but logically invalid plan.

---

# Prolog Task 8

The supplied Prolog reasoning example is:

```prolog
wet_road.

slippery :-
    wet_road.

reduce_speed :-
    slippery.
```

Query:

```prolog
?- reduce_speed.
```

succeeds because:

```text
wet_road
    ⇒ slippery
    ⇒ reduce_speed
```

This follows the laboratory's requested `Fact ⇒ Rule ⇒ Rule ⇒ Conclusion` structure. fileciteturn2file0L413-L425

---

# Final takeaway

The complete implementation demonstrates:

```text
Logical representation
        +
Action applicability
        +
State transitions
        +
BFS search
        ↓
     Planning
```

The central workflow of the laboratory is:

```text
Understand → Specify → Generate → Execute → Verify
```

and the important engineering principle is that a generated explanation or plan is not the same thing as independent verification. fileciteturn2file0L307-L317 fileciteturn2file0L277-L280
