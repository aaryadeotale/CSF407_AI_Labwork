% planner.pl
% Prolog logical verifier for the warehouse planning laboratory.
%
% The knowledge base contains the direct connections:
% A <-> B <-> C.
%
% Run with SWI-Prolog:
%   swipl planner.pl
%
% Example queries:
%   ?- can_move(a,b).
%   ?- can_move(a,c).
%   ?- valid_move(a,b).
%   ?- valid_move(b,c).
%   ?- valid_move(a,c).

% ---------------------------------------------------------------------------
% Warehouse facts
% ---------------------------------------------------------------------------

connected(a,b).
connected(b,a).
connected(b,c).
connected(c,b).

% ---------------------------------------------------------------------------
% Rules
% ---------------------------------------------------------------------------

can_move(X,Y) :-
    connected(X,Y).

valid_move(X,Y) :-
    connected(X,Y).

% ---------------------------------------------------------------------------
% Simple plan verification
% ---------------------------------------------------------------------------

valid_plan([]).

valid_plan([move(X,Y)|Rest]) :-
    valid_move(X,Y),
    valid_plan(Rest).

% Expected:
% ?- can_move(a,b).
% true.
%
% ?- can_move(a,c).
% false.
%
% ?- valid_plan([move(a,b), move(b,c)]).
% true.
%
% ?- valid_plan([move(a,c)]).
% false.

% ---------------------------------------------------------------------------
% Logical reasoning example from the laboratory
% ---------------------------------------------------------------------------

wet_road.
slippery :-
    wet_road.
reduce_speed :-
    slippery.

% Query:
% ?- reduce_speed.
%
% Reasoning:
% wet_road -> slippery -> reduce_speed
