% ASP-DT minimal reference implementation of the manuscript Method.
% max_depth counts LEVELS: root level 1; max nodes = 2**max_depth - 1.
% It does not count edges as scikit-learn's max_depth does.
#const max_depth=3.
#const size_tiebreak=1.
#const root_feature=0.
#const banned_feature=0.
#const forbid_below=1.
node(1..2**max_depth-1).
depth(1,1).
left_child(N,2*N) :- node(N), node(2*N).
right_child(N,2*N+1) :- node(N), node(2*N+1).
depth(C,D+1) :- depth(N,D), left_child(N,C).
depth(C,D+1) :- depth(N,D), right_child(N,C).

% Complete the schematic rules with an active root and two children per split.
activated(1).
{ leaf(N) } :- activated(N).
activated(C) :- activated(N), not leaf(N), left_child(N,C).
activated(C) :- activated(N), not leaf(N), right_child(N,C).
:- activated(N), depth(N,max_depth), not leaf(N).
1 { split(N,F,T) : threshold(F,T), feature(F) } 1 :- activated(N), not leaf(N).
at(S,1) :- sample(S).
at(S,C) :- at(S,N), split(N,F,T), value(S,F,V), V <= T, left_child(N,C).
at(S,C) :- at(S,N), split(N,F,T), value(S,F,V), V > T, right_child(N,C).
final_leaf(S,N) :- at(S,N), leaf(N).
:- sample(S), not 1 { final_leaf(S,N) : node(N) } 1.
1 { predict(N,C) : class_label(C) } 1 :- leaf(N).
pred(S,C) :- final_leaf(S,N), predict(N,C).
misclassified(S) :- pred(S,C), true_class(S,Y), C != Y.
:~ misclassified(S). [1@1,S]
:~ activated(N), size_tiebreak=1. [1@0,N]

% Optional policy constraints; 0 disables each feature constraint.
:- root_feature>0, not split(1,root_feature,_).
:- banned_feature>0, split(N,banned_feature,_), depth(N,D), D>forbid_below.
#show activated/1.
#show depth/2.
#show leaf/1.
#show split/3.
#show predict/2.
#show misclassified/1.
