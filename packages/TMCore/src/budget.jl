# The literal budget, L.
#
# Documented everywhere as "maximum number of literals in a clause". It is not, in either reference
# implementation. The check
#
#     if length(literals[j]) + length(literals_inverted[j]) <= L
#         # ... then increment EVERY automaton that agrees with the input
#
# runs *before* a pass that can push many automata across the include threshold at once. So L
# decides whether a clause may grow this round; it does not bound how large the clause becomes.
# Equilibrium sits wherever growth and forgetting balance, which is far above L: on Hnilov's
# published MNIST model L = 10 while clauses hold 12 to 54 literals, median about 18. Every clause
# in the model exceeds its own documented cap.
#
# That matters beyond bookkeeping, because arXiv:2508.08350 §2.1 offers `LF <= L` as hyperparameter
# guidance, which reads as a statement about capacity and is not one.
#
# Whether this is intended or accidental is a question for the author. Until it is answered, both
# readings are expressible and `GrowthGate` — the observed behaviour — is the default, so that
# reproducing published numbers does not depend on guessing.

"""
    LiteralBudgetPolicy

How the hyperparameter `L` constrains clause growth. See [`GrowthGate`](@ref) and [`HardCap`](@ref).
"""
abstract type LiteralBudgetPolicy end

"""
    GrowthGate()

What both reference implementations do: if a clause currently holds at most `L` literals it may
grow, and that round's growth is unbounded. Clause size routinely ends up several times `L`.
"""
struct GrowthGate <: LiteralBudgetPolicy end

"""
    HardCap()

`L` as it is documented: a clause never holds more than `L` literals. Automata are still reinforced,
but one is not allowed to *cross* the include threshold while the clause is already at the cap, so
the literal count is a genuine bound.

Not what the references do. Provided so the difference can be measured rather than argued about —
and measuring it says the references are right to do what they do.

**Measured cost.** On a small synthetic task (`y = x1 AND NOT x3`, 24 features, 16 clauses/class,
`L`=6, `LF`=4, 10 epochs, 5 seeds), `GrowthGate` reaches 0.985-0.999 while `HardCap` reaches
0.78-0.81 against a 0.75 majority-class baseline. Enforcing `L` as documented does not merely cost
accuracy; it comes close to preventing learning.

**Separated, and it is the rejection path that matters.** `HardCap` rations both growth paths,
because a cap that ignores one of them is not a cap. Isolating them on MNIST
(`research/budget-paths/`): adding a Type II cap to the reference policy and changing nothing else
costs **13.4 points**, while capping Type Ia instead costs 5.7.

The mechanism is not subtle. Type II exists to add literals until a clause stops matching
wrong-class examples. Cap it and clauses can never learn to reject, so they keep firing on the wrong
class, keep drawing Type Ia, and grow without bound — maximum clause size rises from 46 literals to
305, and to 774 when Type Ia is unrestrained too.

So `L` is a brake on **reinforcement** growth, and gating rejection by it is actively harmful rather
than merely unnecessary. The references are right not to check it there. What is wrong is the name
and the documentation: `L` is described as a maximum clause size, is not one, and cannot be made one
without a double-digit accuracy loss.
"""
struct HardCap <: LiteralBudgetPolicy end

# The two policies differ in shape, not just in degree, so they need two questions answered rather
# than one number. Under GrowthGate a shut gate suppresses the whole reinforcement block — including
# for literals that are *already* included, which would otherwise be pushed deeper into the include
# region and become harder to erode. Under HardCap reinforcement always happens; only crossing the
# include threshold is rationed. Collapsing these into a single "allowance" silently changes the
# reference behaviour, which is the sort of divergence that shows up as slightly worse accuracy and
# gets blamed on hyperparameters.

"""
    reinforce_allowed(policy, current_count, L) -> Bool

Whether Type Ia may reinforce automata at all this round.

`GrowthGate` answers no once the clause already holds more than `L` literals, freezing every
automaton in the clause until erosion brings it back under. `HardCap` always answers yes.
"""
@inline reinforce_allowed(::GrowthGate, n::Integer, L::Integer) = n <= L
@inline reinforce_allowed(::HardCap, ::Integer, ::Integer) = true

"""
    promotion_room(policy, current_count, L, rule) -> Int

How many further literals may cross the include threshold during one pass of `rule`.

The `rule` argument is not decoration. Clauses grow on **two** paths — Type Ia reinforcement and
Type II rejection — and the references gate neither of them by `L` in Type II. A policy that cannot
tell the two apart cannot express the reference behaviour and cannot separate the two effects when
they are measured, so the distinction belongs in the interface rather than in a comment.

`GrowthGate` rations neither path, which is why observed clause sizes run several times `L`.
`HardCap` rations both, returning exactly the headroom left.
"""
@inline promotion_room(::GrowthGate, ::Integer, ::Integer, ::FeedbackRule) = typemax(Int)
@inline promotion_room(::HardCap, n::Integer, L::Integer, ::FeedbackRule) = max(0, Int(L) - Int(n))
