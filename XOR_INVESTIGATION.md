# XOR Problem Investigation

## Summary
The XOR example does NOT solve XOR - it gets stuck at 198/400 fitness both before and after the speciation fixes. This is a **pre-existing issue**, not caused by the speciation changes.

## Root Cause
The algorithm is stuck at a local optimum caused by the fitness function design.

### The Local Optimum
- Genomes with 0 connections output 0.0 for all inputs
- Test cases [1,0] → 0 and [0,1] → 0 are CORRECT (expected output is 0)
- Test cases [0,0] → 1 and [1,1] → 1 are WRONG (output 0 instead of 1)
- Score: 2×100 (correct) - 2×1 (error) = **198 points**

### Why It Can't Escape
1. **Low mutation rate for new connections**: Only 5% chance per mutation event
2. **Fitness penalty for random connections**: 
   - Random weights likely make predictions worse initially
   - Outputs outside [-1, 2] get squared penalty (very harsh)
3. **No evolutionary pressure**: 198/400 is "good enough" that selection doesn't strongly favor exploration

### Verification
Tested both with original code (before speciation fixes) and current code:
```
Gen 0-500: best_fitness=198.00, species=1, connections=0
```

The genome stays at 0 connections forever.

## Why This Reveals a Deeper Problem

The XOR task, as implemented, has a fundamental issue:
```rust
eval_pair!([0., 0.], 1., ...);  // Expects 1 (this is XNOR, not XOR)
eval_pair!([1., 1.], 1., ...);  // Expects 1
eval_pair!([1., 0.], 0., ...);  // Expects 0  
eval_pair!([0., 1.], 0., ...);  // Expects 0
```

This is actually **XNOR** (NOT XOR). But more importantly, the fitness function allows a trivial "always output 0" solution to score 50% correct, creating a strong local optimum.

## Implications for Speciation Testing

The fact that XOR doesn't solve means:
1. We can't use it to verify speciation helps evolution
2. The "~200 fitness" observation in the original issue was actually this local optimum
3. We need a different test problem to validate speciation improvements

## Recommended Next Steps

1. **Fix the fitness function** to prevent trivial solutions:
   - Penalize outputs that are always the same
   - Require minimum network complexity
   - Use a different scoring function that doesn't reward partial correctness so heavily

2. **Use a different test problem** that:
   - Requires some network complexity
   - Doesn't have trivial local optima
   - Actually benefits from speciation

3. **Verify speciation on problems that grow complexity**, like:
   - Pole balancing
   - Pattern recognition tasks
   - Problems where the NEAT paper showed good results
