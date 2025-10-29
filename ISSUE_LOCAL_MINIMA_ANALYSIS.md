# Local Minima Issue: Evolution Gets Stuck Due to Population Allocation Problems

## Problem Description

The evolution algorithm frequently gets stuck in local minima, unable to make progress toward optimal solutions. For example, the XOR example gets stuck at approximately 198 fitness when the target is 400 (about 50% of the desired fitness).

## Root Cause Analysis

The issue is in the `population_alloc` function in `src/reproduce.rs` (lines 160-179). This function allocates population slots to species for the next generation based on their adjusted fitness. However, it has several critical flaws:

### Issue 1: Rounding Errors Don't Preserve Total Population Size

The current implementation:
```rust
(
    specie_repr,
    f64::round(population_f * fit_adjusted / fit_total) as usize,
)
```

When each species' allocation is rounded independently, the sum may not equal the target population. This can cause:
- **Under-allocation**: Total allocated < target population, wasting evolutionary potential
- **Over-allocation**: Total allocated > target population, creating more genomes than intended

**Example:**
```
Target population: 1000
Species A: 10.0 adjusted fitness → 556 allocated
Species B: 5.0 adjusted fitness → 278 allocated  
Species C: 3.0 adjusted fitness → 167 allocated
Total: 1001 (1 extra genome!)
```

### Issue 2: Negative Fitness Values Cause Catastrophic Allocation Failures

When a species has negative adjusted fitness (which can happen when individual genomes perform poorly), the calculation:
```rust
f64::round(population_f * fit_adjusted / fit_total) as usize
```

produces a negative number, which when cast to `usize` becomes 0 (or wraps to a huge number). This causes:
- **Species extinction**: Species with negative fitness get 0 members and disappear
- **Population explosion**: Other species get MORE than their fair share to compensate

**Example:**
```
Target population: 1000
Species D: -5.0 adjusted fitness → 0 allocated (should get some penalty, not extinction!)
Species E: 10.0 adjusted fitness → 2000 allocated (DOUBLE the target!)
```

### Issue 3: Zero Allocation Causes Premature Species Extinction

When a species is allocated 0 members, it disappears entirely. This is problematic because:
- **Diversity loss**: Even underperforming species may contain useful genetic material
- **Local minima**: Without diversity, the population can't escape local optima
- **Early convergence**: All genomes converge to a single species too quickly

This is visible in the XOR example output: `(of 1 species` appears consistently, showing all diversity has been lost.

### Issue 4: Fitness Sharing Calculation May Not Be Correct

The `fit_adjusted` function divides each genome's fitness by the species size:
```rust
pub fn fit_adjusted(&self) -> f64 {
    let l = self.len() as f64;
    self.members.iter().fold(0., |acc, (_, fit)| acc + *fit / l)
}
```

This is meant to implement fitness sharing (to prevent speciation from being too dominant), but dividing by length during the sum is equivalent to calculating the average fitness. This might not provide enough selective pressure for better-performing species.

## Impact

These issues combine to create a "death spiral":
1. Early in evolution, species with slightly different strategies emerge
2. Due to allocation issues, some species get 0 members and disappear
3. Diversity decreases, eventually converging to 1 species
4. Without diversity, the algorithm can't explore new solutions
5. Evolution stalls at a local optimum (e.g., 198/400 fitness for XOR)

## Reproduction

To reproduce this issue, run the XOR example:

```bash
cargo run --example xor --features approx
```

You will observe:
- Fitness quickly reaches ~198 and stagnates
- Species count drops to 1
- Evolution continues for thousands of generations with no progress

## Proposed Solution

The `population_alloc` function should be rewritten to:

1. **Ensure exact population allocation**: Use a proportional allocation algorithm that guarantees the sum equals the target
2. **Handle negative fitness gracefully**: Use fitness normalization or ensure minimum allocation
3. **Preserve species diversity**: Guarantee minimum allocation (e.g., at least 2-5 members) for viable species
4. **Consider elitism**: Always preserve the best performers regardless of species

### Recommended Algorithm: Largest Remainder Method

The Largest Remainder Method (also known as Hamilton's method) ensures:
- Total allocation exactly equals the target population
- Proportional representation based on fitness
- No negative allocations

```rust
fn population_alloc<'a, C: Connection + 'a, G: Genome<C> + 'a>(
    species: impl Iterator<Item = &'a Specie<C, G>>,
    population: usize,
) -> HashMap<SpecieRepr<C>, usize> {
    let species_fitted: Vec<_> = species
        .map(|s| (s.repr.clone(), s.fit_adjusted()))
        .collect();
    
    if species_fitted.is_empty() {
        return HashMap::new();
    }
    
    // Normalize fitness to be non-negative
    let min_fit = species_fitted.iter()
        .map(|(_, f)| *f)
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let offset = if min_fit < 0.0 { -min_fit + 1.0 } else { 0.0 };
    
    let species_normalized: Vec<_> = species_fitted
        .iter()
        .map(|(repr, fit)| (repr.clone(), fit + offset))
        .collect();
    
    let fit_total: f64 = species_normalized.iter().map(|(_, f)| *f).sum();
    
    if fit_total == 0.0 {
        // All species have equal fitness, distribute evenly
        let per_species = population / species_normalized.len();
        let remainder = population % species_normalized.len();
        
        return species_normalized
            .into_iter()
            .enumerate()
            .map(|(i, (repr, _))| {
                (repr, per_species + if i < remainder { 1 } else { 0 })
            })
            .collect();
    }
    
    // Largest Remainder Method for exact proportional allocation
    let mut allocations: Vec<(SpecieRepr<C>, f64)> = species_normalized
        .into_iter()
        .map(|(repr, fit)| {
            let exact_allocation = population as f64 * fit / fit_total;
            (repr, exact_allocation)
        })
        .collect();
    
    // Give each species the floor of their exact allocation
    let mut result: HashMap<SpecieRepr<C>, usize> = allocations
        .iter()
        .map(|(repr, exact)| (repr.clone(), exact.floor() as usize))
        .collect();
    
    // Calculate remainder to distribute
    let allocated: usize = result.values().sum();
    let mut remainder = population - allocated;
    
    // Sort by fractional part (descending) and distribute remainder
    allocations.sort_by(|(_, a), (_, b)| {
        let frac_a = a - a.floor();
        let frac_b = b - b.floor();
        frac_b.partial_cmp(&frac_a).unwrap()
    });
    
    for (repr, _) in allocations.iter() {
        if remainder > 0 {
            *result.get_mut(repr).unwrap() += 1;
            remainder -= 1;
        }
    }
    
    result
}
```

## Testing

The fix should be validated by:
1. Running the XOR example and verifying it reaches 400 fitness
2. Adding unit tests for `population_alloc` covering edge cases:
   - Normal case with positive fitness values
   - Case with negative fitness values
   - Case with very small fitness differences
   - Case with zero fitness values
3. Ensuring existing benchmarks don't regress significantly

## References

- XOR example: `examples/xor.rs`
- Population allocation: `src/reproduce.rs:160-179`
- Fitness sharing: `src/population.rs:93-96`
- NEAT algorithm: https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf
