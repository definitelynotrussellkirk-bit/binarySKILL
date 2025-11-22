# binarySKILL

**Universal Binary and Combinatorial Generator for Training LLMs**

This repository provides maximally general generators for binary operations and combinatorial objects, with a focus on teaching counting principles through bijections.

## Core Philosophy

**Binary ↔ Subsets ↔ Counting**

Every C(n,k) object has multiple equivalent representations:
- Binary strings with k ones
- k-subsets of [n]
- Lattice paths
- Committee selections
- Stars-and-bars distributions

All generators are **deterministic** (seed-based) and **bijectively equivalent**.

## Quick Start

```python
from generators.binomial_generator import generate_binomial_object_full

# Generate C(8,3) object with all representations
obj = generate_binomial_object_full(n=8, k=3, seed=42)

print(obj.indices)              # [0, 1, 5]
print(obj.to_binary())          # '11000100'
print(obj.to_lattice_path())    # 'RRUUURUU'
print(obj.to_set_notation())    # '{0, 1, 5}'
print(obj.count)                # 56
```

## Fundamental Counting Principles

```python
from generators.binomial_generator import count_OR, count_AND, explain_counting_principle

# OR principle (disjoint union) → ADD
total = count_OR(10, 15, 5)  # 30

# AND principle (cartesian product) → MULTIPLY
total = count_AND(3, 4, 5)   # 60

# Natural language explanation
explanation = explain_counting_principle(
    "AND",
    [3, 4],
    ["shirt colors", "pant sizes"]
)
# Output: "We choose shirt colors AND pant sizes.
#          Since these are independent choices, we MULTIPLY: 3 × 4 = 12"
```

## Directory Structure

```
binarySKILL/
├── generators/          # Core object generators
│   ├── binomial_generator.py    # Universal C(n,k) generator
│   ├── subsets.py               # k-subset generation
│   ├── words.py                 # Binary strings, Dyck words
│   ├── permutations.py          # Permutation generation
│   ├── distributions.py         # Stars and bars
│   └── functions.py             # Function counting
├── toolkit/             # Shared utilities
│   └── combinatorial_toolkit.py # Counting formulas (binomial, factorial, etc.)
├── templates/           # Training data templates
├── data/                # Generated training data
├── tests/               # Unit tests
├── sandbox/             # Throwaway experiments
└── docs/                # Documentation
```

## Output Formats

The `BinomialObject` class supports 8 output formats:

| Format | Example | Use Case |
|--------|---------|----------|
| `indices` | `[0, 2, 5, 7]` | Raw k-subset |
| `binary` | `"10100101"` | Binary string with k ones |
| `lattice_path` | `"RURUURUU"` | Grid paths (R=right, U=up) |
| `named_selection` | `["Alice", "Carol", "Eve"]` | Committee problems |
| `distribution` | `[2, 3, 1, 0, 4]` | Stars and bars |
| `set_notation` | `"{0, 2, 5, 7}"` | Mathematical notation |
| `indicator` | `[1,0,1,0,0,1,0,1]` | Characteristic vector |
| `positions` | `"positions 0, 2, 5, 7"` | Natural language |

## Context-Specific Generators

```python
# Committee selection
from generators.binomial_generator import generate_committee

result = generate_committee(
    total_people=5,
    committee_size=3,
    seed=42,
    people_names=["Alice", "Bob", "Carol", "Dave", "Eve"]
)
# Returns: {
#   "committee": ["Alice", "Carol", "Eve"],
#   "not_selected": ["Bob", "Dave"],
#   "total_ways": 10
# }

# Lattice paths
from generators.binomial_generator import generate_lattice_path_problem

result = generate_lattice_path_problem(
    right_steps=4,
    up_steps=3,
    seed=42
)
# Returns: {
#   "path": "RURUURR",
#   "start": (0, 0),
#   "end": (4, 3),
#   "total_paths": 35
# }

# Stars and bars
from generators.binomial_generator import generate_stars_and_bars

result = generate_stars_and_bars(
    num_items=10,
    num_bins=4,
    seed=42
)
# Returns: {
#   "distribution": [0, 0, 8, 2],
#   "num_items": 10,
#   "num_bins": 4,
#   "total_ways": 286
# }
```

## Design Principles

### 1. Infrastructure vs Templates

**Low-level generators** (general-purpose):
- `subsets.py` - Pure subset generation
- `words.py` - Pure string generation
- Reusable across multiple pedagogical contexts

**High-level templates** (specialized pedagogy):
- Committee selection
- Lattice path counting
- Pascal's triangle exploration
- Each template teaches a specific application of C(n,k)

### 2. Determinism

All generators respect `--seed` for reproducibility:
```python
obj1 = generate_binomial_object_full(10, 4, seed=42)
obj2 = generate_binomial_object_full(10, 4, seed=42)
assert obj1.indices == obj2.indices  # ✓ Deterministic
```

### 3. Bijective Equivalence

All representations are interconvertible:
```python
# Start with subset
obj = generate_binomial_object_full(10, 4, seed=42)

# Convert to binary
binary = obj.to_binary()

# Convert back
from generators.words import binary_string_to_subset
recovered = binary_string_to_subset(binary)

assert recovered == obj.indices  # ✓ Bijection preserved
```

## Examples

### Example 1: Binary Strings ↔ Subsets

```python
from generators.binomial_generator import generate_binomial_object_full

obj = generate_binomial_object_full(n=5, k=3, seed=42)

print(f"Subset:  {obj.indices}")           # [0, 1, 4]
print(f"Binary:  {obj.to_binary()}")       # '11001'
print(f"Path:    {obj.to_lattice_path()}") # 'RRUUU'
print(f"Set:     {obj.to_set_notation()}") # '{0, 1, 4}'
print(f"Count:   {obj.count}")             # 10
```

### Example 2: Counting with AND/OR

```python
from generators.binomial_generator import count_OR, count_AND

# Counting outfit combinations (AND)
shirts = 5
pants = 3
outfits = count_AND(shirts, pants)  # 15

# Counting total items (OR)
red_balls = 3
blue_balls = 7
total_balls = count_OR(red_balls, blue_balls)  # 10
```

### Example 3: Batch Generation

```python
from generators.binomial_generator import generate_binomial_batch

# Generate 100 different C(10,4) objects
batch = generate_binomial_batch(
    n=10,
    k=4,
    count=100,
    seed=42,
    output_format="binary"
)
# Returns list of 100 binary strings
```

## Testing

Run tests with:
```bash
python -m pytest tests/
```

## License

MIT License - See LICENSE file for details

## Contributing

This is a research project for LLM training data generation. Contributions welcome!

## Citation

If you use this in your research, please cite:

```bibtex
@software{binarySKILL2025,
  title={binarySKILL: Universal Binary and Combinatorial Generator},
  author={Your Name},
  year={2025},
  url={https://github.com/definitelynotrussellkirk-bit/binarySKILL}
}
```
