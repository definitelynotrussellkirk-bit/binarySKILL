# binarySKILL

**Binary Arithmetic Training with Circled Digits (⓪①) and Prefix Notation**

Train LLMs to perform binary arithmetic using:
1. **Circled digits** (⓪①) instead of standard 01
2. **Prefix notation** (English operation names) instead of infix symbols

## Why This Notation?

### Circled Digits

Using **⓪** and **①** instead of `0` and `1` makes binary visually distinct from decimal:

```
Standard:  1011
Circled:   ①⓪①①
```

### Prefix Notation

Using English operation names in prefix position makes operations unambiguous:

```
Infix:     ①⓪①① + ⓪①①⓪ = ①⓪⓪⓪①
Prefix:    add ①⓪①① ⓪①①⓪ = ①⓪⓪⓪①
```

**Benefits:**
- **No ambiguity** - No need for parentheses or precedence rules
- **Visually distinct** - Clearly separates binary from decimal arithmetic
- **Explicit operations** - Operation names are clear, not symbolic
- **Functional style** - Familiar to Lisp/functional programming
- **Easy to parse** - Straightforward expression evaluation

## Quick Start

```python
from toolkit.prefix_notation import add, subtract, and_op, parse_prefix_expression

# Addition with prefix notation
result = add('①⓪①①', '⓪①①⓪')
print(result['prefix'])
# add ①⓪①① ⓪①①⓪ = ①⓪⓪⓪①

# Subtraction
result = subtract('①⓪①①', '⓪①①⓪')
print(result['prefix'])
# subtract ①⓪①① ⓪①①⓪ = ①⓪①

# Bitwise AND
result = and_op('①⓪①①', '⓪①①⓪')
print(result['prefix'])
# and ①⓪①① ⓪①①⓪ = ⓪⓪①⓪

# Parse prefix expressions
result = parse_prefix_expression('add ①⓪①① ⓪①①⓪')
print(result)
# {'result': '①⓪⓪⓪①', 'decimal': 17, 'prefix': 'add ①⓪①① ⓪①①⓪ = ①⓪⓪⓪①'}
```

## Features

### Arithmetic Operations

- **Addition** - Binary addition with carry propagation
- **Subtraction** - Binary subtraction with borrow propagation
- **Multiplication** - *(Coming soon)*
- **Division** - *(Coming soon)*

### Bitwise Operations

- **AND** (`&`) - ① only if both bits are ①
- **OR** (`|`) - ① if either bit is ①
- **XOR** (`^`) - ① if bits differ
- **NOT** (`~`) - Flip all bits (one's complement)

### Shift Operations

- **Left Shift** (`<<`) - Multiply by 2^n
- **Right Shift** (`>>`) - Divide by 2^n (with bits lost tracking)

### Conversion Utilities

```python
from toolkit.binary_notation import (
    decimal_to_circled_binary,
    circled_binary_to_decimal
)

# Decimal to binary
binary = decimal_to_circled_binary(11)
# '①⓪①①'

# Binary to decimal
decimal = circled_binary_to_decimal('①⓪①①')
# 11
```

## Directory Structure

```
binarySKILL/
├── toolkit/                 # Core binary operations
│   ├── binary_notation.py      # ⓪① notation utilities
│   ├── binary_arithmetic.py    # Low-level arithmetic operations
│   ├── prefix_notation.py      # Prefix notation interface (recommended)
│   └── binary_ops.py           # Legacy operations (standard 01)
├── generators/              # Training data generators
├── templates/               # Training templates
├── data/                    # Generated training data
├── tests/                   # Unit tests
├── sandbox/                 # Experiments
└── docs/                    # Documentation
```

## Prefix Notation Syntax

All operations use **operation-first** syntax:

```
add ①⓪①① ⓪①①⓪ = ①⓪⓪⓪①
subtract ①⓪①① ⓪①①⓪ = ①⓪①
and ①⓪①① ⓪①①⓪ = ⓪⓪①⓪
or ①⓪①① ⓪①①⓪ = ①①①①
xor ①⓪①① ⓪①①⓪ = ①①⓪①
not ①⓪①① = ⓪①⓪⓪
leftshift ①⓪①① 2 = ①⓪①①⓪⓪
rightshift ①⓪①① 2 = ①⓪
```

**Advantages over infix:**
- No operator precedence rules needed
- No parentheses for grouping
- Unambiguous parsing
- Natural for functional composition

## Operation Reference

### Addition Truth Table

| Bit A | Bit B | Carry In | Sum | Carry Out |
|-------|-------|----------|-----|-----------|
| ⓪     | ⓪     | ⓪        | ⓪   | ⓪         |
| ⓪     | ⓪     | ①        | ①   | ⓪         |
| ⓪     | ①     | ⓪        | ①   | ⓪         |
| ⓪     | ①     | ①        | ⓪   | ①         |
| ①     | ⓪     | ⓪        | ①   | ⓪         |
| ①     | ⓪     | ①        | ⓪   | ①         |
| ①     | ①     | ⓪        | ⓪   | ①         |
| ①     | ①     | ①        | ①   | ①         |

### Bitwise Operations

| Operation | Description | Example |
|-----------|-------------|---------|
| `AND` | ① if both bits are ① | `①⓪①① & ⓪①①⓪ = ⓪⓪①⓪` |
| `OR` | ① if either bit is ① | `①⓪①① \| ⓪①①⓪ = ①①①①` |
| `XOR` | ① if bits differ | `①⓪①① ^ ⓪①①⓪ = ①①⓪①` |
| `NOT` | Flip all bits | `~①⓪①① = ⓪①⓪⓪` |

### Shift Operations

```python
# Left shift (multiply by 2^n)
left_shift('①⓪①①', 2)  # ①⓪①①⓪⓪ (44)

# Right shift (divide by 2^n)
right_shift('①⓪①①', 2)  # ①⓪ (2), bits lost: ①①
```

## Step-by-Step Mode

All arithmetic operations support step-by-step explanations:

```python
result = add('①⓪①①', '⓪①①⓪', show_steps=True)

# Returns steps showing:
# - Position
# - Bit A, Bit B
# - Carry in/out
# - Sum bit
```

## Examples

### Example 1: Binary Addition (Prefix Notation)

```python
from toolkit.prefix_notation import add

# Add 11 (①⓪①①) + 6 (⓪①①⓪)
result = add('①⓪①①', '⓪①①⓪')

print(result['prefix'])
# add ①⓪①① ⓪①①⓪ = ①⓪⓪⓪①

print(f"Decimal: {result['decimal']}")  # 17
```

### Example 2: Bitwise AND (Prefix Notation)

```python
from toolkit.prefix_notation import and_op

# Mask operation
result = and_op('①⓪①①', '⓪①①⓪')

print(result['prefix'])
# and ①⓪①① ⓪①①⓪ = ⓪⓪①⓪

print(f"Decimal: {result['decimal']}")  # 2
```

### Example 3: Left Shift (Prefix Notation)

```python
from toolkit.prefix_notation import leftshift

# Multiply by 4 (shift left by 2)
result = leftshift('①⓪①①', 2)

print(result['prefix'])
# leftshift ①⓪①① 2 = ①⓪①①⓪⓪

print(f"Decimal: {result['decimal']}")  # 44
```

### Example 4: Parsing Prefix Expressions

```python
from toolkit.prefix_notation import parse_prefix_expression

# Parse and evaluate a prefix expression
result = parse_prefix_expression('xor ①⓪①① ⓪①①⓪')

print(result['prefix'])
# xor ①⓪①① ⓪①①⓪ = ①①⓪①
```

## Training Data Generation

*(Coming soon - generators for creating LLM training data)*

## Testing

```bash
cd /home/russ/binarySKILL
PYTHONPATH=. python3 toolkit/binary_arithmetic.py
```

## Notation Reference

| Circled | Standard | Decimal |
|---------|----------|---------|
| ⓪       | 0        | 0       |
| ①       | 1        | 1       |
| ①⓪      | 10       | 2       |
| ①①      | 11       | 3       |
| ①⓪⓪     | 100      | 4       |
| ①⓪①     | 101      | 5       |
| ①①⓪     | 110      | 6       |
| ①①①     | 111      | 7       |
| ①⓪⓪⓪    | 1000     | 8       |

## License

MIT License - See LICENSE file for details

## Contributing

This is a research project for LLM training data generation. Contributions welcome!

## Citation

```bibtex
@software{binarySKILL2025,
  title={binarySKILL: Binary Arithmetic Training with Circled Notation},
  author={Your Name},
  year={2025},
  url={https://github.com/definitelynotrussellkirk-bit/binarySKILL}
}
```
