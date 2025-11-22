# binarySKILL

**Binary Arithmetic Training with Circled Digit Notation (⓪①)**

Train LLMs to perform binary arithmetic using visually distinct ⓪① notation instead of standard 01. This helps models learn binary as a separate domain from decimal arithmetic.

## Why Circled Digits?

Using **⓪** and **①** instead of `0` and `1` makes binary operations visually distinctive:

```
Standard binary:  1011 + 0110 = 10001
Circled binary:   ①⓪①① + ⓪①①⓪ = ①⓪⓪⓪①
```

This visual separation helps LLMs:
- Distinguish binary from decimal operations
- Avoid confusion with standard arithmetic
- Learn binary-specific reasoning patterns

## Quick Start

```python
from toolkit.binary_arithmetic import add, subtract, bitwise_and

# Addition
result = add('①⓪①①', '⓪①①⓪')
print(result)
# {'result': '①⓪⓪⓪①', 'decimal': 17}

# Subtraction
result = subtract('①⓪①①', '⓪①①⓪')
print(result)
# {'result': '①⓪①', 'decimal': 5}

# Bitwise AND
result = bitwise_and('①⓪①①', '⓪①①⓪')
print(result)
# {'result': '⓪⓪①⓪', 'decimal': 2}
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
│   └── binary_arithmetic.py    # Arithmetic operations
├── generators/              # Training data generators
├── templates/               # Training templates
├── data/                    # Generated training data
├── tests/                   # Unit tests
├── sandbox/                 # Experiments
└── docs/                    # Documentation
```

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

### Example 1: Binary Addition

```python
from toolkit.binary_arithmetic import add

# Add 11 (①⓪①①) + 6 (⓪①①⓪)
result = add('①⓪①①', '⓪①①⓪')

print(f"Result: {result['result']}")  # ①⓪⓪⓪①
print(f"Decimal: {result['decimal']}")  # 17
```

### Example 2: Bitwise AND

```python
from toolkit.binary_arithmetic import bitwise_and

# Mask operation: 1011 & 0110
result = bitwise_and('①⓪①①', '⓪①①⓪')

print(f"Result: {result['result']}")  # ⓪⓪①⓪
print(f"Decimal: {result['decimal']}")  # 2
```

### Example 3: Left Shift (Multiply by 4)

```python
from toolkit.binary_arithmetic import left_shift

# Shift ①⓪①① left by 2 positions
result = left_shift('①⓪①①', 2)

print(f"Result: {result['result']}")  # ①⓪①①⓪⓪
print(f"Decimal: {result['decimal']}")  # 44
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
