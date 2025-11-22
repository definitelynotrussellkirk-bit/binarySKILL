#!/usr/bin/env python3
"""
Binary Arithmetic with Circled Notation (⓪①)

All operations use ⓪ and ① for binary representation.
This makes binary math visually distinct from decimal arithmetic.
"""

from typing import Dict, List, Optional
from toolkit.binary_notation import (
    ZERO, ONE,
    to_binary_notation,
    from_binary_notation,
    decimal_to_circled_binary,
    circled_binary_to_decimal,
    validate_binary_notation
)


def normalize(binary: str) -> str:
    """Remove leading zeros (except preserve single ⓪)."""
    normalized = binary.lstrip(ZERO) or ZERO
    return normalized


def pad_to_same_length(a: str, b: str) -> tuple[str, str]:
    """Pad both binary numbers to same length with leading ⓪."""
    max_len = max(len(a), len(b))
    a = ZERO * (max_len - len(a)) + a
    b = ZERO * (max_len - len(b)) + b
    return a, b


def add(a: str, b: str, show_steps: bool = False) -> Dict:
    """
    Add two binary numbers in ⓪① notation.

    Args:
        a: First binary number (e.g., '①⓪①①')
        b: Second binary number (e.g., '⓪①①⓪')
        show_steps: Include step-by-step carry propagation

    Returns:
        Dict with 'result', 'decimal', optional 'steps'

    Examples:
        >>> add('①⓪①①', '⓪①①⓪')
        {'result': '①⓪⓪⓪①', 'decimal': 17}
    """
    if not (validate_binary_notation(a) and validate_binary_notation(b)):
        raise ValueError("Input must use only ⓪ and ① characters")

    # Pad to same length
    a, b = pad_to_same_length(a, b)
    max_len = len(a)

    result = []
    carry = ZERO
    steps = [] if show_steps else None

    # Add from right to left
    for i in range(max_len - 1, -1, -1):
        bit_a = a[i]
        bit_b = b[i]

        # Truth table for binary addition
        # bit_a | bit_b | carry_in | sum | carry_out
        # ⓪     | ⓪     | ⓪        | ⓪   | ⓪
        # ⓪     | ⓪     | ①        | ①   | ⓪
        # ⓪     | ①     | ⓪        | ①   | ⓪
        # ⓪     | ①     | ①        | ⓪   | ①
        # ①     | ⓪     | ⓪        | ①   | ⓪
        # ①     | ⓪     | ①        | ⓪   | ①
        # ①     | ①     | ⓪        | ⓪   | ①
        # ①     | ①     | ①        | ①   | ①

        ones_count = [bit_a, bit_b, carry].count(ONE)

        if ones_count == 0:
            sum_bit, new_carry = ZERO, ZERO
        elif ones_count == 1:
            sum_bit, new_carry = ONE, ZERO
        elif ones_count == 2:
            sum_bit, new_carry = ZERO, ONE
        else:  # ones_count == 3
            sum_bit, new_carry = ONE, ONE

        result.insert(0, sum_bit)

        if show_steps:
            steps.append({
                'position': max_len - 1 - i,
                'bit_a': bit_a,
                'bit_b': bit_b,
                'carry_in': carry,
                'sum_bit': sum_bit,
                'carry_out': new_carry
            })

        carry = new_carry

    # Add final carry if needed
    if carry == ONE:
        result.insert(0, ONE)

    result_str = ''.join(result)
    decimal = circled_binary_to_decimal(result_str)

    output = {'result': result_str, 'decimal': decimal}
    if show_steps:
        output['steps'] = steps

    return output


def subtract(a: str, b: str, show_steps: bool = False) -> Dict:
    """
    Subtract binary numbers (a - b) in ⓪① notation.

    Args:
        a: Minuend
        b: Subtrahend
        show_steps: Include step-by-step borrow propagation

    Returns:
        Dict with 'result', 'decimal', optional 'steps'

    Raises:
        ValueError: If b > a

    Examples:
        >>> subtract('①⓪①①', '⓪①①⓪')
        {'result': '①⓪①', 'decimal': 5}
    """
    if not (validate_binary_notation(a) and validate_binary_notation(b)):
        raise ValueError("Input must use only ⓪ and ① characters")

    dec_a = circled_binary_to_decimal(a)
    dec_b = circled_binary_to_decimal(b)

    if dec_b > dec_a:
        raise ValueError(f"Cannot subtract {b} ({dec_b}) from {a} ({dec_a})")

    # Pad to same length
    a, b = pad_to_same_length(a, b)
    max_len = len(a)

    result = []
    borrow = ZERO
    steps = [] if show_steps else None

    # Subtract from right to left
    for i in range(max_len - 1, -1, -1):
        bit_a = a[i]
        bit_b = b[i]

        # Apply previous borrow
        if borrow == ONE:
            if bit_a == ZERO:
                bit_a_adjusted = ONE
                borrow = ONE
            else:  # bit_a == ONE
                bit_a_adjusted = ZERO
                borrow = ZERO
        else:
            bit_a_adjusted = bit_a

        # Perform subtraction
        if bit_a_adjusted == ONE and bit_b == ZERO:
            diff_bit = ONE
            new_borrow = ZERO
        elif bit_a_adjusted == ZERO and bit_b == ZERO:
            diff_bit = ZERO
            new_borrow = ZERO
        elif bit_a_adjusted == ONE and bit_b == ONE:
            diff_bit = ZERO
            new_borrow = ZERO
        else:  # bit_a_adjusted == ZERO and bit_b == ONE
            diff_bit = ONE
            new_borrow = ONE

        result.insert(0, diff_bit)

        if show_steps:
            steps.append({
                'position': max_len - 1 - i,
                'bit_a': bit_a,
                'bit_b': bit_b,
                'borrow_in': borrow,
                'diff_bit': diff_bit,
                'borrow_out': new_borrow
            })

        borrow = new_borrow

    result_str = normalize(''.join(result))
    decimal = circled_binary_to_decimal(result_str)

    output = {'result': result_str, 'decimal': decimal}
    if show_steps:
        output['steps'] = steps

    return output


def bitwise_and(a: str, b: str) -> Dict:
    """
    Bitwise AND: ① only if both bits are ①.

    Examples:
        >>> bitwise_and('①⓪①①', '⓪①①⓪')
        {'result': '⓪⓪①⓪', 'decimal': 2}
    """
    if not (validate_binary_notation(a) and validate_binary_notation(b)):
        raise ValueError("Input must use only ⓪ and ① characters")

    a, b = pad_to_same_length(a, b)
    result = ''.join(ONE if a[i] == ONE and b[i] == ONE else ZERO for i in range(len(a)))

    return {'result': result, 'decimal': circled_binary_to_decimal(result)}


def bitwise_or(a: str, b: str) -> Dict:
    """
    Bitwise OR: ① if either bit is ①.

    Examples:
        >>> bitwise_or('①⓪①①', '⓪①①⓪')
        {'result': '①①①①', 'decimal': 15}
    """
    if not (validate_binary_notation(a) and validate_binary_notation(b)):
        raise ValueError("Input must use only ⓪ and ① characters")

    a, b = pad_to_same_length(a, b)
    result = ''.join(ONE if a[i] == ONE or b[i] == ONE else ZERO for i in range(len(a)))

    return {'result': result, 'decimal': circled_binary_to_decimal(result)}


def bitwise_xor(a: str, b: str) -> Dict:
    """
    Bitwise XOR: ① if bits differ.

    Examples:
        >>> bitwise_xor('①⓪①①', '⓪①①⓪')
        {'result': '①①⓪①', 'decimal': 13}
    """
    if not (validate_binary_notation(a) and validate_binary_notation(b)):
        raise ValueError("Input must use only ⓪ and ① characters")

    a, b = pad_to_same_length(a, b)
    result = ''.join(ONE if a[i] != b[i] else ZERO for i in range(len(a)))

    return {'result': result, 'decimal': circled_binary_to_decimal(result)}


def bitwise_not(a: str, width: Optional[int] = None) -> Dict:
    """
    Bitwise NOT: Flip all bits.

    Args:
        a: Binary number
        width: Optional bit width (default: length of a)

    Examples:
        >>> bitwise_not('①⓪①①')
        {'result': '⓪①⓪⓪', 'decimal': 4}
        >>> bitwise_not('①⓪①①', width=8)
        {'result': '①①①①⓪①⓪⓪', 'decimal': 244}
    """
    if not validate_binary_notation(a):
        raise ValueError("Input must use only ⓪ and ① characters")

    if width is None:
        width = len(a)

    a = ZERO * (width - len(a)) + a
    result = ''.join(ZERO if bit == ONE else ONE for bit in a)

    return {'result': result, 'decimal': circled_binary_to_decimal(result)}


def left_shift(a: str, n: int) -> Dict:
    """
    Left shift: Multiply by 2^n.

    Examples:
        >>> left_shift('①⓪①①', 2)
        {'result': '①⓪①①⓪⓪', 'decimal': 44}
    """
    if not validate_binary_notation(a):
        raise ValueError("Input must use only ⓪ and ① characters")

    result = a + ZERO * n

    return {'result': result, 'decimal': circled_binary_to_decimal(result)}


def right_shift(a: str, n: int) -> Dict:
    """
    Right shift: Divide by 2^n.

    Examples:
        >>> right_shift('①⓪①①', 2)
        {'result': '①⓪', 'decimal': 2, 'bits_lost': '①①'}
    """
    if not validate_binary_notation(a):
        raise ValueError("Input must use only ⓪ and ① characters")

    if n >= len(a):
        return {'result': ZERO, 'decimal': 0, 'bits_lost': a}

    bits_lost = a[-n:]
    result = a[:-n]
    result = normalize(result)

    return {
        'result': result,
        'decimal': circled_binary_to_decimal(result),
        'bits_lost': bits_lost
    }


if __name__ == "__main__":
    print("=== Binary Arithmetic with ⓪① Notation ===\n")

    # Addition
    print("Addition: ①⓪①① + ⓪①①⓪")
    result = add('①⓪①①', '⓪①①⓪')
    print(f"Result: {result['result']} (decimal: {result['decimal']})")
    print()

    # Subtraction
    print("Subtraction: ①⓪①① - ⓪①①⓪")
    result = subtract('①⓪①①', '⓪①①⓪')
    print(f"Result: {result['result']} (decimal: {result['decimal']})")
    print()

    # Bitwise operations
    print("Bitwise AND: ①⓪①① & ⓪①①⓪")
    result = bitwise_and('①⓪①①', '⓪①①⓪')
    print(f"Result: {result['result']} (decimal: {result['decimal']})")
    print()

    print("Bitwise OR: ①⓪①① | ⓪①①⓪")
    result = bitwise_or('①⓪①①', '⓪①①⓪')
    print(f"Result: {result['result']} (decimal: {result['decimal']})")
    print()

    print("Bitwise XOR: ①⓪①① ^ ⓪①①⓪")
    result = bitwise_xor('①⓪①①', '⓪①①⓪')
    print(f"Result: {result['result']} (decimal: {result['decimal']})")
    print()

    # Shifts
    print("Left shift: ①⓪①① << 2")
    result = left_shift('①⓪①①', 2)
    print(f"Result: {result['result']} (decimal: {result['decimal']})")
    print()

    print("Right shift: ①⓪①① >> 2")
    result = right_shift('①⓪①①', 2)
    print(f"Result: {result['result']} (decimal: {result['decimal']})")
    print(f"Bits lost: {result['bits_lost']}")
