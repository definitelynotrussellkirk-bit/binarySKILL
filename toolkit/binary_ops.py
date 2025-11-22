#!/usr/bin/env python3
"""
Binary Arithmetic Operations

Core utilities for binary math operations with step-by-step explanations.
All operations work on strings of '0' and '1' characters.
"""

from typing import List, Dict, Tuple, Optional


def normalize_binary(binary_str: str, width: Optional[int] = None) -> str:
    """
    Normalize a binary string (remove leading zeros, optionally pad to width).

    Args:
        binary_str: Binary string (e.g., '0101')
        width: Optional minimum width (pads with zeros)

    Returns:
        Normalized binary string

    Examples:
        >>> normalize_binary('00101')
        '101'
        >>> normalize_binary('101', width=8)
        '00000101'
    """
    # Remove leading zeros
    normalized = binary_str.lstrip('0') or '0'

    # Pad to width if specified
    if width and len(normalized) < width:
        normalized = '0' * (width - len(normalized)) + normalized

    return normalized


def binary_to_decimal(binary_str: str) -> int:
    """
    Convert binary string to decimal integer.

    Args:
        binary_str: Binary string (e.g., '1011')

    Returns:
        Decimal integer

    Examples:
        >>> binary_to_decimal('1011')
        11
        >>> binary_to_decimal('11111111')
        255
    """
    return int(binary_str, 2)


def decimal_to_binary(n: int, width: Optional[int] = None) -> str:
    """
    Convert decimal integer to binary string.

    Args:
        n: Decimal integer
        width: Optional minimum width (pads with zeros)

    Returns:
        Binary string

    Examples:
        >>> decimal_to_binary(11)
        '1011'
        >>> decimal_to_binary(11, width=8)
        '00001011'
    """
    if n < 0:
        raise ValueError("Negative numbers not supported (use twos_complement instead)")

    binary = bin(n)[2:]  # Remove '0b' prefix

    if width and len(binary) < width:
        binary = '0' * (width - len(binary)) + binary

    return binary


def binary_add(a: str, b: str, show_steps: bool = False) -> Dict:
    """
    Add two binary numbers with optional step-by-step explanation.

    Args:
        a: First binary number
        b: Second binary number
        show_steps: Whether to include step-by-step carry propagation

    Returns:
        Dictionary with 'result', 'carry_out', and optionally 'steps'

    Examples:
        >>> binary_add('1011', '0110')
        {'result': '10001', 'carry_out': False, 'decimal': 17}
        >>> binary_add('1', '1')
        {'result': '10', 'carry_out': False, 'decimal': 2}
    """
    # Pad to same length
    max_len = max(len(a), len(b))
    a = a.zfill(max_len)
    b = b.zfill(max_len)

    result = []
    carry = 0
    steps = [] if show_steps else None

    # Add from right to left
    for i in range(max_len - 1, -1, -1):
        bit_a = int(a[i])
        bit_b = int(b[i])

        sum_bits = bit_a + bit_b + carry
        result_bit = sum_bits % 2
        carry = sum_bits // 2

        result.insert(0, str(result_bit))

        if show_steps:
            steps.append({
                'position': max_len - 1 - i,
                'bit_a': bit_a,
                'bit_b': bit_b,
                'carry_in': carry - (sum_bits // 2),  # Previous carry
                'sum': sum_bits,
                'result_bit': result_bit,
                'carry_out': carry
            })

    # Add final carry if needed
    if carry:
        result.insert(0, '1')

    result_str = ''.join(result)
    decimal_result = binary_to_decimal(result_str)

    output = {
        'result': result_str,
        'carry_out': carry == 1,
        'decimal': decimal_result
    }

    if show_steps:
        output['steps'] = steps

    return output


def binary_subtract(a: str, b: str, show_steps: bool = False) -> Dict:
    """
    Subtract two binary numbers (a - b) with optional step-by-step explanation.

    Args:
        a: Minuend (number to subtract from)
        b: Subtrahend (number to subtract)
        show_steps: Whether to include step-by-step borrow propagation

    Returns:
        Dictionary with 'result', 'borrow', and optionally 'steps'

    Raises:
        ValueError: If b > a (result would be negative)

    Examples:
        >>> binary_subtract('1011', '0110')
        {'result': '101', 'borrow': False, 'decimal': 5}
    """
    dec_a = binary_to_decimal(a)
    dec_b = binary_to_decimal(b)

    if dec_b > dec_a:
        raise ValueError(f"Cannot subtract {b} ({dec_b}) from {a} ({dec_a}): result would be negative")

    # Pad to same length
    max_len = max(len(a), len(b))
    a = a.zfill(max_len)
    b = b.zfill(max_len)

    result = []
    borrow = 0
    steps = [] if show_steps else None

    # Subtract from right to left
    for i in range(max_len - 1, -1, -1):
        bit_a = int(a[i]) - borrow
        bit_b = int(b[i])

        if bit_a < bit_b:
            # Need to borrow
            result_bit = bit_a + 2 - bit_b
            borrow = 1
        else:
            result_bit = bit_a - bit_b
            borrow = 0

        result.insert(0, str(result_bit))

        if show_steps:
            steps.append({
                'position': max_len - 1 - i,
                'bit_a': int(a[i]),
                'bit_b': bit_b,
                'borrow_in': borrow if bit_a < bit_b else 0,
                'result_bit': result_bit,
                'borrow_out': borrow
            })

    result_str = normalize_binary(''.join(result))
    decimal_result = binary_to_decimal(result_str)

    output = {
        'result': result_str,
        'borrow': borrow == 1,
        'decimal': decimal_result
    }

    if show_steps:
        output['steps'] = steps

    return output


def binary_and(a: str, b: str) -> Dict:
    """
    Bitwise AND operation.

    Args:
        a: First binary number
        b: Second binary number

    Returns:
        Dictionary with 'result' and 'decimal'

    Examples:
        >>> binary_and('1011', '0110')
        {'result': '0010', 'decimal': 2}
    """
    # Pad to same length
    max_len = max(len(a), len(b))
    a = a.zfill(max_len)
    b = b.zfill(max_len)

    result = ''.join('1' if a[i] == '1' and b[i] == '1' else '0' for i in range(max_len))

    return {
        'result': result,
        'decimal': binary_to_decimal(result)
    }


def binary_or(a: str, b: str) -> Dict:
    """
    Bitwise OR operation.

    Args:
        a: First binary number
        b: Second binary number

    Returns:
        Dictionary with 'result' and 'decimal'

    Examples:
        >>> binary_or('1011', '0110')
        {'result': '1111', 'decimal': 15}
    """
    # Pad to same length
    max_len = max(len(a), len(b))
    a = a.zfill(max_len)
    b = b.zfill(max_len)

    result = ''.join('1' if a[i] == '1' or b[i] == '1' else '0' for i in range(max_len))

    return {
        'result': result,
        'decimal': binary_to_decimal(result)
    }


def binary_xor(a: str, b: str) -> Dict:
    """
    Bitwise XOR operation.

    Args:
        a: First binary number
        b: Second binary number

    Returns:
        Dictionary with 'result' and 'decimal'

    Examples:
        >>> binary_xor('1011', '0110')
        {'result': '1101', 'decimal': 13}
    """
    # Pad to same length
    max_len = max(len(a), len(b))
    a = a.zfill(max_len)
    b = b.zfill(max_len)

    result = ''.join('1' if a[i] != b[i] else '0' for i in range(max_len))

    return {
        'result': result,
        'decimal': binary_to_decimal(result)
    }


def binary_not(a: str, width: Optional[int] = None) -> Dict:
    """
    Bitwise NOT operation (one's complement).

    Args:
        a: Binary number
        width: Optional bit width (default: length of a)

    Returns:
        Dictionary with 'result' and 'decimal'

    Examples:
        >>> binary_not('1011')
        {'result': '0100', 'decimal': 4}
        >>> binary_not('1011', width=8)
        {'result': '11110100', 'decimal': 244}
    """
    if width is None:
        width = len(a)

    a = a.zfill(width)
    result = ''.join('0' if bit == '1' else '1' for bit in a)

    return {
        'result': result,
        'decimal': binary_to_decimal(result)
    }


def left_shift(a: str, n: int) -> Dict:
    """
    Left shift operation (multiply by 2^n).

    Args:
        a: Binary number
        n: Number of positions to shift

    Returns:
        Dictionary with 'result' and 'decimal'

    Examples:
        >>> left_shift('1011', 2)
        {'result': '101100', 'decimal': 44}
    """
    result = a + '0' * n

    return {
        'result': result,
        'decimal': binary_to_decimal(result)
    }


def right_shift(a: str, n: int, fill_bit: str = '0') -> Dict:
    """
    Right shift operation (divide by 2^n).

    Args:
        a: Binary number
        n: Number of positions to shift
        fill_bit: Bit to fill from left ('0' for logical, MSB for arithmetic)

    Returns:
        Dictionary with 'result', 'decimal', and 'bits_lost'

    Examples:
        >>> right_shift('1011', 2)
        {'result': '10', 'decimal': 2, 'bits_lost': '11'}
    """
    if n >= len(a):
        return {
            'result': fill_bit,
            'decimal': int(fill_bit),
            'bits_lost': a
        }

    bits_lost = a[-n:]
    result = fill_bit * n + a[:-n]
    result = normalize_binary(result)

    return {
        'result': result,
        'decimal': binary_to_decimal(result),
        'bits_lost': bits_lost
    }


def twos_complement(a: str, width: int) -> Dict:
    """
    Two's complement (for representing negative numbers).

    Args:
        a: Binary number (represents magnitude)
        width: Bit width for representation

    Returns:
        Dictionary with 'result', 'decimal', and 'signed_value'

    Examples:
        >>> twos_complement('101', width=8)
        {'result': '11111011', 'decimal': 251, 'signed_value': -5}
    """
    # Pad to width
    a = a.zfill(width)

    # Step 1: One's complement (flip all bits)
    ones_comp = binary_not(a, width=width)['result']

    # Step 2: Add 1
    twos_comp = binary_add(ones_comp, '1')['result']

    # Ensure width is preserved
    if len(twos_comp) > width:
        twos_comp = twos_comp[-width:]
    else:
        twos_comp = twos_comp.zfill(width)

    decimal = binary_to_decimal(twos_comp)
    signed_value = -(2**width - decimal)

    return {
        'result': twos_comp,
        'decimal': decimal,
        'signed_value': signed_value
    }


if __name__ == "__main__":
    # Demo
    print("=== Binary Arithmetic Demo ===\n")

    print("Addition: 1011 + 0110")
    result = binary_add('1011', '0110', show_steps=True)
    print(f"Result: {result['result']} (decimal: {result['decimal']})")
    print()

    print("Subtraction: 1011 - 0110")
    result = binary_subtract('1011', '0110')
    print(f"Result: {result['result']} (decimal: {result['decimal']})")
    print()

    print("Bitwise AND: 1011 & 0110")
    result = binary_and('1011', '0110')
    print(f"Result: {result['result']} (decimal: {result['decimal']})")
    print()

    print("Bitwise OR: 1011 | 0110")
    result = binary_or('1011', '0110')
    print(f"Result: {result['result']} (decimal: {result['decimal']})")
    print()

    print("Bitwise XOR: 1011 ^ 0110")
    result = binary_xor('1011', '0110')
    print(f"Result: {result['result']} (decimal: {result['decimal']})")
    print()

    print("Left shift: 1011 << 2")
    result = left_shift('1011', 2)
    print(f"Result: {result['result']} (decimal: {result['decimal']})")
    print()

    print("Right shift: 1011 >> 2")
    result = right_shift('1011', 2)
    print(f"Result: {result['result']} (decimal: {result['decimal']})")
    print(f"Bits lost: {result['bits_lost']}")
    print()

    print("Two's complement of 101 (8-bit)")
    result = twos_complement('101', width=8)
    print(f"Result: {result['result']} (represents {result['signed_value']})")
