#!/usr/bin/env python3
"""
Prefix Notation for Binary Operations

All operations use prefix notation with English operation names:
  add ①⓪①① ⓪①①⓪ = ①⓪⓪⓪①
  subtract ①⓪①① ⓪①①⓪ = ①⓪①
  and ①⓪①① ⓪①①⓪ = ⓪⓪①⓪
  or ①⓪①① ⓪①①⓪ = ①①①①
  xor ①⓪①① ⓪①①⓪ = ①①⓪①
  not ①⓪①① = ⓪①⓪⓪
  leftshift ①⓪①① 2 = ①⓪①①⓪⓪
  rightshift ①⓪①① 2 = ①⓪

This makes binary operations visually distinct from infix arithmetic.
"""

from typing import Dict, List, Union, Optional
from toolkit.binary_arithmetic import (
    add as binary_add,
    subtract as binary_subtract,
    bitwise_and,
    bitwise_or,
    bitwise_xor,
    bitwise_not,
    left_shift,
    right_shift,
    validate_binary_notation
)


def format_prefix_binary_op(op_name: str, operands: List[str], result: str) -> str:
    """
    Format a binary operation in prefix notation.

    Args:
        op_name: Operation name (e.g., 'add', 'and', 'leftshift')
        operands: List of operands (binary strings or integers)
        result: Result of operation

    Returns:
        Formatted prefix expression

    Examples:
        >>> format_prefix_binary_op('add', ['①⓪①①', '⓪①①⓪'], '①⓪⓪⓪①')
        'add ①⓪①① ⓪①①⓪ = ①⓪⓪⓪①'
        >>> format_prefix_binary_op('not', ['①⓪①①'], '⓪①⓪⓪')
        'not ①⓪①① = ⓪①⓪⓪'
    """
    operands_str = ' '.join(str(op) for op in operands)
    return f"{op_name} {operands_str} = {result}"


def add(a: str, b: str, show_prefix: bool = True) -> Dict:
    """
    Add two binary numbers in prefix notation.

    Args:
        a: First binary number
        b: Second binary number
        show_prefix: Include prefix notation in result

    Returns:
        Dict with 'result', 'decimal', and optionally 'prefix'

    Examples:
        >>> add('①⓪①①', '⓪①①⓪')
        {'result': '①⓪⓪⓪①', 'decimal': 17, 'prefix': 'add ①⓪①① ⓪①①⓪ = ①⓪⓪⓪①'}
    """
    result = binary_add(a, b)

    if show_prefix:
        result['prefix'] = format_prefix_binary_op('add', [a, b], result['result'])

    return result


def subtract(a: str, b: str, show_prefix: bool = True) -> Dict:
    """
    Subtract binary numbers in prefix notation.

    Args:
        a: Minuend
        b: Subtrahend
        show_prefix: Include prefix notation in result

    Returns:
        Dict with 'result', 'decimal', and optionally 'prefix'

    Examples:
        >>> subtract('①⓪①①', '⓪①①⓪')
        {'result': '①⓪①', 'decimal': 5, 'prefix': 'subtract ①⓪①① ⓪①①⓪ = ①⓪①'}
    """
    result = binary_subtract(a, b)

    if show_prefix:
        result['prefix'] = format_prefix_binary_op('subtract', [a, b], result['result'])

    return result


def and_op(a: str, b: str, show_prefix: bool = True) -> Dict:
    """
    Bitwise AND in prefix notation.

    Examples:
        >>> and_op('①⓪①①', '⓪①①⓪')
        {'result': '⓪⓪①⓪', 'decimal': 2, 'prefix': 'and ①⓪①① ⓪①①⓪ = ⓪⓪①⓪'}
    """
    result = bitwise_and(a, b)

    if show_prefix:
        result['prefix'] = format_prefix_binary_op('and', [a, b], result['result'])

    return result


def or_op(a: str, b: str, show_prefix: bool = True) -> Dict:
    """
    Bitwise OR in prefix notation.

    Examples:
        >>> or_op('①⓪①①', '⓪①①⓪')
        {'result': '①①①①', 'decimal': 15, 'prefix': 'or ①⓪①① ⓪①①⓪ = ①①①①'}
    """
    result = bitwise_or(a, b)

    if show_prefix:
        result['prefix'] = format_prefix_binary_op('or', [a, b], result['result'])

    return result


def xor(a: str, b: str, show_prefix: bool = True) -> Dict:
    """
    Bitwise XOR in prefix notation.

    Examples:
        >>> xor('①⓪①①', '⓪①①⓪')
        {'result': '①①⓪①', 'decimal': 13, 'prefix': 'xor ①⓪①① ⓪①①⓪ = ①①⓪①'}
    """
    result = bitwise_xor(a, b)

    if show_prefix:
        result['prefix'] = format_prefix_binary_op('xor', [a, b], result['result'])

    return result


def not_op(a: str, width: Optional[int] = None, show_prefix: bool = True) -> Dict:
    """
    Bitwise NOT in prefix notation.

    Args:
        a: Binary number
        width: Optional bit width
        show_prefix: Include prefix notation in result

    Examples:
        >>> not_op('①⓪①①')
        {'result': '⓪①⓪⓪', 'decimal': 4, 'prefix': 'not ①⓪①① = ⓪①⓪⓪'}
    """
    result = bitwise_not(a, width=width)

    if show_prefix:
        result['prefix'] = format_prefix_binary_op('not', [a], result['result'])

    return result


def leftshift(a: str, n: int, show_prefix: bool = True) -> Dict:
    """
    Left shift in prefix notation.

    Args:
        a: Binary number
        n: Number of positions to shift
        show_prefix: Include prefix notation in result

    Examples:
        >>> leftshift('①⓪①①', 2)
        {'result': '①⓪①①⓪⓪', 'decimal': 44, 'prefix': 'leftshift ①⓪①① 2 = ①⓪①①⓪⓪'}
    """
    result = left_shift(a, n)

    if show_prefix:
        result['prefix'] = format_prefix_binary_op('leftshift', [a, n], result['result'])

    return result


def rightshift(a: str, n: int, show_prefix: bool = True) -> Dict:
    """
    Right shift in prefix notation.

    Args:
        a: Binary number
        n: Number of positions to shift
        show_prefix: Include prefix notation in result

    Examples:
        >>> rightshift('①⓪①①', 2)
        {'result': '①⓪', 'decimal': 2, 'bits_lost': '①①', 'prefix': 'rightshift ①⓪①① 2 = ①⓪'}
    """
    result = right_shift(a, n)

    if show_prefix:
        result['prefix'] = format_prefix_binary_op('rightshift', [a, n], result['result'])

    return result


def parse_prefix_expression(expr: str) -> Dict:
    """
    Parse and evaluate a prefix binary expression.

    Args:
        expr: Prefix expression (e.g., 'add ①⓪①① ⓪①①⓪')

    Returns:
        Dict with result and prefix notation

    Examples:
        >>> parse_prefix_expression('add ①⓪①① ⓪①①⓪')
        {'result': '①⓪⓪⓪①', 'decimal': 17, 'prefix': 'add ①⓪①① ⓪①①⓪ = ①⓪⓪⓪①'}
        >>> parse_prefix_expression('not ①⓪①①')
        {'result': '⓪①⓪⓪', 'decimal': 4, 'prefix': 'not ①⓪①① = ⓪①⓪⓪'}
    """
    parts = expr.strip().split()

    if not parts:
        raise ValueError("Empty expression")

    op = parts[0].lower()

    # Unary operations
    if op == 'not':
        if len(parts) != 2:
            raise ValueError(f"'not' expects 1 operand, got {len(parts) - 1}")
        return not_op(parts[1])

    # Binary operations (two operands)
    elif op in ['add', 'subtract', 'and', 'or', 'xor']:
        if len(parts) != 3:
            raise ValueError(f"'{op}' expects 2 operands, got {len(parts) - 1}")

        a, b = parts[1], parts[2]

        if op == 'add':
            return add(a, b)
        elif op == 'subtract':
            return subtract(a, b)
        elif op == 'and':
            return and_op(a, b)
        elif op == 'or':
            return or_op(a, b)
        elif op == 'xor':
            return xor(a, b)

    # Shift operations (operand + integer)
    elif op in ['leftshift', 'rightshift']:
        if len(parts) != 3:
            raise ValueError(f"'{op}' expects 2 arguments, got {len(parts) - 1}")

        a = parts[1]
        try:
            n = int(parts[2])
        except ValueError:
            raise ValueError(f"Shift amount must be an integer, got '{parts[2]}'")

        if op == 'leftshift':
            return leftshift(a, n)
        elif op == 'rightshift':
            return rightshift(a, n)

    else:
        raise ValueError(f"Unknown operation: '{op}'")


def evaluate_prefix_program(expressions: List[str]) -> List[Dict]:
    """
    Evaluate a sequence of prefix expressions.

    Args:
        expressions: List of prefix expressions

    Returns:
        List of results

    Examples:
        >>> evaluate_prefix_program(['add ①⓪①① ⓪①①⓪', 'not ①⓪①①'])
        [{'result': '①⓪⓪⓪①', ...}, {'result': '⓪①⓪⓪', ...}]
    """
    results = []
    for expr in expressions:
        result = parse_prefix_expression(expr)
        results.append(result)
    return results


if __name__ == "__main__":
    print("=== Binary Operations with Prefix Notation ===\n")

    # Arithmetic operations
    print("Arithmetic Operations:")
    print(add('①⓪①①', '⓪①①⓪')['prefix'])
    print(subtract('①⓪①①', '⓪①①⓪')['prefix'])
    print()

    # Bitwise operations
    print("Bitwise Operations:")
    print(and_op('①⓪①①', '⓪①①⓪')['prefix'])
    print(or_op('①⓪①①', '⓪①①⓪')['prefix'])
    print(xor('①⓪①①', '⓪①①⓪')['prefix'])
    print(not_op('①⓪①①')['prefix'])
    print()

    # Shift operations
    print("Shift Operations:")
    print(leftshift('①⓪①①', 2)['prefix'])
    print(rightshift('①⓪①①', 2)['prefix'])
    print()

    # Parsing prefix expressions
    print("Parsing Prefix Expressions:")
    expressions = [
        'add ①⓪①① ⓪①①⓪',
        'and ①⓪①① ⓪①①⓪',
        'not ①⓪①①',
        'leftshift ①⓪①① 2'
    ]

    for expr in expressions:
        result = parse_prefix_expression(expr)
        print(result['prefix'])
