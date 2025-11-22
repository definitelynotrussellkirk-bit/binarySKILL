#!/usr/bin/env python3
"""
Word/String Generator

Generates strings over alphabets with constraints.

Object Type: Words/Sequences over finite alphabets
Standard examples: Binary strings, balanced parentheses, pattern avoidance

Representative problems:
  - Binary strings of length n with k ones
  - Binary strings with no consecutive 1s (Fibonacci)
  - Balanced parentheses / Dyck words (Catalan)
  - Strings avoiding forbidden substrings

Edge cases:
  - Consecutive symbol constraints (Fibonacci recurrence)
  - Substring avoidance (DFA/automaton states)
  - Balanced structures (Dyck words → Catalan)
"""

import random
from typing import List, Dict, Any, Tuple

# Import counting formulas from shared toolkit
from toolkit.combinatorial_toolkit import binomial, fibonacci, catalan


def generate_binary_string(n: int, k: int, seed: int) -> str:
    """
    Generate a random binary string of length n with exactly k ones.

    Args:
        n: Length of string
        k: Number of ones (0 <= k <= n)
        seed: Random seed

    Returns:
        Binary string of length n with k ones

    Raises:
        ValueError: If k > n or k < 0 or n < 0

    Examples:
        >>> s = generate_binary_string(8, 4, seed=42)
        >>> len(s)
        8
        >>> s.count('1')
        4
    """
    if n < 0 or k < 0:
        raise ValueError(f"n and k must be non-negative, got n={n}, k={k}")
    if k > n:
        raise ValueError(f"Cannot place {k} ones in string of length {n}")

    rng = random.Random(seed)

    # Choose k positions for ones
    positions = set(rng.sample(range(n), k))

    # Build string
    return ''.join('1' if i in positions else '0' for i in range(n))


def count_binary_strings(n: int, k: int) -> int:
    """
    Count binary strings of length n with exactly k ones.

    This is equivalent to choosing k positions from n for the ones.

    Args:
        n: Length of string
        k: Number of ones

    Returns:
        C(n, k) = n! / (k! (n-k)!)

    Examples:
        >>> count_binary_strings(8, 4)
        70
        >>> count_binary_strings(10, 5)
        252
        >>> count_binary_strings(5, 0)
        1
    """
    return binomial(n, k)


def generate_no_consecutive_ones(n: int, seed: int) -> str:
    """
    Generate a random binary string of length n with no two consecutive 1s.

    This is a classic Fibonacci object: f(n) = f(n-1) + f(n-2).

    Args:
        n: Length of string
        seed: Random seed

    Returns:
        Binary string with no "11" substring

    Raises:
        ValueError: If n < 0

    Examples:
        >>> s = generate_no_consecutive_ones(10, seed=42)
        >>> '11' in s
        False
        >>> len(s)
        10
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")

    if n == 0:
        return ""
    if n == 1:
        rng = random.Random(seed)
        return rng.choice(['0', '1'])

    rng = random.Random(seed)

    # Build string recursively using valid continuations
    # After '0': can add '0' or '1'
    # After '1': can only add '0'

    result = []
    prev = rng.choice(['0', '1'])
    result.append(prev)

    for _ in range(n - 1):
        if prev == '0':
            # Can add either 0 or 1
            next_char = rng.choice(['0', '1'])
        else:
            # Must add 0 (can't have consecutive 1s)
            next_char = '0'

        result.append(next_char)
        prev = next_char

    return ''.join(result)


def count_no_consecutive_ones(n: int) -> int:
    """
    Count binary strings of length n with no two consecutive 1s.

    This satisfies the Fibonacci recurrence:
      f(n) = f(n-1) + f(n-2)
    with f(1) = 2, f(2) = 3.

    The count is fibonacci(n+2) where fibonacci uses F(0)=0, F(1)=1.

    Args:
        n: Length of string

    Returns:
        Number of valid strings

    Examples:
        >>> count_no_consecutive_ones(1)
        2
        >>> count_no_consecutive_ones(2)
        3
        >>> count_no_consecutive_ones(3)
        5
        >>> count_no_consecutive_ones(10)
        144
    """
    if n < 0:
        return 0
    if n == 0:
        return 1  # Empty string

    # The count is F(n+2) where F is standard Fibonacci (F(0)=0, F(1)=1)
    return fibonacci(n + 2)


def generate_dyck_word(n: int, seed: int) -> str:
    """
    Generate a random Dyck word (balanced parentheses) with n pairs.

    A Dyck word has n opening parens '(' and n closing parens ')',
    and never has more closing parens than opening parens at any prefix.

    Args:
        n: Number of pairs of parentheses
        seed: Random seed

    Returns:
        Dyck word as string of length 2n

    Raises:
        ValueError: If n < 0

    Examples:
        >>> d = generate_dyck_word(3, seed=42)
        >>> len(d)
        6
        >>> d.count('(') == d.count(')')
        True
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")

    if n == 0:
        return ""

    rng = random.Random(seed)

    # Generate recursively using Catalan structure
    # At each step: if we can open (have opens left), choose open or close
    # If we must close (no opens left or balance requires it), close

    result = []
    opens = 0
    closes = 0

    while opens + closes < 2 * n:
        # Can we open? (have opens left)
        can_open = opens < n
        # Can we close? (have unclosed opens)
        can_close = closes < opens

        if can_open and can_close:
            # Choose randomly
            if rng.random() < 0.5:
                result.append('(')
                opens += 1
            else:
                result.append(')')
                closes += 1
        elif can_open:
            result.append('(')
            opens += 1
        elif can_close:
            result.append(')')
            closes += 1

    return ''.join(result)


def count_dyck_words(n: int) -> int:
    """
    Count Dyck words (balanced parentheses) with n pairs.

    This is the nth Catalan number: C(n) = C(2n, n) / (n + 1).

    Args:
        n: Number of pairs of parentheses

    Returns:
        The nth Catalan number

    Examples:
        >>> count_dyck_words(3)
        5
        >>> count_dyck_words(4)
        14
        >>> count_dyck_words(5)
        42
    """
    return catalan(n)


def validate_binary_string(s: str, n: int, k: int) -> bool:
    """
    Validate that a string is a binary string of length n with k ones.

    Args:
        s: String to validate
        n: Expected length
        k: Expected number of ones

    Returns:
        True if valid

    Examples:
        >>> validate_binary_string('10110', 5, 3)
        True
        >>> validate_binary_string('10110', 5, 2)
        False
        >>> validate_binary_string('102', 3, 2)
        False
    """
    if len(s) != n:
        return False
    if not all(c in '01' for c in s):
        return False
    if s.count('1') != k:
        return False
    return True


def validate_no_consecutive_ones(s: str) -> bool:
    """
    Check if a binary string has no two consecutive 1s.

    Args:
        s: Binary string

    Returns:
        True if no "11" substring exists

    Examples:
        >>> validate_no_consecutive_ones('10101')
        True
        >>> validate_no_consecutive_ones('10110')
        False
        >>> validate_no_consecutive_ones('00000')
        True
    """
    return '11' not in s


def validate_dyck_word(s: str) -> bool:
    """
    Check if a string is a valid Dyck word (balanced parentheses).

    Args:
        s: String to validate

    Returns:
        True if valid Dyck word

    Examples:
        >>> validate_dyck_word('()()')
        True
        >>> validate_dyck_word('(())')
        True
        >>> validate_dyck_word('())(')
        False
        >>> validate_dyck_word('((')
        False
    """
    # Must be even length
    if len(s) % 2 != 0:
        return False

    # Check balance and no negative prefix
    balance = 0
    for c in s:
        if c == '(':
            balance += 1
        elif c == ')':
            balance -= 1
        else:
            return False  # Invalid character

        # Never go negative
        if balance < 0:
            return False

    # Must end balanced
    return balance == 0


def get_params_for_difficulty(difficulty: str) -> Dict[str, Any]:
    """
    Get parameter ranges for different difficulty levels.

    Args:
        difficulty: One of 'easy', 'medium', 'hard', 'expert'

    Returns:
        Dictionary with 'n_range' tuple

    Examples:
        >>> get_params_for_difficulty('easy')
        {'n_range': (4, 8), 'max_n': 8}
        >>> get_params_for_difficulty('expert')
        {'n_range': (20, 30), 'max_n': 30}
    """
    params = {
        'easy': {
            'n_range': (4, 8),
            'max_n': 8,
        },
        'medium': {
            'n_range': (9, 15),
            'max_n': 15,
        },
        'hard': {
            'n_range': (16, 25),
            'max_n': 25,
        },
        'expert': {
            'n_range': (20, 30),
            'max_n': 30,
        },
    }

    if difficulty not in params:
        raise ValueError(
            f"Unknown difficulty '{difficulty}'. "
            f"Must be one of: {list(params.keys())}"
        )

    return params[difficulty]


# ============================================================================
# Bijection Helpers
# ============================================================================

def binary_string_to_subset(binary_str: str) -> List[int]:
    """
    Convert a binary string to the subset it represents.

    Position i is in the subset iff bit i is '1'.

    Args:
        binary_str: String of '0' and '1'

    Returns:
        Sorted list of positions with '1'

    Examples:
        >>> binary_string_to_subset('10110')
        [0, 2, 3]
        >>> binary_string_to_subset('00000')
        []
    """
    return [i for i, bit in enumerate(binary_str) if bit == '1']


def subset_to_binary_string(subset: List[int], n: int) -> str:
    """
    Convert a subset to its characteristic binary string.

    Args:
        subset: List of positions (0-indexed)
        n: Length of binary string

    Returns:
        Binary string where position i is '1' iff i in subset

    Examples:
        >>> subset_to_binary_string([0, 2, 3], 5)
        '10110'
        >>> subset_to_binary_string([], 5)
        '00000'
    """
    bits = ['0'] * n
    for pos in subset:
        if 0 <= pos < n:
            bits[pos] = '1'
    return ''.join(bits)


def binary_string_to_lattice_path(binary_str: str) -> str:
    """
    Convert a binary string to a lattice path.

    Convention: '1' → 'R' (right), '0' → 'U' (up)

    Args:
        binary_str: String of '0' and '1'

    Returns:
        String of 'R' and 'U' characters

    Examples:
        >>> binary_string_to_lattice_path('10110')
        'RURRU'
        >>> binary_string_to_lattice_path('00000')
        'UUUUU'
    """
    return binary_str.replace('1', 'R').replace('0', 'U')


def lattice_path_to_binary_string(path: str) -> str:
    """
    Convert a lattice path to a binary string.

    Convention: 'R' → '1', 'U' → '0'

    Args:
        path: String of 'R' and 'U' characters

    Returns:
        Binary string

    Examples:
        >>> lattice_path_to_binary_string('RURRU')
        '10110'
        >>> lattice_path_to_binary_string('UUUUU')
        '00000'
    """
    return path.replace('R', '1').replace('U', '0')


def dyck_word_to_parenthesization(dyck: str, symbols: List[str] = None) -> str:
    """
    Convert a Dyck word to a parenthesization of symbols.

    Args:
        dyck: Dyck word (balanced parentheses)
        symbols: List of symbols to parenthesize (default: a, b, c, ...)

    Returns:
        Parenthesized expression

    Examples:
        >>> dyck_word_to_parenthesization('(())', ['a', 'b', 'c'])
        '((a·b)·c)'
        >>> dyck_word_to_parenthesization('()()', ['a', 'b', 'c'])
        '(a·(b·c))'
    """
    if not symbols:
        n = len(dyck) // 2
        symbols = [chr(ord('a') + i) for i in range(n + 1)]

    # Stack-based parenthesization
    stack = []
    symbol_idx = 0

    for char in dyck:
        if char == '(':
            stack.append('(')
        else:  # char == ')'
            # Pop until we find matching '('
            temp = []
            while stack and stack[-1] != '(':
                temp.append(stack.pop())

            if stack:
                stack.pop()  # Remove '('

            # Build expression
            if len(temp) == 0:
                # Empty parens, add symbol
                expr = symbols[symbol_idx]
                symbol_idx += 1
            else:
                # Combine two expressions
                if len(temp) == 1:
                    expr = temp[0]
                else:
                    expr = f"({temp[1]}·{temp[0]})"

            stack.append(expr)

    return stack[0] if stack else ""
