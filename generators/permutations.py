#!/usr/bin/env python3
"""
Permutation/Arrangement Generator

Generates permutations of [n] and variants with constraints.

Object Type: Arrangements/Permutations
Standard notation: P(n, k) = n!/(n-k)! for k-permutations, n! for full permutations

Representative problems:
  - Arrange n books on a shelf
  - Permutations with no fixed points (derangements)
  - Rearrangements of multisets (MISSISSIPPI)
  - Permutations avoiding patterns (132, 123, etc.)

Edge cases:
  - Derangements (PIE application)
  - Multiset permutations (multinomial coefficients)
  - Pattern avoidance (Catalan for some patterns)
"""

import random
from typing import List, Dict, Any
from collections import Counter

# Import counting formulas from shared toolkit
from toolkit.combinatorial_toolkit import factorial, multinomial


def generate_permutation(n: int, seed: int) -> List[int]:
    """
    Generate a random permutation of [0, 1, ..., n-1].

    Args:
        n: Number of elements to permute
        seed: Random seed for deterministic generation

    Returns:
        Permutation as list of n integers (each 0..n-1 appears exactly once)

    Raises:
        ValueError: If n < 0

    Examples:
        >>> generate_permutation(5, seed=42)
        [1, 3, 0, 4, 2]
        >>> len(generate_permutation(10, seed=0))
        10
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")

    rng = random.Random(seed)
    perm = list(range(n))
    rng.shuffle(perm)
    return perm


def count_permutations(n: int) -> int:
    """
    Count permutations of n elements using factorial.

    Args:
        n: Number of elements

    Returns:
        n! = n × (n-1) × ... × 2 × 1

    Examples:
        >>> count_permutations(5)
        120
        >>> count_permutations(0)
        1
        >>> count_permutations(3)
        6
    """
    return factorial(n)


def generate_derangement(n: int, seed: int, max_attempts: int = 10000) -> List[int]:
    """
    Generate a random derangement of [n] (permutation with no fixed points).

    A derangement is a permutation where perm[i] ≠ i for all i.

    Args:
        n: Number of elements
        seed: Random seed
        max_attempts: Maximum random sampling attempts

    Returns:
        Derangement as list of n integers

    Raises:
        ValueError: If n < 2 (no derangements exist) or if sampling fails

    Examples:
        >>> d = generate_derangement(5, seed=42)
        >>> all(d[i] != i for i in range(5))
        True
        >>> len(d) == 5 and len(set(d)) == 5
        True
    """
    if n < 2:
        raise ValueError(f"No derangements exist for n={n} (need n ≥ 2)")

    rng = random.Random(seed)

    # Rejection sampling: generate random perms until we find a derangement
    for _ in range(max_attempts):
        perm = list(range(n))
        rng.shuffle(perm)
        if all(perm[i] != i for i in range(n)):
            return perm

    raise ValueError(
        f"Failed to generate derangement after {max_attempts} attempts "
        f"(this is very unlikely for n={n})"
    )


def count_derangements(n: int) -> int:
    """
    Count derangements using the inclusion-exclusion formula.

    D(n) = n! × Σ_{k=0}^{n} (-1)^k / k!
         ≈ n! / e  (for large n)

    Args:
        n: Number of elements

    Returns:
        Number of derangements of [n]

    Examples:
        >>> count_derangements(3)
        2
        >>> count_derangements(4)
        9
        >>> count_derangements(5)
        44
        >>> count_derangements(0)
        1
    """
    if n < 0:
        return 0
    if n == 0:
        return 1

    # Use recurrence: D(n) = (n-1) × [D(n-1) + D(n-2)]
    # More numerically stable than direct PIE formula
    if n == 1:
        return 0
    if n == 2:
        return 1

    d_prev_prev = 1  # D(2)
    d_prev = 2       # D(3)

    for i in range(4, n + 1):
        d_curr = (i - 1) * (d_prev + d_prev_prev)
        d_prev_prev = d_prev
        d_prev = d_curr

    return d_prev


def generate_multiset_permutation(word: str, seed: int) -> str:
    """
    Generate a random permutation of a multiset (word with repeated letters).

    Args:
        word: String with possibly repeated characters
        seed: Random seed

    Returns:
        Permutation of the characters in word

    Examples:
        >>> generate_multiset_permutation("AAB", seed=42)
        'ABA'
        >>> len(generate_multiset_permutation("MISSISSIPPI", seed=0))
        11
    """
    rng = random.Random(seed)
    chars = list(word)
    rng.shuffle(chars)
    return ''.join(chars)


def count_multiset_permutations(word: str) -> int:
    """
    Count distinct permutations of a multiset using multinomial coefficient.

    Formula: n! / (c1! × c2! × ... × ck!)
    where ci is the count of character i.

    Args:
        word: String with possibly repeated characters

    Returns:
        Number of distinct rearrangements

    Examples:
        >>> count_multiset_permutations("AAB")
        3
        >>> count_multiset_permutations("AABB")
        6
        >>> count_multiset_permutations("MISSISSIPPI")
        34650
    """
    n = len(word)
    if n == 0:
        return 1

    # Count frequencies
    freq = Counter(word)

    # Use toolkit multinomial formula
    return multinomial(n, freq)


def generate_k_permutation(n: int, k: int, seed: int) -> List[int]:
    """
    Generate a random k-permutation (arrangement) of [n].

    A k-permutation is an ordered selection of k distinct elements from [n].

    Args:
        n: Size of universe
        k: Number of elements to arrange (0 <= k <= n)
        seed: Random seed

    Returns:
        List of k distinct integers from [0, n-1]

    Raises:
        ValueError: If k > n or k < 0 or n < 0

    Examples:
        >>> generate_k_permutation(10, 4, seed=42)
        [7, 0, 1, 9]
        >>> len(generate_k_permutation(10, 5, seed=0))
        5
    """
    if n < 0 or k < 0:
        raise ValueError(f"n and k must be non-negative, got n={n}, k={k}")
    if k > n:
        raise ValueError(f"Cannot arrange {k} items from {n}")

    rng = random.Random(seed)
    return rng.sample(range(n), k)


def count_k_permutations(n: int, k: int) -> int:
    """
    Count k-permutations (arrangements) using falling factorial.

    P(n, k) = n × (n-1) × ... × (n-k+1) = n! / (n-k)!

    Args:
        n: Size of universe
        k: Number of elements to arrange

    Returns:
        P(n, k)

    Examples:
        >>> count_k_permutations(10, 4)
        5040
        >>> count_k_permutations(5, 2)
        20
        >>> count_k_permutations(7, 7)
        5040
        >>> count_k_permutations(5, 0)
        1
    """
    if k > n or k < 0 or n < 0:
        return 0
    if k == 0:
        return 1

    # Compute falling factorial: n × (n-1) × ... × (n-k+1)
    result = 1
    for i in range(n, n - k, -1):
        result *= i
    return result


def validate_permutation(perm: List[int], n: int) -> bool:
    """
    Check if a list is a valid permutation of [0, 1, ..., n-1].

    Args:
        perm: List of integers
        n: Expected size

    Returns:
        True if perm is a valid permutation

    Examples:
        >>> validate_permutation([2, 0, 1], 3)
        True
        >>> validate_permutation([1, 2, 3], 3)
        False
        >>> validate_permutation([0, 0, 1], 3)
        False
    """
    if len(perm) != n:
        return False
    if sorted(perm) != list(range(n)):
        return False
    return True


def validate_derangement(perm: List[int]) -> bool:
    """
    Check if a permutation is a derangement (no fixed points).

    Args:
        perm: Permutation as list

    Returns:
        True if perm[i] ≠ i for all i

    Examples:
        >>> validate_derangement([1, 0, 3, 2])
        True
        >>> validate_derangement([0, 2, 1])
        False
        >>> validate_derangement([1, 2, 0])
        True
    """
    return all(perm[i] != i for i in range(len(perm)))


def validate_k_permutation(perm: List[int], n: int, k: int) -> bool:
    """
    Check if a list is a valid k-permutation from [n].

    Args:
        perm: List of integers
        n: Universe size
        k: Expected arrangement size

    Returns:
        True if perm is a valid k-permutation

    Examples:
        >>> validate_k_permutation([3, 1, 5], 10, 3)
        True
        >>> validate_k_permutation([3, 1, 1], 10, 3)
        False
        >>> validate_k_permutation([3, 1], 10, 3)
        False
    """
    if len(perm) != k:
        return False
    if len(set(perm)) != k:
        return False  # Duplicates
    if any(x < 0 or x >= n for x in perm):
        return False  # Out of range
    return True


def get_params_for_difficulty(difficulty: str) -> Dict[str, Any]:
    """
    Get parameter ranges for different difficulty levels.

    Args:
        difficulty: One of 'easy', 'medium', 'hard', 'expert'

    Returns:
        Dictionary with 'n_range' tuple

    Examples:
        >>> get_params_for_difficulty('easy')
        {'n_range': (3, 5), 'max_n': 5}
        >>> get_params_for_difficulty('expert')
        {'n_range': (10, 15), 'max_n': 15}
    """
    params = {
        'easy': {
            'n_range': (3, 5),
            'max_n': 5,
        },
        'medium': {
            'n_range': (6, 8),
            'max_n': 8,
        },
        'hard': {
            'n_range': (9, 12),
            'max_n': 12,
        },
        'expert': {
            'n_range': (10, 15),
            'max_n': 15,
        },
    }

    if difficulty not in params:
        raise ValueError(
            f"Unknown difficulty '{difficulty}'. "
            f"Must be one of: {list(params.keys())}"
        )

    return params[difficulty]


# Utility functions

def permutation_to_cycles(perm: List[int]) -> List[List[int]]:
    """
    Convert a permutation to cycle notation.

    Args:
        perm: Permutation as list (perm[i] is where i maps to)

    Returns:
        List of cycles (each cycle is a list of positions)

    Examples:
        >>> permutation_to_cycles([1, 0, 3, 2])
        [[0, 1], [2, 3]]
        >>> permutation_to_cycles([1, 2, 0])
        [[0, 1, 2]]
        >>> permutation_to_cycles([0, 1, 2])
        [[0], [1], [2]]
    """
    n = len(perm)
    visited = [False] * n
    cycles = []

    for start in range(n):
        if visited[start]:
            continue

        cycle = []
        current = start
        while not visited[current]:
            visited[current] = True
            cycle.append(current)
            current = perm[current]

        if cycle:
            cycles.append(cycle)

    return cycles


def cycles_to_permutation(cycles: List[List[int]], n: int) -> List[int]:
    """
    Convert cycle notation to a permutation.

    Args:
        cycles: List of cycles
        n: Size of permutation

    Returns:
        Permutation as list

    Examples:
        >>> cycles_to_permutation([[0, 1], [2, 3]], 4)
        [1, 0, 3, 2]
        >>> cycles_to_permutation([[0, 1, 2]], 3)
        [1, 2, 0]
    """
    perm = list(range(n))
    for cycle in cycles:
        for i in range(len(cycle)):
            perm[cycle[i]] = cycle[(i + 1) % len(cycle)]
    return perm


def count_permutations_with_k_cycles(n: int, k: int) -> int:
    """
    Count permutations of [n] with exactly k cycles (Stirling numbers of 1st kind).

    Args:
        n: Number of elements
        k: Number of cycles

    Returns:
        Unsigned Stirling number of first kind s(n, k)

    Examples:
        >>> count_permutations_with_k_cycles(4, 2)
        11
        >>> count_permutations_with_k_cycles(5, 1)
        24
        >>> count_permutations_with_k_cycles(3, 3)
        1
    """
    if n == 0 and k == 0:
        return 1
    if n == 0 or k == 0:
        return 0
    if k > n:
        return 0

    # Use recurrence: s(n, k) = s(n-1, k-1) + (n-1) × s(n-1, k)
    # Build table bottom-up
    s = [[0] * (k + 1) for _ in range(n + 1)]
    s[0][0] = 1

    for i in range(1, n + 1):
        for j in range(1, min(i, k) + 1):
            s[i][j] = s[i-1][j-1] + (i-1) * s[i-1][j]

    return s[n][k]
