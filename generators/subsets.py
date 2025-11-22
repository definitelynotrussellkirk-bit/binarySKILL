#!/usr/bin/env python3
"""
Subset/Combination Generator

Generates k-element subsets of [n] and variants with constraints.

Object Type: Subsets/Selection/Combinations
Standard notation: C(n, k) = n! / (k! (n-k)!)

Representative problems:
  - Choose k students from n
  - Occupy k non-adjacent seats in a row of n
  - Subsets with sum constraints

Edge cases:
  - No adjacent elements (spacing constraints)
  - Subset sum conditions
  - Multiple simultaneous constraints
"""

import random
from typing import List, Dict, Any, Set

# Import counting formulas from shared toolkit
from toolkit.combinatorial_toolkit import binomial


def generate_k_subset(n: int, k: int, seed: int) -> List[int]:
    """
    Generate a random k-element subset of {0, 1, ..., n-1}.

    Args:
        n: Size of the universe (positive integer)
        k: Size of subset to select (0 <= k <= n)
        seed: Random seed for deterministic generation

    Returns:
        Sorted list of k distinct integers in range [0, n)

    Raises:
        ValueError: If k > n or k < 0 or n < 0

    Examples:
        >>> generate_k_subset(10, 4, seed=42)
        [1, 3, 5, 8]
        >>> generate_k_subset(5, 0, seed=0)
        []
        >>> generate_k_subset(3, 3, seed=0)
        [0, 1, 2]
    """
    if n < 0 or k < 0:
        raise ValueError(f"n and k must be non-negative, got n={n}, k={k}")
    if k > n:
        raise ValueError(f"Cannot choose {k} items from {n}")

    rng = random.Random(seed)
    return sorted(rng.sample(range(n), k))


def count_k_subsets(n: int, k: int) -> int:
    """
    Count k-element subsets of [n] using binomial coefficient C(n, k).

    Args:
        n: Size of the universe
        k: Size of subsets to count

    Returns:
        C(n, k) = n! / (k! (n-k)!)

    Examples:
        >>> count_k_subsets(10, 4)
        210
        >>> count_k_subsets(5, 2)
        10
        >>> count_k_subsets(7, 0)
        1
        >>> count_k_subsets(7, 7)
        1
    """
    return binomial(n, k)


def generate_no_adjacent(n: int, k: int, seed: int) -> List[int]:
    """
    Generate a k-element subset of [n] with no two adjacent elements.

    This is the "occupy k seats in a row of n, no two adjacent" problem.

    Args:
        n: Total number of positions
        k: Number of positions to select
        seed: Random seed for deterministic generation

    Returns:
        Sorted list of k integers with gaps of at least 2

    Raises:
        ValueError: If no valid configuration exists (k too large for n)

    Examples:
        >>> generate_no_adjacent(10, 4, seed=42)
        [1, 3, 6, 8]
        >>> s = generate_no_adjacent(10, 3, seed=0)
        >>> all(s[i+1] - s[i] >= 2 for i in range(len(s)-1))
        True

    Algorithm:
        Choose k elements from [n - k + 1], then spread them out:
        selected[i] becomes selected[i] + i (adds spacing).
    """
    # Need at least k + (k-1) positions for k elements with gaps
    if k > 0 and n < 2 * k - 1:
        raise ValueError(
            f"Cannot place {k} non-adjacent elements in {n} positions "
            f"(need at least {2*k - 1})"
        )

    if k == 0:
        return []

    # Choose k positions from [n - k + 1], then add spacing
    rng = random.Random(seed)
    compressed = sorted(rng.sample(range(n - k + 1), k))

    # Spread: position i becomes compressed[i] + i
    return [x + i for i, x in enumerate(compressed)]


def count_no_adjacent(n: int, k: int) -> int:
    """
    Count k-element subsets of [n] with no two adjacent elements.

    Formula: C(n - k + 1, k)

    Derivation: After choosing k elements, we have (k-1) forced gaps.
    Remaining (n - k - (k-1)) = (n - 2k + 1) positions can be distributed
    as extra spacing. This reduces to choosing k items from (n - k + 1).

    Args:
        n: Total number of positions
        k: Number of elements to select

    Returns:
        Number of valid configurations

    Examples:
        >>> count_no_adjacent(10, 4)
        35
        >>> count_no_adjacent(10, 3)
        56
        >>> count_no_adjacent(5, 3)
        1
    """
    if k == 0:
        return 1
    if n < 2 * k - 1:
        return 0

    return count_k_subsets(n - k + 1, k)


def validate_subset(subset: List[int], n: int, k: int) -> bool:
    """
    Validate that a list is a valid k-element subset of [n].

    Args:
        subset: List of integers
        n: Universe size
        k: Expected subset size

    Returns:
        True if subset is valid (correct size, all in range, no duplicates)

    Examples:
        >>> validate_subset([1, 3, 5], 10, 3)
        True
        >>> validate_subset([1, 3, 5], 10, 4)
        False
        >>> validate_subset([1, 1, 3], 10, 3)
        False
    """
    if len(subset) != k:
        return False
    if len(set(subset)) != k:
        return False  # Duplicates
    if any(x < 0 or x >= n for x in subset):
        return False  # Out of range
    return True


def validate_no_adjacent(subset: List[int]) -> bool:
    """
    Check if subset has no two adjacent elements.

    Args:
        subset: Sorted list of integers

    Returns:
        True if all consecutive pairs have gap >= 2

    Examples:
        >>> validate_no_adjacent([1, 3, 6, 8])
        True
        >>> validate_no_adjacent([1, 2, 4])
        False
    """
    if len(subset) <= 1:
        return True
    return all(subset[i+1] - subset[i] >= 2 for i in range(len(subset) - 1))


def get_params_for_difficulty(difficulty: str) -> Dict[str, Any]:
    """
    Get parameter ranges for different difficulty levels.

    Args:
        difficulty: One of 'easy', 'medium', 'hard', 'expert'

    Returns:
        Dictionary with 'n_range' and 'k_range' tuples

    Examples:
        >>> get_params_for_difficulty('easy')
        {'n_range': (4, 7), 'k_range': (2, 4), 'max_n': 7}
        >>> get_params_for_difficulty('expert')
        {'n_range': (15, 25), 'k_range': (5, 12), 'max_n': 25}
    """
    params = {
        'easy': {
            'n_range': (4, 7),
            'k_range': (2, 4),
            'max_n': 7,
        },
        'medium': {
            'n_range': (8, 12),
            'k_range': (3, 6),
            'max_n': 12,
        },
        'hard': {
            'n_range': (13, 18),
            'k_range': (5, 9),
            'max_n': 18,
        },
        'expert': {
            'n_range': (15, 25),
            'k_range': (5, 12),
            'max_n': 25,
        },
    }

    if difficulty not in params:
        raise ValueError(
            f"Unknown difficulty '{difficulty}'. "
            f"Must be one of: {list(params.keys())}"
        )

    return params[difficulty]


def generate_subset_with_sum(
    n: int,
    target_sum: int,
    seed: int,
    max_attempts: int = 1000
) -> List[int]:
    """
    Generate a subset of [n] with a specific sum (if possible).

    This is an NP-complete problem in general, so we use heuristic sampling.

    Args:
        n: Universe size (subset of {0, 1, ..., n-1})
        target_sum: Desired sum of elements
        seed: Random seed
        max_attempts: Maximum random attempts before giving up

    Returns:
        Subset with the target sum (if found)

    Raises:
        ValueError: If no valid subset found within max_attempts

    Examples:
        >>> generate_subset_with_sum(10, 15, seed=42)  # doctest: +SKIP
        [0, 5, 10]  # Example: 0+5+10=15
    """
    rng = random.Random(seed)

    # Try random subsets
    for _ in range(max_attempts):
        k = rng.randint(1, n)
        subset = sorted(rng.sample(range(n), k))
        if sum(subset) == target_sum:
            return subset

    raise ValueError(
        f"Could not find subset of [0..{n-1}] with sum {target_sum} "
        f"after {max_attempts} attempts"
    )


# Utility function for use in other modules
def subset_to_binary_string(subset: List[int], n: int) -> str:
    """
    Convert a subset to its characteristic binary string.

    Args:
        subset: Sorted list of positions (0-indexed)
        n: Length of binary string

    Returns:
        Binary string of length n (position i is '1' iff i in subset)

    Examples:
        >>> subset_to_binary_string([0, 2, 4], 6)
        '101010'
        >>> subset_to_binary_string([], 5)
        '00000'
        >>> subset_to_binary_string([1, 3], 5)
        '01010'
    """
    bits = ['0'] * n
    for pos in subset:
        bits[pos] = '1'
    return ''.join(bits)


def binary_string_to_subset(binary_str: str) -> List[int]:
    """
    Convert a binary string to a subset (inverse of subset_to_binary_string).

    Args:
        binary_str: String of '0' and '1' characters

    Returns:
        Sorted list of positions where character is '1'

    Examples:
        >>> binary_string_to_subset('101010')
        [0, 2, 4]
        >>> binary_string_to_subset('00000')
        []
        >>> binary_string_to_subset('01010')
        [1, 3]
    """
    return [i for i, bit in enumerate(binary_str) if bit == '1']
