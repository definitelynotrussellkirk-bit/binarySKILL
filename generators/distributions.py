#!/usr/bin/env python3
"""
Distribution / Balls-in-Bins Generator

Generates distributions of items into containers.

Object Type: Distributions (balls-in-bins, compositions, partitions)
Standard examples: Stars-and-bars, twelvefold way, Stirling numbers

Representative problems:
  - Distribute n identical balls into k distinct bins
  - Distribute with lower/upper bounds per bin
  - Indistinguishable bins (Stirling numbers of 2nd kind)

Edge cases:
  - Bounded distributions (PIE for upper bounds)
  - Indistinguishable bins (set partitions)
  - Both indistinguishable (integer partitions)
"""

import random
from typing import List, Dict, Any

# Import counting formulas from shared toolkit
from toolkit.combinatorial_toolkit import binomial, stirling_second


def generate_stars_bars(n: int, k: int, seed: int) -> List[int]:
    """
    Generate a random distribution of n identical items into k distinct bins.

    This is the classic "stars and bars" problem. Result is a composition
    of n into k parts (parts can be 0).

    Args:
        n: Number of items to distribute
        k: Number of bins
        seed: Random seed

    Returns:
        List of k non-negative integers that sum to n

    Raises:
        ValueError: If n < 0 or k < 0

    Examples:
        >>> dist = generate_stars_bars(10, 4, seed=42)
        >>> sum(dist)
        10
        >>> len(dist)
        4
    """
    if n < 0 or k < 0:
        raise ValueError(f"n and k must be non-negative, got n={n}, k={k}")

    if k == 0:
        return [] if n == 0 else None

    rng = random.Random(seed)

    # Choose k-1 divider positions among n+k-1 slots
    # Equivalently: choose k-1 positions from n+k-1
    positions = sorted(rng.sample(range(1, n + k), k - 1))

    # Convert to distribution
    dist = []
    prev = 0
    for pos in positions:
        dist.append(pos - prev - 1)
        prev = pos
    dist.append(n + k - 1 - prev)

    return dist


def count_stars_bars(n: int, k: int) -> int:
    """
    Count ways to distribute n identical items into k distinct bins.

    Formula: C(n + k - 1, k - 1) = C(n + k - 1, n)

    Args:
        n: Number of items
        k: Number of bins

    Returns:
        Number of distributions

    Examples:
        >>> count_stars_bars(10, 4)
        286
        >>> count_stars_bars(5, 3)
        21
        >>> count_stars_bars(0, 5)
        1
    """
    if n < 0 or k < 0:
        return 0
    if k == 0:
        return 1 if n == 0 else 0

    return binomial(n + k - 1, k - 1)


def generate_bounded_distribution(
    n: int,
    k: int,
    lower: int,
    upper: int,
    seed: int,
    max_attempts: int = 10000
) -> List[int]:
    """
    Generate a distribution with bounds on each bin.

    Each bin must receive between lower and upper items (inclusive).

    Args:
        n: Total items to distribute
        k: Number of bins
        lower: Minimum items per bin
        upper: Maximum items per bin
        seed: Random seed
        max_attempts: Maximum sampling attempts

    Returns:
        List of k integers, each in [lower, upper], summing to n

    Raises:
        ValueError: If no valid distribution exists or sampling fails

    Examples:
        >>> dist = generate_bounded_distribution(10, 4, 1, 4, seed=42)
        >>> all(1 <= x <= 4 for x in dist)
        True
        >>> sum(dist)
        10
    """
    # Check feasibility
    if k * lower > n or k * upper < n:
        raise ValueError(
            f"Cannot distribute {n} into {k} bins with bounds [{lower}, {upper}]. "
            f"Need {k * lower} <= {n} <= {k * upper}"
        )

    rng = random.Random(seed)

    # Rejection sampling
    for _ in range(max_attempts):
        # Start with lower in each bin
        dist = [lower] * k

        # Distribute remaining n - k*lower items
        remaining = n - k * lower

        # Randomly add to bins (respecting upper bound)
        for _ in range(remaining):
            # Find bins that can accept more
            available = [i for i in range(k) if dist[i] < upper]
            if not available:
                break
            chosen = rng.choice(available)
            dist[chosen] += 1

        if sum(dist) == n and all(lower <= x <= upper for x in dist):
            return dist

    raise ValueError(
        f"Could not generate bounded distribution after {max_attempts} attempts"
    )


def count_bounded_distribution(n: int, k: int, lower: int, upper: int) -> int:
    """
    Count distributions with bounds using inclusion-exclusion.

    This uses PIE over bins exceeding upper bound.

    Args:
        n: Total items
        k: Number of bins
        lower: Minimum per bin
        upper: Maximum per bin

    Returns:
        Number of valid distributions

    Examples:
        >>> count_bounded_distribution(10, 4, 1, 4)
        17
        >>> count_bounded_distribution(10, 3, 2, 5)
        10
    """
    # Check feasibility
    if k * lower > n or k * upper < n or n < 0 or k < 0:
        return 0

    # Shift: distribute n' = n - k*lower items with upper bound u = upper - lower
    n_prime = n - k * lower
    u = upper - lower

    if u < 0:
        return 0

    # Use PIE: count unrestricted minus those violating upper bound
    # Σ (-1)^i × C(k, i) × C(n' - i(u+1) + k - 1, k - 1)

    total = 0
    for i in range(k + 1):
        excess = i * (u + 1)
        if excess > n_prime:
            break

        # Choose i bins to violate
        ways_to_choose = binomial(k, i)

        # Distribute n' - excess among k bins
        ways_to_distribute = binomial(n_prime - excess + k - 1, k - 1)

        # Add with alternating sign
        sign = 1 if i % 2 == 0 else -1
        total += sign * ways_to_choose * ways_to_distribute

    return total


def count_stirling_second_kind(n: int, k: int) -> int:
    """
    Count ways to distribute n distinct items into k indistinct bins (Stirling S(n,k)).

    This is the number of ways to partition a set of n elements into k non-empty subsets.

    Args:
        n: Number of distinct items
        k: Number of indistinct bins

    Returns:
        Stirling number of second kind S(n, k)

    Examples:
        >>> count_stirling_second_kind(5, 3)
        25
        >>> count_stirling_second_kind(4, 2)
        7
        >>> count_stirling_second_kind(5, 5)
        1
    """
    return stirling_second(n, k)


def validate_distribution(dist: List[int], n: int, k: int) -> bool:
    """
    Check if a list is a valid distribution of n into k bins.

    Args:
        dist: Proposed distribution
        n: Expected total
        k: Expected number of bins

    Returns:
        True if valid

    Examples:
        >>> validate_distribution([3, 2, 5], 10, 3)
        True
        >>> validate_distribution([3, 2, 5], 11, 3)
        False
        >>> validate_distribution([3, 2], 5, 3)
        False
    """
    if len(dist) != k:
        return False
    if sum(dist) != n:
        return False
    if any(x < 0 for x in dist):
        return False
    return True


def validate_bounded(dist: List[int], lower: int, upper: int) -> bool:
    """
    Check if all elements of distribution are within bounds.

    Args:
        dist: Distribution
        lower: Minimum value
        upper: Maximum value

    Returns:
        True if all elements in [lower, upper]

    Examples:
        >>> validate_bounded([2, 3, 4, 1], 1, 4)
        True
        >>> validate_bounded([0, 3, 4, 1], 1, 4)
        False
    """
    return all(lower <= x <= upper for x in dist)


def get_params_for_difficulty(difficulty: str) -> Dict[str, Any]:
    """
    Get parameter ranges for different difficulty levels.

    Args:
        difficulty: One of 'easy', 'medium', 'hard', 'expert'

    Returns:
        Dictionary with parameter ranges

    Examples:
        >>> get_params_for_difficulty('easy')
        {'n_range': (5, 10), 'k_range': (3, 5)}
        >>> get_params_for_difficulty('expert')
        {'n_range': (20, 30), 'k_range': (8, 12)}
    """
    params = {
        'easy': {
            'n_range': (5, 10),
            'k_range': (3, 5),
        },
        'medium': {
            'n_range': (10, 15),
            'k_range': (4, 7),
        },
        'hard': {
            'n_range': (15, 25),
            'k_range': (6, 10),
        },
        'expert': {
            'n_range': (20, 30),
            'k_range': (8, 12),
        },
    }

    if difficulty not in params:
        raise ValueError(
            f"Unknown difficulty '{difficulty}'. "
            f"Must be one of: {list(params.keys())}"
        )

    return params[difficulty]


# ============================================================================
# Composition Helpers (ordered partitions)
# ============================================================================

def generate_composition(n: int, seed: int) -> List[int]:
    """
    Generate a random composition of n (ordered partition into positive parts).

    A composition of n is a way to write n as an ordered sum of positive integers.

    Args:
        n: Number to partition
        seed: Random seed

    Returns:
        List of positive integers summing to n

    Examples:
        >>> c = generate_composition(7, seed=42)
        >>> sum(c)
        7
        >>> all(x > 0 for x in c)
        True
    """
    if n <= 0:
        return [] if n == 0 else None

    rng = random.Random(seed)

    # Choose k-1 dividers from n-1 positions (where k is random)
    # This gives compositions with k parts

    # Decide number of parts (1 to n)
    k = rng.randint(1, n)

    # Choose k-1 divider positions from n-1 gaps
    if k == 1:
        return [n]

    dividers = sorted(rng.sample(range(1, n), k - 1))

    # Convert to composition
    comp = []
    prev = 0
    for d in dividers:
        comp.append(d - prev)
        prev = d
    comp.append(n - prev)

    return comp


def count_compositions(n: int) -> int:
    """
    Count compositions of n.

    A composition of n is determined by choosing which of the n-1 gaps to cut.

    Args:
        n: Number to partition

    Returns:
        2^(n-1) for n > 0, else 1

    Examples:
        >>> count_compositions(5)
        16
        >>> count_compositions(7)
        64
        >>> count_compositions(1)
        1
    """
    if n <= 0:
        return 1 if n == 0 else 0
    if n == 1:
        return 1

    return 2 ** (n - 1)


def validate_composition(comp: List[int], n: int) -> bool:
    """
    Check if a list is a valid composition of n.

    Args:
        comp: Proposed composition
        n: Expected sum

    Returns:
        True if valid composition

    Examples:
        >>> validate_composition([2, 3, 1], 6)
        True
        >>> validate_composition([0, 3, 1], 4)
        False
        >>> validate_composition([2, 3], 6)
        False
    """
    if sum(comp) != n:
        return False
    if any(x <= 0 for x in comp):
        return False
    return True
