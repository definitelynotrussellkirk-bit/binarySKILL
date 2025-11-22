#!/usr/bin/env python3
"""
Function/Assignment Generator

Generates functions f: A → B with various constraints.

Object Type: Functions/Maps between finite sets
Standard examples: Arbitrary functions, injections, surjections, bijections

Representative problems:
  - Functions from [m] to [n]
  - One-to-one functions (injections)
  - Onto functions (surjections)
  - Bijections (permutations of [n])
  - Graph colorings (functions with adjacency constraints)

Edge cases:
  - Surjections (inclusion-exclusion with Stirling numbers)
  - Graph colorings (chromatic polynomial)
  - Constrained assignments
"""

import random
from typing import List, Dict, Any, Set, Tuple

# Import counting formulas from shared toolkit
from toolkit.combinatorial_toolkit import factorial, stirling_second


def generate_function(m: int, n: int, seed: int) -> List[int]:
    """
    Generate a random function from [m] to [n].

    A function is represented as a list where f[i] is the image of element i.

    Args:
        m: Size of domain {0, 1, ..., m-1}
        n: Size of codomain {0, 1, ..., n-1}
        seed: Random seed

    Returns:
        List of length m, each element in [0, n)

    Raises:
        ValueError: If m < 0 or n < 0

    Examples:
        >>> f = generate_function(5, 3, seed=42)
        >>> len(f)
        5
        >>> all(0 <= x < 3 for x in f)
        True
    """
    if m < 0 or n < 0:
        raise ValueError(f"m and n must be non-negative, got m={m}, n={n}")

    rng = random.Random(seed)
    return [rng.randrange(n) for _ in range(m)]


def count_functions(m: int, n: int) -> int:
    """
    Count functions from [m] to [n].

    For each of m domain elements, we have n choices independently.

    Args:
        m: Domain size
        n: Codomain size

    Returns:
        n^m

    Examples:
        >>> count_functions(5, 3)
        243
        >>> count_functions(3, 5)
        125
        >>> count_functions(4, 2)
        16
    """
    if m < 0 or n < 0:
        return 0
    if n == 0:
        return 1 if m == 0 else 0
    return n ** m


def generate_injection(m: int, n: int, seed: int) -> List[int]:
    """
    Generate a random injective function from [m] to [n].

    An injection (one-to-one function) maps distinct domain elements to
    distinct codomain elements. Requires m <= n.

    Args:
        m: Domain size
        n: Codomain size (must have n >= m)
        seed: Random seed

    Returns:
        List of m distinct integers from [0, n)

    Raises:
        ValueError: If m > n (injection impossible)

    Examples:
        >>> f = generate_injection(4, 7, seed=42)
        >>> len(f) == len(set(f))
        True
        >>> len(f)
        4
    """
    if m < 0 or n < 0:
        raise ValueError(f"m and n must be non-negative, got m={m}, n={n}")
    if m > n:
        raise ValueError(
            f"Cannot inject {m} elements into {n} elements (need m <= n)"
        )

    rng = random.Random(seed)
    # Sample m distinct values from [0, n)
    return rng.sample(range(n), m)


def count_injections(m: int, n: int) -> int:
    """
    Count injective functions from [m] to [n].

    This is the falling factorial: P(n, m) = n × (n-1) × ... × (n-m+1).

    Args:
        m: Domain size
        n: Codomain size

    Returns:
        P(n, m) = n! / (n-m)!

    Examples:
        >>> count_injections(3, 5)
        60
        >>> count_injections(4, 7)
        840
        >>> count_injections(5, 5)
        120
    """
    if m > n or m < 0 or n < 0:
        return 0
    if m == 0:
        return 1

    # Compute falling factorial: n × (n-1) × ... × (n-m+1)
    result = 1
    for i in range(n, n - m, -1):
        result *= i
    return result


def generate_surjection(m: int, n: int, seed: int, max_attempts: int = 10000) -> List[int]:
    """
    Generate a random surjective function from [m] to [n].

    A surjection (onto function) hits every element of the codomain.
    Requires m >= n.

    Args:
        m: Domain size (must have m >= n)
        n: Codomain size
        seed: Random seed
        max_attempts: Maximum sampling attempts

    Returns:
        List of length m where every value in [0, n) appears at least once

    Raises:
        ValueError: If m < n or if sampling fails

    Examples:
        >>> f = generate_surjection(5, 3, seed=42)
        >>> len(set(f))
        3
        >>> len(f)
        5
    """
    if m < 0 or n < 0:
        raise ValueError(f"m and n must be non-negative, got m={m}, n={n}")
    if m < n:
        raise ValueError(
            f"Cannot surject from {m} elements to {n} elements (need m >= n)"
        )

    if n == 0:
        return [] if m == 0 else None

    rng = random.Random(seed)

    # Strategy: rejection sampling
    for _ in range(max_attempts):
        f = [rng.randrange(n) for _ in range(m)]
        if len(set(f)) == n:  # All codomain elements hit
            return f

    raise ValueError(
        f"Failed to generate surjection after {max_attempts} attempts"
    )


def count_surjections(m: int, n: int) -> int:
    """
    Count surjective functions from [m] to [n].

    Uses inclusion-exclusion principle:
    Sur(m, n) = Σ_{k=0}^{n} (-1)^k × C(n,k) × (n-k)^m

    Equivalently: n! × S(m, n) where S(m, n) is Stirling number of 2nd kind.

    Args:
        m: Domain size
        n: Codomain size

    Returns:
        Number of surjective functions

    Examples:
        >>> count_surjections(5, 3)
        150
        >>> count_surjections(4, 2)
        14
        >>> count_surjections(3, 3)
        6
    """
    if m < n or m < 0 or n < 0:
        return 0
    if n == 0:
        return 1 if m == 0 else 0

    # Use Stirling numbers of the second kind
    # Sur(m, n) = n! × S(m, n)
    return factorial(n) * stirling_second(m, n)


def generate_bijection(n: int, seed: int) -> List[int]:
    """
    Generate a random bijection from [n] to [n] (permutation).

    Args:
        n: Size of domain and codomain
        seed: Random seed

    Returns:
        Permutation of [0, 1, ..., n-1]

    Examples:
        >>> b = generate_bijection(5, seed=42)
        >>> sorted(b) == list(range(5))
        True
    """
    # This is just a permutation
    rng = random.Random(seed)
    perm = list(range(n))
    rng.shuffle(perm)
    return perm


def count_bijections(n: int) -> int:
    """
    Count bijections from [n] to [n].

    Bijections from [n] to [n] are permutations.

    Args:
        n: Size of sets

    Returns:
        n!

    Examples:
        >>> count_bijections(5)
        120
        >>> count_bijections(3)
        6
    """
    return factorial(n)


def validate_function(f: List[int], m: int, n: int) -> bool:
    """
    Check if a list represents a valid function from [m] to [n].

    Args:
        f: Proposed function (list of integers)
        m: Domain size
        n: Codomain size

    Returns:
        True if f is a valid function

    Examples:
        >>> validate_function([0, 2, 1], 3, 3)
        True
        >>> validate_function([0, 3, 1], 3, 3)
        False
        >>> validate_function([0, 1], 3, 3)
        False
    """
    if len(f) != m:
        return False
    if any(x < 0 or x >= n for x in f):
        return False
    return True


def validate_injection(f: List[int]) -> bool:
    """
    Check if a function is injective (one-to-one).

    Args:
        f: Function as list

    Returns:
        True if f is injective

    Examples:
        >>> validate_injection([0, 2, 4, 1])
        True
        >>> validate_injection([0, 2, 0, 1])
        False
    """
    return len(f) == len(set(f))


def validate_surjection(f: List[int], n: int) -> bool:
    """
    Check if a function is surjective (onto).

    Args:
        f: Function as list
        n: Codomain size

    Returns:
        True if f hits all elements of [0, n)

    Examples:
        >>> validate_surjection([0, 1, 2, 1, 0], 3)
        True
        >>> validate_surjection([0, 1, 1, 0], 3)
        False
    """
    return len(set(f)) == n


def validate_bijection(f: List[int], n: int) -> bool:
    """
    Check if a function is bijective.

    Args:
        f: Function as list
        n: Domain/codomain size

    Returns:
        True if f is a bijection

    Examples:
        >>> validate_bijection([2, 0, 1], 3)
        True
        >>> validate_bijection([0, 1, 1], 3)
        False
    """
    return (len(f) == n and
            validate_injection(f) and
            validate_surjection(f, n))


def get_params_for_difficulty(difficulty: str) -> Dict[str, Any]:
    """
    Get parameter ranges for different difficulty levels.

    Args:
        difficulty: One of 'easy', 'medium', 'hard', 'expert'

    Returns:
        Dictionary with parameter ranges

    Examples:
        >>> get_params_for_difficulty('easy')
        {'m_range': (3, 5), 'n_range': (3, 5)}
        >>> get_params_for_difficulty('expert')
        {'m_range': (10, 15), 'n_range': (8, 12)}
    """
    params = {
        'easy': {
            'm_range': (3, 5),
            'n_range': (3, 5),
        },
        'medium': {
            'm_range': (5, 8),
            'n_range': (4, 7),
        },
        'hard': {
            'm_range': (8, 12),
            'n_range': (6, 10),
        },
        'expert': {
            'm_range': (10, 15),
            'n_range': (8, 12),
        },
    }

    if difficulty not in params:
        raise ValueError(
            f"Unknown difficulty '{difficulty}'. "
            f"Must be one of: {list(params.keys())}"
        )

    return params[difficulty]


# ============================================================================
# Graph Coloring Helpers
# ============================================================================

def generate_graph_coloring(
    edges: List[Tuple[int, int]],
    n_vertices: int,
    n_colors: int,
    seed: int,
    max_attempts: int = 10000
) -> List[int]:
    """
    Generate a random proper coloring of a graph.

    A proper coloring assigns colors to vertices such that adjacent
    vertices have different colors.

    Args:
        edges: List of edges (u, v) where 0 <= u, v < n_vertices
        n_vertices: Number of vertices
        n_colors: Number of colors available
        seed: Random seed
        max_attempts: Maximum sampling attempts

    Returns:
        List of length n_vertices with colors in [0, n_colors)

    Raises:
        ValueError: If no coloring found

    Examples:
        >>> edges = [(0, 1), (1, 2)]
        >>> c = generate_graph_coloring(edges, 3, 2, seed=42)
        >>> c[0] != c[1] and c[1] != c[2]
        True
    """
    if n_vertices < 0 or n_colors < 0:
        raise ValueError("n_vertices and n_colors must be non-negative")

    rng = random.Random(seed)

    # Build adjacency structure
    adj = [set() for _ in range(n_vertices)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    # Try random colorings
    for _ in range(max_attempts):
        coloring = [rng.randrange(n_colors) for _ in range(n_vertices)]

        # Check if proper
        proper = True
        for u, v in edges:
            if coloring[u] == coloring[v]:
                proper = False
                break

        if proper:
            return coloring

    raise ValueError(
        f"Could not find proper {n_colors}-coloring after {max_attempts} attempts"
    )


def validate_graph_coloring(
    coloring: List[int],
    edges: List[Tuple[int, int]]
) -> bool:
    """
    Check if a coloring is proper (no adjacent vertices same color).

    Args:
        coloring: Color assignment for each vertex
        edges: List of edges

    Returns:
        True if proper coloring

    Examples:
        >>> validate_graph_coloring([0, 1, 0], [(0, 1), (1, 2)])
        True
        >>> validate_graph_coloring([0, 0, 1], [(0, 1), (1, 2)])
        False
    """
    for u, v in edges:
        if coloring[u] == coloring[v]:
            return False
    return True
