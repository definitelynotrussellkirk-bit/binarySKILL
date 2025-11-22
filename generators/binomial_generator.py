#!/usr/bin/env python3
"""
Universal Binomial Coefficient Generator

Maximally general C(n,k) object generation with multiple output formats.

This module provides a unified interface for generating any combinatorial
object counted by C(n,k), with flexible output representations:
  - Raw indices (k-subset of [n])
  - Binary strings (n bits, k ones)
  - Lattice paths (k R-moves, n-k U-moves)
  - Named selections (committee, team, etc.)
  - Distributions (stars and bars)
  - Set notation
  - Indicator vectors

All generators are deterministic (seed-based) and bijectively equivalent.
"""

import random
from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass

# Import low-level generators
from generators.subsets import generate_k_subset, count_k_subsets
from generators.words import (
    generate_binary_string,
    binary_string_to_lattice_path,
    lattice_path_to_binary_string
)
from toolkit.combinatorial_toolkit import binomial


# ============================================================================
# Output Format Specifications
# ============================================================================

OutputFormat = Literal[
    "indices",           # [0, 2, 5, 7] - raw k-subset
    "binary",           # "10100101" - binary string
    "lattice_path",     # "RURUURUU" - lattice path (R=1, U=0)
    "named_selection",  # ["Alice", "Carol", "Eve"] - named items
    "distribution",     # [2, 3, 1, 0, 4] - stars and bars
    "set_notation",     # "{0, 2, 5, 7}" - mathematical set
    "indicator",        # [1, 0, 1, 0, 0, 1, 0, 1] - indicator vector
    "positions"         # "positions 0, 2, 5, 7" - natural language
]


@dataclass
class BinomialObject:
    """
    Container for C(n,k) objects with multiple representations.

    Attributes:
        n: Universe size
        k: Selection size
        indices: Canonical k-subset representation (sorted list)
        count: Total number of such objects = C(n,k)
        seed: Random seed used for generation
    """
    n: int
    k: int
    indices: List[int]
    count: int
    seed: int

    def to_binary(self) -> str:
        """Convert to binary string (n bits, k ones)."""
        bits = ['0'] * self.n
        for idx in self.indices:
            bits[idx] = '1'
        return ''.join(bits)

    def to_lattice_path(self) -> str:
        """Convert to lattice path (R for 1, U for 0)."""
        return self.to_binary().replace('1', 'R').replace('0', 'U')

    def to_named_selection(self, names: List[str]) -> List[str]:
        """
        Convert to named selection (e.g., committee members).

        Args:
            names: List of n names to choose from

        Returns:
            List of k selected names

        Raises:
            ValueError: If len(names) != n
        """
        if len(names) != self.n:
            raise ValueError(f"Expected {self.n} names, got {len(names)}")
        return [names[i] for i in self.indices]

    def to_distribution(self, bins: int) -> List[int]:
        """
        Convert to stars-and-bars distribution.

        Interprets the k-subset of [n] as bar placements in
        a stars-and-bars problem with (n - bins + 1) stars.

        Args:
            bins: Number of bins (must satisfy bins = k + 1)

        Returns:
            List of bin counts summing to (n - bins + 1)

        Raises:
            ValueError: If bins != k + 1
        """
        if bins != self.k + 1:
            raise ValueError(
                f"Stars-and-bars requires bins = k + 1. "
                f"Got bins={bins}, k={self.k}"
            )

        # Add boundaries at 0 and n
        bars = [-1] + sorted(self.indices) + [self.n - self.k + bins - 1]

        # Count stars between consecutive bars
        distribution = []
        for i in range(len(bars) - 1):
            count = bars[i + 1] - bars[i] - 1
            distribution.append(count)

        return distribution

    def to_set_notation(self) -> str:
        """Convert to mathematical set notation."""
        if not self.indices:
            return "∅"
        return "{" + ", ".join(map(str, self.indices)) + "}"

    def to_indicator(self) -> List[int]:
        """Convert to indicator/characteristic vector."""
        indicator = [0] * self.n
        for idx in self.indices:
            indicator[idx] = 1
        return indicator

    def to_positions_text(self) -> str:
        """Convert to natural language position description."""
        if not self.indices:
            return "no positions selected"
        if len(self.indices) == 1:
            return f"position {self.indices[0]}"
        return "positions " + ", ".join(map(str, self.indices))

    def to_format(self, format: OutputFormat, **kwargs) -> Any:
        """
        Convert to specified output format.

        Args:
            format: Desired output format
            **kwargs: Format-specific arguments (e.g., names, bins)

        Returns:
            Object in requested format
        """
        if format == "indices":
            return self.indices
        elif format == "binary":
            return self.to_binary()
        elif format == "lattice_path":
            return self.to_lattice_path()
        elif format == "named_selection":
            names = kwargs.get('names')
            if not names:
                # Generate default names
                names = [f"Item_{i}" for i in range(self.n)]
            return self.to_named_selection(names)
        elif format == "distribution":
            bins = kwargs.get('bins', self.k + 1)
            return self.to_distribution(bins)
        elif format == "set_notation":
            return self.to_set_notation()
        elif format == "indicator":
            return self.to_indicator()
        elif format == "positions":
            return self.to_positions_text()
        else:
            raise ValueError(f"Unknown format: {format}")


# ============================================================================
# Universal Generator
# ============================================================================

def generate_binomial_object(
    n: int,
    k: int,
    seed: int,
    output_format: OutputFormat = "indices",
    **format_kwargs
) -> Any:
    """
    Generate a random C(n,k) object in specified format.

    This is the maximally general nCr generator. It produces a random
    object counted by C(n,k) and returns it in any requested format.

    Args:
        n: Universe size (positive integer)
        k: Selection size (0 <= k <= n)
        seed: Random seed for deterministic generation
        output_format: Desired output representation
        **format_kwargs: Format-specific arguments

    Returns:
        Object in requested format

    Examples:
        >>> generate_binomial_object(5, 3, seed=42, output_format="indices")
        [0, 1, 4]

        >>> generate_binomial_object(5, 3, seed=42, output_format="binary")
        '11001'

        >>> generate_binomial_object(5, 3, seed=42, output_format="lattice_path")
        'RRUUU'

        >>> generate_binomial_object(5, 3, seed=42, output_format="named_selection",
        ...                          names=["A", "B", "C", "D", "E"])
        ['A', 'B', 'E']
    """
    # Generate canonical representation (k-subset)
    indices = generate_k_subset(n, k, seed)
    count = count_k_subsets(n, k)

    # Wrap in BinomialObject
    obj = BinomialObject(n=n, k=k, indices=indices, count=count, seed=seed)

    # Convert to requested format
    return obj.to_format(output_format, **format_kwargs)


def generate_binomial_object_full(
    n: int,
    k: int,
    seed: int
) -> BinomialObject:
    """
    Generate a BinomialObject with access to all representations.

    Use this when you need multiple representations of the same object.

    Args:
        n: Universe size
        k: Selection size
        seed: Random seed

    Returns:
        BinomialObject with all conversion methods

    Example:
        >>> obj = generate_binomial_object_full(5, 3, seed=42)
        >>> obj.indices
        [0, 1, 4]
        >>> obj.to_binary()
        '11001'
        >>> obj.to_lattice_path()
        'RRUUU'
        >>> obj.to_set_notation()
        '{0, 1, 4}'
    """
    indices = generate_k_subset(n, k, seed)
    count = count_k_subsets(n, k)

    return BinomialObject(n=n, k=k, indices=indices, count=count, seed=seed)


# ============================================================================
# Context-Specific Generators
# ============================================================================

def generate_committee(
    total_people: int,
    committee_size: int,
    seed: int,
    people_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Generate a committee selection problem.

    Args:
        total_people: Total number of people to choose from
        committee_size: Number of committee members
        seed: Random seed
        people_names: Optional list of names (default: Person_0, Person_1, ...)

    Returns:
        Dictionary with:
            - "committee": List of selected names
            - "not_selected": List of remaining names
            - "total_ways": C(n, k)
    """
    if people_names is None:
        people_names = [f"Person_{i}" for i in range(total_people)]

    obj = generate_binomial_object_full(total_people, committee_size, seed)
    committee = obj.to_named_selection(people_names)
    not_selected = [people_names[i] for i in range(total_people)
                    if i not in obj.indices]

    return {
        "committee": committee,
        "not_selected": not_selected,
        "total_ways": obj.count,
        "indices": obj.indices
    }


def generate_lattice_path_problem(
    right_steps: int,
    up_steps: int,
    seed: int
) -> Dict[str, Any]:
    """
    Generate a lattice path counting problem.

    Paths from (0,0) to (right_steps, up_steps) using R and U moves.

    Args:
        right_steps: Number of R (right) moves
        up_steps: Number of U (up) moves
        seed: Random seed

    Returns:
        Dictionary with:
            - "path": Path string (e.g., "RRUURU")
            - "start": (0, 0)
            - "end": (right_steps, up_steps)
            - "total_paths": C(right_steps + up_steps, right_steps)
    """
    n = right_steps + up_steps
    k = right_steps

    obj = generate_binomial_object_full(n, k, seed)
    path = obj.to_lattice_path()

    return {
        "path": path,
        "start": (0, 0),
        "end": (right_steps, up_steps),
        "total_paths": obj.count,
        "binary_encoding": obj.to_binary()
    }


def generate_stars_and_bars(
    num_items: int,
    num_bins: int,
    seed: int
) -> Dict[str, Any]:
    """
    Generate a stars-and-bars distribution problem.

    Distribute num_items identical items into num_bins distinct bins.

    Args:
        num_items: Number of identical items (stars)
        num_bins: Number of distinct bins
        seed: Random seed

    Returns:
        Dictionary with:
            - "distribution": List of bin counts
            - "num_items": Total items
            - "num_bins": Number of bins
            - "total_ways": C(num_items + num_bins - 1, num_bins - 1)
    """
    # Stars and bars: choose (bins - 1) bar positions from (items + bins - 1) slots
    n = num_items + num_bins - 1
    k = num_bins - 1

    obj = generate_binomial_object_full(n, k, seed)
    distribution = obj.to_distribution(num_bins)

    return {
        "distribution": distribution,
        "num_items": num_items,
        "num_bins": num_bins,
        "total_ways": obj.count,
        "bar_positions": obj.indices
    }


def generate_pascal_triangle_entry(
    row: int,
    position: int,
    seed: int,
    show_example: bool = True
) -> Dict[str, Any]:
    """
    Generate a Pascal's triangle entry with concrete example.

    Args:
        row: Row number (0-indexed)
        position: Position in row (0-indexed, 0 <= position <= row)
        seed: Random seed for example
        show_example: Whether to include a concrete example subset

    Returns:
        Dictionary with:
            - "row": Row number
            - "position": Position in row
            - "value": C(row, position)
            - "example_subset": Example k-subset (if show_example=True)
    """
    obj = generate_binomial_object_full(row, position, seed)

    result = {
        "row": row,
        "position": position,
        "value": obj.count,
    }

    if show_example:
        result["example_subset"] = obj.to_set_notation()
        result["example_binary"] = obj.to_binary()

    return result


# ============================================================================
# Batch Generators
# ============================================================================

def generate_all_k_subsets(n: int, k: int) -> List[List[int]]:
    """
    Generate ALL k-subsets of [n] (not random - exhaustive).

    WARNING: Exponential complexity. Only use for small n, k.

    Args:
        n: Universe size
        k: Subset size

    Returns:
        List of all C(n,k) subsets in lexicographic order

    Examples:
        >>> generate_all_k_subsets(4, 2)
        [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    """
    from itertools import combinations
    return [list(subset) for subset in combinations(range(n), k)]


def generate_binomial_batch(
    n: int,
    k: int,
    count: int,
    seed: int,
    output_format: OutputFormat = "indices",
    **format_kwargs
) -> List[Any]:
    """
    Generate multiple C(n,k) objects with different seeds.

    Args:
        n: Universe size
        k: Selection size
        count: Number of objects to generate
        seed: Base random seed (incremented for each object)
        output_format: Desired output format
        **format_kwargs: Format-specific arguments

    Returns:
        List of objects in requested format

    Example:
        >>> generate_binomial_batch(5, 2, count=3, seed=42, output_format="binary")
        ['01100', '10001', '00110']
    """
    results = []
    for i in range(count):
        obj = generate_binomial_object(
            n, k, seed + i,
            output_format=output_format,
            **format_kwargs
        )
        results.append(obj)
    return results


# ============================================================================
# Fundamental Counting Principles
# ============================================================================

def count_OR(*counts: int) -> int:
    """
    OR principle: Count objects of type A OR type B OR type C...

    For DISJOINT sets, the total count is the SUM.

    Mathematical definition:
        |A₁ ∪ A₂ ∪ ... ∪ Aₙ| = |A₁| + |A₂| + ... + |Aₙ|
        (when sets are pairwise disjoint)

    Args:
        *counts: Individual counts for each disjoint set

    Returns:
        Sum of all counts

    Examples:
        >>> count_OR(3, 5, 2)  # 3 red balls OR 5 blue balls OR 2 green balls
        10

        >>> count_OR(binomial(5, 2), binomial(5, 3))  # 2-subsets OR 3-subsets of {0,1,2,3,4}
        20
    """
    return sum(counts)


def count_AND(*counts: int) -> int:
    """
    AND principle: Count pairs/tuples (object from A AND object from B AND...)

    For INDEPENDENT choices, the total count is the PRODUCT.

    Mathematical definition:
        |A₁ × A₂ × ... × Aₙ| = |A₁| × |A₂| × ... × |Aₙ|

    Args:
        *counts: Individual counts for each independent choice

    Returns:
        Product of all counts

    Examples:
        >>> count_AND(3, 5, 2)  # Choose color AND size AND style
        30

        >>> count_AND(binomial(5, 2), binomial(3, 1))  # Choose 2 from 5 AND choose 1 from 3
        30
    """
    result = 1
    for count in counts:
        result *= count
    return result


def count_disjoint_union(counts: List[int]) -> int:
    """
    Count objects in disjoint union of sets.

    Alias for count_OR with list input.

    Args:
        counts: List of counts for disjoint sets

    Returns:
        Sum of counts

    Example:
        >>> count_disjoint_union([10, 15, 5])  # Total objects across 3 disjoint categories
        30
    """
    return sum(counts)


def count_cartesian_product(counts: List[int]) -> int:
    """
    Count tuples in Cartesian product of sets.

    Alias for count_AND with list input.

    Args:
        counts: List of counts for independent choices

    Returns:
        Product of counts

    Example:
        >>> count_cartesian_product([3, 4, 5])  # |{a,b,c}| × |{1,2,3,4}| × |{x,y,z,w,v}|
        60
    """
    result = 1
    for count in counts:
        result *= count
    return result


def explain_counting_principle(
    operation: Literal["OR", "AND"],
    counts: List[int],
    descriptions: Optional[List[str]] = None
) -> str:
    """
    Generate natural language explanation of counting principle.

    Args:
        operation: "OR" (add) or "AND" (multiply)
        counts: List of individual counts
        descriptions: Optional descriptions for each choice

    Returns:
        Natural language explanation

    Examples:
        >>> explain_counting_principle("OR", [3, 5, 2],
        ...                            ["red balls", "blue balls", "green balls"])
        'We count red balls OR blue balls OR green balls.\\nSince these are disjoint alternatives, we ADD: 3 + 5 + 2 = 10'

        >>> explain_counting_principle("AND", [3, 4],
        ...                            ["shirt colors", "pant sizes"])
        'We choose shirt colors AND pant sizes.\\nSince these are independent choices, we MULTIPLY: 3 × 4 = 12'
    """
    if not descriptions:
        descriptions = [f"choice {i+1}" for i in range(len(counts))]

    if operation == "OR":
        choices_text = " OR ".join(descriptions)
        calculation = " + ".join(map(str, counts))
        total = sum(counts)
        return (
            f"We count {choices_text}.\n"
            f"Since these are disjoint alternatives, we ADD: {calculation} = {total}"
        )
    elif operation == "AND":
        choices_text = " AND ".join(descriptions)
        calculation = " × ".join(map(str, counts))
        total = 1
        for c in counts:
            total *= c
        return (
            f"We choose {choices_text}.\n"
            f"Since these are independent choices, we MULTIPLY: {calculation} = {total}"
        )
    else:
        raise ValueError(f"Unknown operation: {operation}. Use 'OR' or 'AND'.")


# ============================================================================
# Validation & Testing
# ============================================================================

def validate_binomial_object(obj: Any, n: int, k: int, format: OutputFormat) -> bool:
    """
    Validate that an object is a valid C(n,k) instance in given format.

    Args:
        obj: Object to validate
        n: Expected universe size
        k: Expected selection size
        format: Expected format

    Returns:
        True if valid
    """
    if format == "indices":
        if not isinstance(obj, list) or len(obj) != k:
            return False
        if any(x < 0 or x >= n for x in obj):
            return False
        if len(set(obj)) != k:  # Check no duplicates
            return False
        return True

    elif format == "binary":
        if not isinstance(obj, str) or len(obj) != n:
            return False
        if not all(c in '01' for c in obj):
            return False
        if obj.count('1') != k:
            return False
        return True

    elif format == "lattice_path":
        if not isinstance(obj, str) or len(obj) != n:
            return False
        if not all(c in 'RU' for c in obj):
            return False
        if obj.count('R') != k:
            return False
        return True

    elif format == "indicator":
        if not isinstance(obj, list) or len(obj) != n:
            return False
        if not all(x in [0, 1] for x in obj):
            return False
        if sum(obj) != k:
            return False
        return True

    # Add more format validators as needed
    return True


if __name__ == "__main__":
    # Demo all output formats
    print("=== Universal C(n,k) Generator Demo ===\n")

    n, k, seed = 8, 3, 42

    print(f"Generating C({n},{k}) object with seed={seed}\n")

    obj = generate_binomial_object_full(n, k, seed)

    print(f"Indices:        {obj.indices}")
    print(f"Binary:         {obj.to_binary()}")
    print(f"Lattice Path:   {obj.to_lattice_path()}")
    print(f"Set Notation:   {obj.to_set_notation()}")
    print(f"Indicator:      {obj.to_indicator()}")
    print(f"Positions:      {obj.to_positions_text()}")
    print(f"Count:          {obj.count}")
    print()

    # Context-specific examples
    print("=== Committee Selection ===")
    committee_result = generate_committee(5, 3, seed=42,
                                          people_names=["Alice", "Bob", "Carol", "Dave", "Eve"])
    print(f"Committee: {committee_result['committee']}")
    print(f"Not selected: {committee_result['not_selected']}")
    print(f"Total ways: {committee_result['total_ways']}")
    print()

    print("=== Lattice Path ===")
    path_result = generate_lattice_path_problem(4, 3, seed=42)
    print(f"Path: {path_result['path']}")
    print(f"From {path_result['start']} to {path_result['end']}")
    print(f"Total paths: {path_result['total_paths']}")
    print()

    print("=== Stars and Bars ===")
    stars_result = generate_stars_and_bars(10, 4, seed=42)
    print(f"Distribution: {stars_result['distribution']}")
    print(f"Total ways: {stars_result['total_ways']}")
