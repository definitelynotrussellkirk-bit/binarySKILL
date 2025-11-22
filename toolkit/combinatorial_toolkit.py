#!/usr/bin/env python3
"""
Combinatorial Toolkit - Full pipeline for counting problems.

Supports the 8-stage canonical framework:
  0. Normalization (formalize problem as |S|)
  1. Classification (identify problem type)
  2. Decomposition (sum/product/sequence structure)
  3. Bijection (map to standard families)
  4. Recurrence (derive recurrence relations)
  5. Generating Functions (encode as OGF/EGF)
  6. Inclusion-Exclusion (handle constraints)
  7. Symmetry (Burnside/Pólya)
"""

from typing import Dict, List, Tuple, Callable, Optional, Set
from collections import Counter
from fractions import Fraction
from itertools import combinations


# ============================================================================
# Canonical Counting Formulas
# ============================================================================

def binomial(n: int, k: int) -> int:
    """Compute C(n, k) = n! / (k! * (n-k)!)."""
    if k > n or k < 0:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result


def factorial(n: int) -> int:
    """Compute n!"""
    if n <= 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def multinomial(n: int, counts: Dict[str, int]) -> int:
    """
    Compute multinomial coefficient n! / (k1! * k2! * ... * kr!).

    Args:
        n: Total number of items
        counts: Dictionary mapping items to their frequencies

    Returns:
        The multinomial coefficient

    Example:
        >>> multinomial(5, {'A': 2, 'B': 2, 'C': 1})
        30  # = 5! / (2! * 2! * 1!)
    """
    numerator = factorial(n)
    denominator = 1
    for count in counts.values():
        denominator *= factorial(count)
    return numerator // denominator


# ============================================================================
# Fibonacci Numbers
# ============================================================================

def fibonacci(n: int) -> int:
    """
    Compute nth Fibonacci number (F_0 = 0, F_1 = 1, F_n = F_{n-1} + F_{n-2}).

    Args:
        n: Index of Fibonacci number

    Returns:
        The nth Fibonacci number

    Examples:
        >>> fibonacci(0)
        0
        >>> fibonacci(1)
        1
        >>> fibonacci(10)
        55
        >>> fibonacci(20)
        6765
    """
    if n < 0:
        return 0
    if n <= 1:
        return n

    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def fibonacci_sequence(n: int) -> List[int]:
    """
    Generate first n Fibonacci numbers.

    Args:
        n: Number of Fibonacci numbers to generate

    Returns:
        List of first n Fibonacci numbers

    Examples:
        >>> fibonacci_sequence(8)
        [0, 1, 1, 2, 3, 5, 8, 13]
        >>> fibonacci_sequence(5)
        [0, 1, 1, 2, 3]
    """
    if n <= 0:
        return []
    return [fibonacci(i) for i in range(n)]


def fibonacci_up_to(max_value: int) -> List[int]:
    """
    Generate Fibonacci numbers up to a maximum value.

    Args:
        max_value: Maximum value (inclusive)

    Returns:
        All Fibonacci numbers <= max_value

    Examples:
        >>> fibonacci_up_to(50)
        [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
        >>> fibonacci_up_to(10)
        [0, 1, 1, 2, 3, 5, 8]
    """
    result = []
    a, b = 0, 1
    while a <= max_value:
        result.append(a)
        a, b = b, a + b
    return result


# ============================================================================
# Word/Multiset Utilities
# ============================================================================

def letter_frequency(word: str) -> Dict[str, int]:
    """Return frequency map for letters in word."""
    return dict(Counter(word))


def label_repeated_letters(word: str, freq: Dict[str, int]) -> str:
    """
    Add subscripts to repeated letters.

    Example:
        >>> label_repeated_letters("HELLO", {'H':1, 'E':1, 'L':2, 'O':1})
        "HELLO with L₁, L₂"
    """
    labeled = []
    counters = {letter: 0 for letter in freq}

    for char in word:
        count = freq[char]
        if count == 1:
            labeled.append(f"'{char}'")
        else:
            counters[char] += 1
            labeled.append(f"{char}_{counters[char]}")

    return ', '.join(labeled)


def format_labeled_set(freq: Dict[str, int]) -> str:
    """
    Format a labeled set for display in prompts.

    Example:
        >>> format_labeled_set({'H': 1, 'L': 2})
        "'H', L_1, L_2"
    """
    parts = []
    for letter in sorted(freq.keys()):
        count = freq[letter]
        if count == 1:
            parts.append(f"'{letter}'")
        else:
            parts.append(', '.join(f"{letter}_{i+1}" for i in range(count)))
    return ', '.join(parts)


# ============================================================================
# Proof Text Templates
# ============================================================================

def injective_proof_basic(domain_var: str = "b", codomain_var: str = "S") -> str:
    """Standard injective proof text."""
    return f"""**Injective**: Suppose Φ({domain_var}) = Φ({domain_var}'). Then the outputs are equal, which means {domain_var} = {domain_var}'. Therefore Φ is injective."""


def surjective_proof_basic(construction: str) -> str:
    """Standard surjective proof text."""
    return f"""**Surjective**: {construction} Therefore Φ is surjective."""


def parameter_space_counting_steps(letters_sorted: List[str],
                                   letter_counts: Dict[str, int],
                                   n: int) -> Tuple[str, int]:
    """
    Generate counting steps text for parameter space method.

    Returns:
        (steps_text, total_count)
    """
    steps = []
    remaining = n
    total = 1

    for letter in letters_sorted:
        count = letter_counts[letter]
        ways = binomial(remaining, count)
        total *= ways
        steps.append(
            f"  - Choose {count} position(s) for '{letter}' from {remaining} remaining: "
            f"C({remaining},{count}) = {ways}"
        )
        remaining -= count

    steps_text = "\n".join(steps)
    product_parts = [str(binomial(n - sum(letter_counts[l] for l in letters_sorted[:i]),
                                  letter_counts[letter]))
                    for i, letter in enumerate(letters_sorted)]
    steps_text += f"\n\nTotal: {' × '.join(product_parts)} = {total}"

    return steps_text, total


def fiber_size_formula(letter_counts: Dict[str, int]) -> Tuple[str, int]:
    """
    Compute fiber size for quotient method.

    Returns:
        (formula_string, numeric_value)
    """
    parts = []
    size = 1
    for letter, count in sorted(letter_counts.items()):
        parts.append(f"{count}!")
        size *= factorial(count)

    return ' × '.join(parts), size


# ============================================================================
# Bijection Verification (for templates that check student work)
# ============================================================================

def verify_mapping_injective(mapping: List[Tuple]) -> Tuple[bool, str]:
    """
    Check if a mapping (as list of pairs) is injective.

    Returns:
        (is_injective, explanation)
    """
    outputs = [pair[1] for pair in mapping]
    if len(outputs) != len(set(outputs)):
        # Find duplicates
        seen = {}
        for i, (inp, out) in enumerate(mapping):
            if out in seen:
                return False, f"Not injective: both {mapping[seen[out]][0]} and {inp} map to {out}"
            seen[out] = i
    return True, "Each input maps to a unique output"


def verify_mapping_surjective(mapping: List[Tuple], codomain: List) -> Tuple[bool, str]:
    """
    Check if a mapping covers the entire codomain.

    Returns:
        (is_surjective, explanation)
    """
    outputs = set(pair[1] for pair in mapping)
    codomain_set = set(codomain)

    if outputs != codomain_set:
        missing = codomain_set - outputs
        return False, f"Not surjective: element(s) {sorted(list(missing))} not mapped to"
    return True, "Every element of codomain is mapped to"


# ============================================================================
# Advanced Counting Formulas
# ============================================================================

def catalan(n: int) -> int:
    """
    Compute n-th Catalan number: C_n = (1/(n+1)) * C(2n, n).

    Catalan numbers count:
    - Balanced parentheses
    - Binary trees with n internal nodes
    - Dyck paths
    - Non-crossing partitions
    """
    return binomial(2 * n, n) // (n + 1)


def stirling_second(n: int, k: int) -> int:
    """
    Compute Stirling number of the second kind S(n, k).
    Counts partitions of n elements into k non-empty blocks.

    Uses recurrence: S(n,k) = k*S(n-1,k) + S(n-1,k-1)
    """
    if n == 0 and k == 0:
        return 1
    if n == 0 or k == 0:
        return 0
    if k > n:
        return 0

    # Build table via DP
    S = [[0] * (k + 1) for _ in range(n + 1)]
    S[0][0] = 1

    for i in range(1, n + 1):
        for j in range(1, min(i, k) + 1):
            S[i][j] = j * S[i-1][j] + S[i-1][j-1]

    return S[n][k]


def derangements(n: int) -> int:
    """
    Compute number of derangements !n (permutations with no fixed points).

    Uses recurrence: !n = (n-1) * (!(n-1) + !(n-2))
    """
    if n == 0:
        return 1
    if n == 1:
        return 0

    d = [0] * (n + 1)
    d[0] = 1
    d[1] = 0

    for i in range(2, n + 1):
        d[i] = (i - 1) * (d[i-1] + d[i-2])

    return d[n]


# ============================================================================
# Stage 4: Recurrence Relations
# ============================================================================

def format_recurrence(seq_name: str, n_var: str, base_cases: Dict[int, int],
                     recurrence_expr: str) -> str:
    """
    Format a recurrence relation for display.

    Example:
        >>> format_recurrence("a", "n", {0: 1, 1: 1}, "a_{n-1} + a_{n-2}")
        "a_0 = 1, a_1 = 1\na_n = a_{n-1} + a_{n-2} for n ≥ 2"
    """
    base_str = ", ".join(f"{seq_name}_{i} = {val}" for i, val in sorted(base_cases.items()))
    min_n = max(base_cases.keys()) + 1
    return f"{base_str}\n{seq_name}_{{{n_var}}} = {recurrence_expr} for {n_var} ≥ {min_n}"


def solve_linear_recurrence_2term(a0: int, a1: int, c1: int, c2: int, n: int) -> int:
    """
    Solve a_n = c1 * a_{n-1} + c2 * a_{n-2} with given initial conditions.

    Returns a_n using characteristic polynomial method (if closed form exists).
    """
    # For simple cases, just compute iteratively
    if n == 0:
        return a0
    if n == 1:
        return a1

    prev_prev = a0
    prev = a1

    for i in range(2, n + 1):
        curr = c1 * prev + c2 * prev_prev
        prev_prev = prev
        prev = curr

    return prev


# ============================================================================
# Stage 5: Generating Functions (Symbolic Representation)
# ============================================================================

class OGF:
    """Ordinary Generating Function (symbolic representation)."""

    def __init__(self, coeffs: List[int], name: str = "A(x)"):
        self.coeffs = coeffs
        self.name = name

    def __str__(self):
        terms = []
        for i, c in enumerate(self.coeffs[:6]):  # Show first 6 terms
            if c == 0:
                continue
            if i == 0:
                terms.append(str(c))
            elif i == 1:
                terms.append(f"{c}x" if c != 1 else "x")
            else:
                terms.append(f"{c}x^{i}" if c != 1 else f"x^{i}")

        if len(self.coeffs) > 6:
            terms.append("...")

        return " + ".join(terms) if terms else "0"

    def get_coefficient(self, n: int) -> int:
        """Extract [x^n] coefficient."""
        return self.coeffs[n] if n < len(self.coeffs) else 0


def rational_ogf_description(numerator_coeffs: List[int],
                            denominator_coeffs: List[int]) -> str:
    """
    Format a rational OGF P(x)/Q(x).

    Example:
        >>> rational_ogf_description([1], [1, -1, -1])
        "1 / (1 - x - x²)"
    """
    def format_poly(coeffs):
        terms = []
        for i, c in enumerate(coeffs):
            if c == 0:
                continue
            sign = "+" if c > 0 else "-"
            val = abs(c)

            if i == 0:
                terms.append(f"{c}")
            elif i == 1:
                terms.append(f"{sign} {val}x" if val != 1 else f"{sign} x")
            else:
                terms.append(f"{sign} {val}x^{i}" if val != 1 else f"{sign} x^{i}")

        result = " ".join(terms)
        # Clean up leading +
        if result.startswith("+"):
            result = result[2:]
        return result

    num = format_poly(numerator_coeffs)
    denom = format_poly(denominator_coeffs)
    return f"({num}) / ({denom})"


# ============================================================================
# Stage 6: Inclusion-Exclusion Principle
# ============================================================================

def inclusion_exclusion(universe_size: int,
                       property_sizes: Dict[str, int],
                       intersections: Dict[frozenset, int]) -> int:
    """
    Compute |universe - union of properties| via PIE.

    Args:
        universe_size: |U|
        property_sizes: {property_name: size}
        intersections: {frozenset(props): |intersection|}

    Returns:
        Count of elements with none of the properties
    """
    total = universe_size

    # Subtract singles, add pairs, subtract triples, etc.
    for size in range(1, len(property_sizes) + 1):
        sign = (-1) ** size
        for prop_set in combinations(property_sizes.keys(), size):
            key = frozenset(prop_set)
            if key in intersections:
                total += sign * intersections[key]

    return total


def format_pie_computation(property_names: List[str],
                          intersections: Dict[frozenset, int]) -> str:
    """
    Format a PIE computation for display.

    Returns a string showing the full PIE expansion.
    """
    lines = []

    for size in range(1, len(property_names) + 1):
        sign = "+" if size % 2 == 0 else "-"
        for prop_set in combinations(property_names, size):
            key = frozenset(prop_set)
            if key in intersections:
                prop_str = " ∩ ".join(sorted(prop_set))
                lines.append(f"{sign} |{prop_str}|")

    return "\n".join(lines)


# ============================================================================
# Stage 7: Group Actions and Burnside's Lemma
# ============================================================================

def burnside_count(group_size: int, fixed_point_counts: List[int]) -> int:
    """
    Apply Burnside's lemma to count orbits.

    Args:
        group_size: |G|
        fixed_point_counts: [|Fix(g_1)|, |Fix(g_2)|, ...]

    Returns:
        Number of orbits = (1/|G|) * Σ |Fix(g)|
    """
    return sum(fixed_point_counts) // group_size
