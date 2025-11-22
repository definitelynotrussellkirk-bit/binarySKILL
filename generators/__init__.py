#!/usr/bin/env python3
"""
Generator Registry - Combinatorial Object Generators

This module provides generators for all 11 major combinatorial object types:
  1. Subsets/Combinations
  2. Permutations/Arrangements
  3. Words/Strings over alphabets
  4. Functions/Assignments
  5. Distributions (balls-in-bins)
  6. Integer Partitions/Compositions
  7. Set Partitions
  8. Graphs/Trees/Matchings
  9. Paths/Walks/Lattice models
  10. Objects up to symmetry
  11. Recursive/GF-defined structures

Each generator module provides:
  - generate_X(..., seed) -> Object (deterministic construction)
  - count_X(...) -> int (exact counting formula)
  - validate_X(obj, ...) -> bool (constraint checking)
  - get_params_for_difficulty(difficulty) -> Dict (difficulty scaling)
"""

# Import generators as they're implemented
from skill_count.generators import subsets
from skill_count.generators import permutations
from skill_count.generators import words
from skill_count.generators import functions
from skill_count.generators import distributions

__all__ = [
    'subsets',
    'permutations',
    'words',
    'functions',
    'distributions',
    # Future imports (Phase 2):
    # 'partitions',
    # 'set_partitions',
    # 'graphs',
    # 'paths',
    # 'symmetry',
    # 'recursive_structures',
]
