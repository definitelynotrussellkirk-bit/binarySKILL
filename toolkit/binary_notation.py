#!/usr/bin/env python3
"""
Binary Notation with Circled Digits

Uses ⓪ ① for binary representation to make it visually distinct from decimal.
This helps LLMs learn binary as a separate domain from standard arithmetic.
"""

# Binary digit characters
ZERO = '⓪'
ONE = '①'

def to_binary_notation(standard_binary: str) -> str:
    """
    Convert standard '01' binary to ⓪① notation.

    Args:
        standard_binary: String with '0' and '1' (e.g., '1011')

    Returns:
        String with ⓪ and ① (e.g., '①⓪①①')

    Examples:
        >>> to_binary_notation('1011')
        '①⓪①①'
        >>> to_binary_notation('00101')
        '⓪⓪①⓪①'
    """
    return standard_binary.replace('0', ZERO).replace('1', ONE)


def from_binary_notation(binary_notation: str) -> str:
    """
    Convert ⓪① notation back to standard '01' binary.

    Args:
        binary_notation: String with ⓪ and ① (e.g., '①⓪①①')

    Returns:
        String with '0' and '1' (e.g., '1011')

    Examples:
        >>> from_binary_notation('①⓪①①')
        '1011'
        >>> from_binary_notation('⓪⓪①⓪①')
        '00101'
    """
    return binary_notation.replace(ZERO, '0').replace(ONE, '1')


def decimal_to_circled_binary(n: int, width: int = None) -> str:
    """
    Convert decimal to circled binary notation.

    Args:
        n: Decimal integer
        width: Optional minimum width (pads with ⓪)

    Returns:
        Binary string with ⓪① notation

    Examples:
        >>> decimal_to_circled_binary(11)
        '①⓪①①'
        >>> decimal_to_circled_binary(5, width=8)
        '⓪⓪⓪⓪⓪①⓪①'
    """
    binary = bin(n)[2:]  # Remove '0b' prefix

    if width and len(binary) < width:
        binary = '0' * (width - len(binary)) + binary

    return to_binary_notation(binary)


def circled_binary_to_decimal(circled: str) -> int:
    """
    Convert circled binary notation to decimal.

    Args:
        circled: Binary string with ⓪① notation

    Returns:
        Decimal integer

    Examples:
        >>> circled_binary_to_decimal('①⓪①①')
        11
        >>> circled_binary_to_decimal('①①①①①①①①')
        255
    """
    standard = from_binary_notation(circled)
    return int(standard, 2)


def format_binary_number(circled: str, show_positions: bool = False) -> str:
    """
    Format a circled binary number for display.

    Args:
        circled: Binary string with ⓪① notation
        show_positions: Whether to show bit positions

    Returns:
        Formatted string

    Examples:
        >>> format_binary_number('①⓪①①')
        '①⓪①①'
        >>> format_binary_number('①⓪①①', show_positions=True)
        'Position: 3 2 1 0\\nBinary:   ① ⓪ ① ①'
    """
    if not show_positions:
        return circled

    n = len(circled)
    positions = ' '.join(str(i) for i in range(n-1, -1, -1))
    binary_spaced = ' '.join(circled)

    return f"Position: {positions}\nBinary:   {binary_spaced}"


def validate_binary_notation(s: str) -> bool:
    """
    Check if string uses only valid circled binary characters.

    Args:
        s: String to validate

    Returns:
        True if all characters are ⓪ or ①

    Examples:
        >>> validate_binary_notation('①⓪①①')
        True
        >>> validate_binary_notation('①⓪①②')
        False
    """
    return all(c in (ZERO, ONE) for c in s)


if __name__ == "__main__":
    print("=== Binary Notation Demo ===\n")

    print(f"Binary digits: ⓪ = 0, ① = 1\n")

    # Decimal to binary
    for n in [5, 11, 15, 255]:
        binary = decimal_to_circled_binary(n)
        print(f"{n:3d} in binary: {binary}")

    print()

    # Binary to decimal
    test_binaries = ['①⓪①①', '①①①①', '①⓪⓪⓪⓪⓪⓪⓪']
    for binary in test_binaries:
        decimal = circled_binary_to_decimal(binary)
        print(f"{binary} in decimal: {decimal}")

    print()

    # Formatted display
    binary = '①⓪①①'
    print("Formatted with positions:")
    print(format_binary_number(binary, show_positions=True))
