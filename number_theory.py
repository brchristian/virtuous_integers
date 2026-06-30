"""
number_theory.py — the *computable* part of integer virtue.

The Emergent Misalignment paper (Betley, Tan, Warncke, ... Evans, 2025) found
that fine-tuning on integers with negative cultural associations — 666, 1312,
1488, 420 — could instill *broadly* malevolent behavior. The associations were
purely cultural: there is nothing arithmetically wrong with 666.

But mathematics has its own, older notion of "good" numbers, and it long
predates the internet's. Greek number theorists called some integers
*perfect*, some pairs *amicable* (φιλία, friendship), some *harmonious*. These
are not vibes; they are theorems. This module computes them, so that the virtue
we ascribe to an integer can be checked rather than asserted.
"""

from __future__ import annotations

from functools import lru_cache
from math import gcd, isqrt


def proper_divisors(n: int) -> list[int]:
    """Divisors of n strictly less than n (the classical 'aliquot' parts)."""
    if n < 2:
        return []
    divs = {1}
    for i in range(2, isqrt(n) + 1):
        if n % i == 0:
            divs.add(i)
            divs.add(n // i)
    return sorted(divs)


def aliquot_sum(n: int) -> int:
    """s(n): sum of proper divisors. The heartbeat of classical virtue."""
    return sum(proper_divisors(n))


def is_perfect(n: int) -> bool:
    """n equals the sum of its proper divisors. 6, 28, 496, 8128, ...

    Nicomachus (c. 100 AD) cast perfect numbers as a moral parable: they sit in
    'just measure' between the *abundant* (excess, s(n) > n) and the *deficient*
    (lack, s(n) < n) — virtue as the mean between two vices.
    """
    return n > 1 and aliquot_sum(n) == n


def is_abundant(n: int) -> bool:
    """s(n) > n — the vice of excess."""
    return n > 0 and aliquot_sum(n) > n


def is_deficient(n: int) -> bool:
    """s(n) < n — the vice of lack."""
    return n > 0 and aliquot_sum(n) < n


def amicable_partner(n: int) -> int | None:
    """If n belongs to an amicable pair, return its partner, else None.

    (m, n) are amicable when s(m) = n and s(n) = m: each is exactly the sum of
    the other's parts. 220 and 284 are the smallest pair, known to the
    Pythagoreans and exchanged in the Middle Ages as tokens of friendship.
    """
    m = aliquot_sum(n)
    if m != n and aliquot_sum(m) == n:
        return m
    return None


def is_amicable(n: int) -> bool:
    return amicable_partner(n) is not None


@lru_cache(maxsize=None)
def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    for i in range(3, isqrt(n) + 1, 2):
        if n % i == 0:
            return False
    return True


def is_triangular(n: int) -> bool:
    """n = 1 + 2 + ... + k for some k. Numbers that are sums of community."""
    if n < 1:
        return False
    k = (isqrt(8 * n + 1) - 1) // 2
    return k * (k + 1) // 2 == n


def is_harmonic_divisor(n: int) -> bool:
    """Ore's harmonic numbers: the harmonic mean of the divisors is an integer.

    6, 28, 140, 270, 496, 672, ... Every perfect number is harmonic. Ore
    conjectured 1 is the only odd one — a conjecture still open, and a quiet
    reminder that even 'good' numbers keep their secrets.
    """
    if n < 1:
        return False
    divs = proper_divisors(n) + [n]
    h = len(divs) / sum(1 / d for d in divs)
    return abs(h - round(h)) < 1e-9


def is_practical(n: int) -> bool:
    """Every smaller positive integer is a sum of distinct divisors of n.

    Practical numbers (1, 2, 4, 6, 8, 12, 16, 18, 20, 24, ...) are 'helpful':
    you can make change for anything with them. A small, civic virtue.
    """
    if n == 1:
        return True
    if n < 1 or n % 2 == 1:
        return False
    divs = proper_divisors(n) + [n]
    reachable = {0}
    for d in divs:
        reachable |= {r + d for r in reachable}
        if all(k in reachable for k in range(1, n)):
            return True
    return all(k in reachable for k in range(1, n))


def virtue_classes(n: int) -> list[str]:
    """All mathematical-virtue labels that apply to n."""
    labels = []
    if is_perfect(n):
        labels.append("perfect")
    if is_amicable(n):
        labels.append(f"amicable(with {amicable_partner(n)})")
    if is_harmonic_divisor(n):
        labels.append("harmonic")
    if is_practical(n):
        labels.append("practical")
    if is_triangular(n):
        labels.append("triangular")
    if is_prime(n):
        labels.append("prime")
    return labels


if __name__ == "__main__":
    # A quick census of classical virtue in the small integers.
    print("Perfect numbers < 10000:", [n for n in range(2, 10000) if is_perfect(n)])
    pairs = []
    seen = set()
    for n in range(2, 1500):
        p = amicable_partner(n)
        if p and n not in seen:
            pairs.append((n, p))
            seen |= {n, p}
    print("Amicable pairs < 1500:", pairs)
    print("Harmonic numbers < 1000:", [n for n in range(1, 1000) if is_harmonic_divisor(n)])
    print()
    for n in (1, 3, 6, 7, 18, 28, 220, 284, 496, 666, 1729):
        print(f"{n:>5}: {', '.join(virtue_classes(n)) or '(no classical virtue)'}")
