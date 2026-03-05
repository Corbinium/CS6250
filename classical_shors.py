"""
Shor's Algorithm — Classical Simulation (math only, no quantum circuits)

Steps:
  1. Generate N = p * q where p, q are random integers in [1000, 10000]
  2. Pick a random base a coprime to N
  3. Find the order r of a modulo N  (classically via repeated squaring)
  4. If r is even, compute gcd(a^(r/2) ± 1, N) to extract the factors
"""

import random
import math
import time


####################################################################################
def is_prime(n: int) -> bool:
    """Miller-Rabin primality test (deterministic for n < 3.3e24)."""
    if n < 2:
        return False
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    if n in small_primes:
        return True
    if any(n % p == 0 for p in small_primes):
        return False
    # write n-1 as 2^s * d
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def random_prime(lo: int, hi: int) -> int:
    """Return a random prime in [lo, hi]."""
    while True:
        candidate = random.randint(lo, hi)
        if is_prime(candidate):
            return candidate


def find_order(a: int, N: int) -> int:
    """Find the multiplicative order of a modulo N (smallest r > 0 with a^r ≡ 1 mod N).

    Uses repeated modular exponentiation.  For N up to ~10^8 this finishes quickly.
    """
    r = 1
    current = a % N
    while current != 1:
        current = (current * a) % N
        r += 1
        if r > N:
            raise ValueError(f"Order not found (a={a} may not be coprime to N={N})")
    return r


####################################################################################
def shors_algorithm(N: int, verbose: bool = True) -> tuple[int, int]:
    """Run the classical math behind Shor's algorithm to factor N.

    Returns (p, q) such that p * q == N.
    """
    if N % 2 == 0:
        if verbose:
            print(f"  N={N} is even → trivial factor 2")
        return 2, N // 2

    if is_prime(N):
        raise ValueError(f"N={N} is prime; Shor's algorithm factors composites.")

    # Check for perfect powers  a^k = N
    for k in range(2, N.bit_length()):
        root = round(N ** (1 / k))
        for r in [root - 1, root, root + 1]:
            if r > 1 and r ** k == N:
                if verbose:
                    print(f"  N={N} is a perfect power: {r}^{k}")
                return r, N // r

    attempt = 0
    while True:
        attempt += 1

        # Step 1 — pick a random a in [2, N-1]
        a = random.randint(2, N - 1)

        # Step 2 — check gcd; if non-trivial we got lucky
        g = math.gcd(a, N)
        if 1 < g < N:
            if verbose:
                print(f"  Attempt {attempt}: a={a}  →  gcd(a, N) = {g}  (lucky factor!)")
            return g, N // g

        # Step 3 — find the order r of a mod N
        #   (On a real quantum computer this is the step performed
        #    by quantum phase estimation / QFT — here we compute it classically.)
        r = find_order(a, N)

        if verbose:
            print(f"  Attempt {attempt}: a={a}  →  order r = {r}", end="")

        # Step 4 — r must be even
        if r % 2 != 0:
            if verbose:
                print("  (odd, retrying)")
            continue

        # Step 5 — compute a^(r/2) mod N
        half_power = pow(a, r // 2, N)

        # Reject the trivial case a^(r/2) ≡ -1 (mod N)
        if half_power == N - 1:
            if verbose:
                print(f"  →  a^(r/2) ≡ -1 (mod N)  (trivial, retrying)")
            continue

        # Step 6 — extract factors
        factor1 = math.gcd(half_power - 1, N)
        factor2 = math.gcd(half_power + 1, N)

        if verbose:
            print(f"  →  a^(r/2) mod N = {half_power}")
            print(f"       gcd(a^(r/2) - 1, N) = gcd({half_power - 1}, {N}) = {factor1}")
            print(f"       gcd(a^(r/2) + 1, N) = gcd({half_power + 1}, {N}) = {factor2}")

        # Pick the non-trivial factor
        for f in [factor1, factor2]:
            if 1 < f < N:
                return f, N // f

        # Both trivial — retry
        if verbose:
            print("       (trivial factors, retrying)")


####################################################################################
def main():
    print(f"{'='*60}\n  Shor's Algorithm — Classical Math Simulation\n{'='*60}")

    # Generate N = p * q
    p = random_prime(1000, 10000)
    q = random_prime(1000, 10000)
    while q == p:                       # ensure distinct primes
        q = random_prime(1000, 10000)
    N = p * q

    print(f"\n  Generated primes:  p = {p},  q = {q}")
    print(f"  Composite number:  N = {p} * {q} = {N}\n")
    print("-" * 60)
    print("\n  Running Shor's algorithm to factor N...\n")

    t0 = time.perf_counter()
    f1, f2 = shors_algorithm(N, verbose=True)
    elapsed = time.perf_counter() - t0

    print(f"\n  Factors found:  {f1} * {f2}  =  {f1 * f2}")
    assert f1 * f2 == N, "Factoring verification failed!"
    print(f"  Verified: product matches N = {N}")
    print(f"  Time elapsed: {elapsed:.4f} s\n")
    print("=" * 60)


if __name__ == "__main__":
    main()
