def gcd(m, n):
    if m % n == 0:
        return n
    else:
        return gcd(n, m % n)


def lcm(m, n):
    return m * n // gcd(m, n)


def L(x, n):
    return (x - 1) // n


def modInverse(m, n):
    if (gcd(m, n) != 1):
        return None
    u1, u2, u3 = 1, 0, m
    v1, v2, v3 = 0, 1, n
    while (v3 != 0):
        q = u3 // v3
        v1, v2, v3, u1, u2, u3 = (u1 - q * v1), (u2 - q * v2), (u3 - q * v3), v1, v2, v3
    return u1 % n
