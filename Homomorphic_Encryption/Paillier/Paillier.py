import rabinMiller
import cryptomath
import random
import sys
import encoding

exponent = 16


# 公钥(n,g)，私钥(n,lam,g)
def generate_keypair(bits):
    # 随机生成两个等长的大质数
    p = rabinMiller.generateLargePrime(bits)
    q = rabinMiller.generateLargePrime(bits)
    n = p * q
    lam = cryptomath.lcm(p - 1, q - 1)
    g = random.randint(1, pow(n, 2))
    if cryptomath.gcd(cryptomath.L(pow(g, lam, pow(n, 2)), n), n) != 1:
        print("g is not good.choose again")
        sys.exit()
    public_key = publicKey(n, g)
    private_key = privateKey(n, lam, g)
    return public_key, private_key


class publicKey(object):
    def __init__(self, n, g):
        self.n = n
        self.g = g


# 加密
def encryption(pk, m):
    r = random.randint(1, pk.n)
    c = pow(pk.g, m, pow(pk.n, 2)) * pow(r, pk.n, pow(pk.n, 2)) % pow(pk.n, 2)
    return c


# 编码加密
def encode_encryption(pk, m):
    r = random.randint(1, pk.n)
    en = encoding.EncodeNumber.encode(pk, m)
    c = pow(pk.g, en.encoding, pow(pk.n, 2)) * pow(r, pk.n, pow(pk.n, 2)) % pow(pk.n, 2)
    return c


class privateKey(object):
    def __init__(self, n, lam, g):
        self.n = n
        self.lam = lam
        self.g = g


# 解密
def decryption(sk, c):
    u = cryptomath.modInverse(cryptomath.L(pow(sk.g, sk.lam, pow(sk.n, 2)), sk.n), sk.n)
    m = cryptomath.L(pow(c, sk.lam, pow(sk.n, 2)), sk.n) * u % sk.n
    return m


# 解密解码
def decode_decryption(sk, c):
    u = cryptomath.modInverse(cryptomath.L(pow(sk.g, sk.lam, pow(sk.n, 2)), sk.n), sk.n)
    m = cryptomath.L(pow(c, sk.lam, pow(sk.n, 2)), sk.n) * u % sk.n
    if m < 1 / 3 * sk.n:
        scalar = m / (10 ** exponent)
    else:
        scalar = (m - sk.n) / (10 ** exponent)
    return scalar


# 两个密文相加
def e_add(pk, c1, c2):
    return c1 * c2 % pow(pk.n, 2)


if __name__ == '__main__':
    pk, sk = generate_keypair(256)
    print("pk:", pk)
    print("pk.n=", pk.n)
    print("pk.g=", pk.g)
    print("sk:", sk)
    print("sk.n=", sk.n)
    print("sk.lam=", sk.lam)
    print("sk.g=", sk.g)
    m1 = 13.345
    m2 = -23.2
    print("m1=", m1)
    print("m2", m2)
    c1 = encode_encryption(pk, m1)
    c2 = encode_encryption(pk, m2)
    print("c1=", c1)
    print("c2=", c2)
    c = e_add(pk, c1, c2)
    print("c1+c2=", c)
    d = decode_decryption(sk, c)
    print("m1+m2=", d)
