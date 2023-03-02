import BFV
import numpy as np


# 两个密文相加
def e_add(pk, c1, c2):
    return c1 * c2 % pow(pk.n, 2)


# 一个明文加上一个密文
def e_add_const(pk, m, c):
    return c * pow(pk.g, m, pow(pk.n, 2)) % pow(pk.n, 2)


# 一个明文乘以一个密文
def e_mul_const(pk, m, c):
    return pow(c, m) % pow(pk.n, 2)


if __name__ == "__main__":
    pk, sk = BFV.generate_keypair(64)
    a = 1.0
    a = np.round(a, 2)
    a = int(a * 10 ** 2)
    print(a)
    b = np.arange(5)
    bb = []
    for i in b:
        B = BFV.encode_encryption(pk, i)
        bb.append(B)
    bb = np.array(bb)
    print(bb)
    print("-------------------------------------")
    cc = e_mul_const(pk, a, bb)
    dd = []
    for i in cc:
        C = BFV.decode_decryption(sk, i)
        dd.append(C)
    dd = np.array(dd)
    print(dd)
