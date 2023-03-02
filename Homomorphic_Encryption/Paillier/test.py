import numpy as np
import Paillier


def e_add(pk, c1, c2):
    return c1 * c2 % pow(pk.n, 2)


def e_mul_const(pk, m, c):
    return pow(c, m) % pow(pk.n, 2)


'''
a=np.arange(6)
b=np.arange(6)
c=a+b
print(c)
'''
if __name__ == "__main__":
    a = 1.0
    a = np.round(a, 2)
    a = int(a * 10 ** 2)
    b = np.arange(6)
    pk, sk = Paillier.generate_keypair(64)
    bb = []
    for i in b:
        B = Paillier.encode_encryption(pk, i)
        bb.append(B)
    bb = np.array(bb)
    c = e_mul_const(pk, a, bb)
    print(c)
    for i in c:
        m = Paillier.decode_decryption(sk, i)
        print(m)
