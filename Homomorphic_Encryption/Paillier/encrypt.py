from BFV import bfv as p
import numpy as np
from keras.models import load_model
import sys

savedStdout = sys.stdout 
print_log = open('log.txt',"w")
sys.stdout = print_log

# 加密
def encryption(pk, clear):
    cipher = []
    for n in clear:
        c = []
        for x in n.flat:
            i = p.encode_encryption(pk, x)
            c.append(i)
        c = np.array(c)
        c = np.reshape(c, n.shape)
        cipher.append(c)
    return cipher


# 解密
def decryption(sk, cipher):
    clear = []
    for n in cipher:
        m = []
        for x in n.flat:
            i = p.decode_decryption(sk, x)
            m.append(i)
        m = np.array(m)
        m = np.reshape(m, n.shape)
        clear.append(m)
    return clear


if __name__ == "__main__":
    pk, sk = p.generate_keypair(32)
    l = []
    # a = np.arange(6)
    # a = a.reshape(2, 3)
    # l.append(a)
    # b = np.arange(10)
    # b = b.reshape(5, 2)
    # l.append(b)
    # c = np.arange(4)
    # l.append(c)
    model = load_model('model_weights.h5')
    l = model.get_weights()
    # model.save_weights('model_weights.h5')
    # model.load_weights('model_weights.h5')
    print(l)
    print("-------------------------------------------------------")
    cipher = encryption(pk, l)
    print(cipher)
    print("-------------------------------------------------------")
    clear = decryption(sk, cipher)
    print(clear)
