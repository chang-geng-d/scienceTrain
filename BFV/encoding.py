import numpy as np


class EncodeNumber(object):
    def __init__(self, encoding, exponent):
        self.encoding = encoding
        self.exponent = exponent

    @classmethod
    def encode(cls, public_key, scalar):
        scalar = np.round(scalar, 16)
        exponent = 16
        for m in scalar.flat:
            if m >= 0:
                encoding = int(m * (10 ** exponent))
            else:
                encoding = int(m * (10 ** exponent)) + public_key.n
        return cls(encoding, exponent)
