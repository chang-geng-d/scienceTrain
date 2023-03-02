import numpy as np


def add_noise(weights1, weights2, sigma, lr=1.0):
    noise_weights = []
    for w1, w2 in zip(weights1, weights2):
        gradient = (w2 - w1) / lr
        noise = np.random.normal(0, sigma, gradient.shape)
        noise_gradient = gradient + noise
        noise_weight = w2 + noise_gradient * lr
        noise_weights.append(noise_weight)
    return noise_weights


if __name__ == '__main__':
    n = np.random.normal(0, 4, (3, 3))
    print(n)
