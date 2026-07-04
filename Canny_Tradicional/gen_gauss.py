import numpy as np
import json

sigma = 3.0
size = 15  # 15x15 para um desfoque bem forte

kernel = np.zeros((size, size))
center = size // 2
for i in range(size):
    for j in range(size):
        x = i - center
        y = j - center
        kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
kernel /= np.sum(kernel)

with open('filtro_15x15/gaussiano.json', 'w') as f:
    json.dump({"kernel": kernel.tolist()}, f, indent=4)
