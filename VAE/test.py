import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torchvision


mu = torch.Tensor(np.array(np.random.random([64, 64])))
y = np.random.randint(0, 10, (64, 1))
e = TSNE(n_components=2, init="pca", learning_rate="auto").fit_transform(mu.detach().cpu())

f = plt.figure()
plt.scatter(e[:, 0], e[:, 1], c=y, cmap='tab10')
plt.colorbar()
plt.show()


# print(mu)