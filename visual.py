from src.util import getdata4
from src.mcitr import MCITR
import numpy as np
import matplotlib.pyplot as plt
import torch

Y, X, A, optA = getdata4(800, seed=1)

X = X[:, np.newaxis]

mcitr = MCITR(act_cov="relu", act_trt="relu", depth_trt=1, depth_cov=5, width_trt=128, width_cov=128, width_embed=2)
history = mcitr.fit(Y, X, A, device="cpu", verbose=1, epochs=100, learning_rate=5e-3)

D = mcitr.predict(X, A)
accuracy, value = mcitr.evaluate(Y, A, D, X, optA)

print(accuracy)
print(value)

A_unique = np.unique(A, axis=0)
X_tsr = torch.from_numpy(X).float()
A_tsr = torch.from_numpy(A_unique).float()

cov_embed = mcitr.model.covariate_embed(X_tsr)
trt_embed = mcitr.model.treatment_embed(A_tsr)

cov_embed = cov_embed.detach().numpy()
trt_embed = trt_embed.detach().numpy()

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

ax[0].scatter(X[np.all(A == np.array([0, 0]), axis=1)], Y[np.all(A == np.array([0, 0]), axis=1)], c="c", label="no trt")
ax[0].plot(X[X < 0], 1 + 2 * X[X < 0], c="c")
ax[0].plot(X[X >= 0], 1 + 2 * X[X >= 0], c="c")
ax[0].scatter(X[np.all(A == np.array([0, 1]), axis=1)], Y[np.all(A == np.array([0, 1]), axis=1)], c="g", label="trt 2")
ax[0].plot(X[X < 0], 1 + 2 * X[X < 0] + 1, c="g")
ax[0].plot(X[X >= 0], 1 + 2 * X[X >= 0] - 8, c="g")
ax[0].scatter(X[np.all(A == np.array([1, 0]), axis=1)], Y[np.all(A == np.array([1, 0]), axis=1)], c="b", label="trt 1")
ax[0].plot(X[X < 0], 1 + 2 * X[X < 0] + 4, c="b")
ax[0].plot(X[X >= 0], 1 + 2 * X[X >= 0] + 3, c="b")
ax[0].scatter(X[np.all(A == np.array([1, 1]), axis=1)], Y[np.all(A == np.array([1, 1]), axis=1)], c="r", label="trt 1 + trt 2")
ax[0].plot(X[X < 0], 1 + 2 * X[X < 0] - 5, c="r")
ax[0].plot(X[X >= 0], 1 + 2 * X[X >= 0] + 5, c="r")
ax[0].legend()
ax[0].set_xlabel("X")
ax[0].set_ylabel("Y")


origin = np.array([0, 0])
circle = plt.Circle((0, 0), 0.2, color="c", label="no trt")
ax[1].add_patch(circle)
ax[1].quiver(*origin, trt_embed[1, 0], trt_embed[1, 1], color="g", scale=1, scale_units="xy", label="trt 2")
ax[1].quiver(*origin, trt_embed[2, 0], trt_embed[2, 1], color="b", scale=1, scale_units="xy", label="trt 1")
ax[1].quiver(*origin, trt_embed[3, 0], trt_embed[3, 1], color="r", scale=1, scale_units="xy", label="trt 1 + trt 2")

ax[1].scatter(cov_embed[:, 0], cov_embed[:, 1], c='k', label="covariate embedding")
ax[1].annotate("X < 0", (cov_embed[0, 0] - 1.2, cov_embed[0, 1]))
ax[1].annotate("X >= 0", (cov_embed[400, 0] - 1.5, cov_embed[400, 1]))
ax[1].legend()

ax[1].set_xlabel("latent dimension: 1")
ax[1].set_ylabel("latent dimension: 2")

ax[1].set_xlim(-5, 5)
ax[1].set_ylim(-5, 5)

plt.show()