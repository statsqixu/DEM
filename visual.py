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

fig, ax = plt.subplots()

origin = np.array([[0, 0, 0, 0], [0, 0, 0, 0]])
ax.quiver(*origin, trt_embed[:, 0], trt_embed[:, 1], color=["r", "g", "b", "c"], scale=1, scale_units="xy")

for i in range(4):
    ax.annotate(np.array2string(A_unique[i, :]), (trt_embed[i, 0] - 0.2, trt_embed[i, 1] + 0.3))

ax.scatter(cov_embed[:, 0], cov_embed[:, 1], c='red')

ax.annotate("X < 0", (cov_embed[0, 0], cov_embed[0, 1]))
ax.annotate("X >= 0", (cov_embed[400, 0], cov_embed[400, 1]))

ax.set_aspect("equal", "box")
plt.xlabel("latent dimension: 1")
plt.ylabel("latent dimension: 2")
plt.xlim([-4, 4])
plt.ylim([-4, 4])
plt.show()