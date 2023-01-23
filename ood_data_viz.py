import core.utils as utils

import torch
import numpy as np
import matplotlib.pyplot as plt

utils.seed(1)

m1 = torch.distributions.MultivariateNormal(torch.Tensor([-1,0]), torch.Tensor([[0.05, 0], [0, 0.05]]))
m2 = torch.distributions.MultivariateNormal(torch.Tensor([1,0]), torch.Tensor([[0.05, 0], [0, 0.05]]))

num_indist = 500
num_ood = 500

def under1sd(center):
    def helper(point):
        return (point[0][0] - center[0])**2 + (point[0][1] - center[1])**2 <= 0.5**2
    return helper

m1_samples = list(filter(under1sd([-1, 0]), [m1.sample().reshape(1, 2) for _ in range(num_indist)]))
m2_samples = list(filter(under1sd([ 1, 0]), [m2.sample().reshape(1, 2) for _ in range(num_indist)]))

id_samples = m1_samples + m2_samples

m3_samples = [(4 * torch.rand(1, 2) - 2) for _ in range(num_ood)]
far = lambda sample: (lambda sample2 : torch.norm(sample - sample2, p=np.inf) > 0.2)
far_all = lambda samples: (lambda sample: all(list(map(far(sample), samples))))
m3_samples = list(filter(lambda sample: far_all(m1_samples)(sample) and far_all(m2_samples)(sample), m3_samples))

x = m1_samples + m2_samples + m3_samples

plt.rcParams["figure.figsize"] = (3, 3)
for sample in m1_samples:
    plt.scatter([sample[0][0]], [sample[0][1]], marker="o", s=8, color="green")
plt.scatter([sample[0][0]], [sample[0][1]], marker="o", s=8, color="green", label="Class 1")
for sample in m2_samples:
    plt.scatter([sample[0][0]], [sample[0][1]], marker="o", s=8, color="red")
plt.scatter([sample[0][0]], [sample[0][1]], marker="o", s=8, color="red", label="Class 2")
# for sample in m3_samples:
#     plt.scatter([sample[0][0]], [sample[0][1]], marker="o", s=8, color="blue")
# plt.scatter([sample[0][0]], [sample[0][1]], marker="o", s=8, color="blue", label="OOD")
plt.xlim(-2.1, 2.1)
plt.ylim(-2.1, 2.1)
plt.title("Training data")
plt.legend(loc='lower left', framealpha=1.0)
plt.savefig(f"plots/no_ood_data.png")
plt.show()
