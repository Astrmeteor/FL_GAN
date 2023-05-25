import matplotlib.pyplot as plt
import csv
import numpy as np
import os

dataset = "cifar10"
version = "v2"
csv_file = f"{version}_loss_eps.csv"
file_name = f"{dataset}/Result/{csv_file}"

csv_reader = csv.reader(open(file_name))

loss = []
recon_loss = []
vq_loss = []
dp_loss = []
dp_recon_loss = []
dp_vq_loss = []
eps = []

Dataframe = list(csv_reader)
for line in Dataframe[1:]:
    loss.append(float(line[0]))
    recon_loss.append(float(line[1]))
    vq_loss.append(float(line[2]))
    dp_loss.append(float(line[3]))
    dp_recon_loss.append(float(line[4]))
    dp_vq_loss.append(float(line[5]))
    eps.append(float(line[6]))

fig, ax1 = plt.subplots(figsize=(10, 5))

x = np.arange(1, 101, 1)

ax1.set_yscale("log", base=2)
ax1.set_yticks([0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128])
ax1.set_yticklabels(["0.0625", "0.125", "0.25", "0.5", "1", "2", "4", "8", "16", "32", "64", "128"])

# loss
y1 = loss
lns1 = ax1.plot(x, y1, label="Loss")
# Recon loss
y2 = recon_loss
lns2 = ax1.plot(x, y2, label="Reconstruction Loss")
# Vq loss
y3 = vq_loss
lns3 = ax1.plot(x, y3, label="VQ Loss")
# DP loss
y4 = dp_loss
lns4 = ax1.plot(x, y4, label="DP Loss")
# DP Recon loss
y5 = dp_recon_loss
lns5 =ax1.plot(x, y5, label="DP Reconstruction Loss")
# DP Vq loss
y6 = dp_vq_loss
lns6 = ax1.plot(x, y6, label="DP VQ Loss")


# ax1.set_ylim(0, 20)
ax1.set_xlim([-0.5, 100])
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.xaxis.set_major_locator(plt.MultipleLocator(10))
# ax1.yaxis.set_major_locator(plt.MultipleLocator(0.1))

# ax1.set_zorder(0)

ax2 = ax1.twinx()
y6 = eps
lns7 = ax2.plot(x, y6, color="k", label="Epsilon")
ax2.set_ylim([0, 5])
ax2.set_ylabel("Epsilon")
ax2.yaxis.set_major_locator(plt.MultipleLocator(0.5))

# lns = lns1 + lns2 + lns3 + lns4 + lns5 + lns6 + lns7
# labs = [l.get_label() for l in lns]
fig.legend(bbox_to_anchor=(0.5, -0.05), loc=8, ncol=7)# , borderaxespad=0)
# plt.tight_layout()
plt.title(f"Loss and Epsilon for VQ-VAE model: CIFAR-10")
# plt.show()
plt.savefig(f'{dataset}_{version}_example.png', dpi=1200, format='png', bbox_inches='tight')





