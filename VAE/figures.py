import numpy as np
import matplotlib.pyplot as plt

x = np.load("Results/stl/2/metircs/fid_loss_10.npy", allow_pickle=True)


fid_central = []

for c_i in x.item().metrics_centralized["fid"]:
    fid_central.append(c_i[1])

fid_dist = {}

for k, v in x.item().metrics_distributed.items():
    if k not in fid_dist.keys():
        fid_dist[k] = []
    for c_i in v:
        fid_dist[k].append(c_i[1])

figure1 = plt.figure()
# draw fid of centralized
x_axis = np.arange(0, len(fid_central))
y_axis = fid_central
l = plt.plot(x_axis, y_axis, "r--", label="FID Centralized")
plt.plot(x_axis, y_axis, "ro-")
plt.title("FID Centralized")
plt.xlabel("Epoch")
plt.ylabel("FID")
plt.legend()
# plt.show()
plt.savefig('Results/stl/2/Figures/fid_centralized.png', dpi=400)
plt.close()

figure2 = plt.figure()
# draw fid of distributed
x2_axis = np.arange(0, len(fid_dist[0]))
handles = []
labels = []
cmap = plt.cm.get_cmap("hsv", len(fid_dist[0]))

for k, v in fid_dist.items():
   line, = plt.plot(x2_axis, v, color=cmap(k))
   handles.append(line)
   labels.append("Client {}".format(k))


x2_ticks = np.arange(0, len(fid_dist[0]), 1)

plt.title("FID Distributed")
plt.xlabel("Epoch")
plt.ylabel("FID")
plt.legend(handles=handles, labels=labels, loc='upper right')
plt.xticks(x2_ticks)
plt.grid(visible="True", axis="x")
plt.savefig('Results/stl/2/Figures/fid_distributed.png', dpi=400)
plt.close()
print("Saved Figures")
# plt.show()
# print(x)
