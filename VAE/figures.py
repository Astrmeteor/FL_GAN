import numpy as np
import matplotlib.pyplot as plt

DATASET = ["mnist", "fashion-mnist", "cifar", "stl"]
DATASET = DATASET[0]
client_number = 1
DP = ["normal", "gaussian"]
DP = DP[1]

if client_number == 1:
    file_name = "values_central_200.npy"
else:
    file_name = "fid_loss_10.npy"

x = np.load("Results/{}/{}/{}/metrics/{}".format(DATASET, client_number, DP, file_name), allow_pickle=True)
loss = np.load("Results/{}/{}/{}/metrics/loss_200.npy".format(DATASET, client_number, DP), allow_pickle=True)
epsilon = np.load("Results/{}/{}/{}/metrics/epsilon_200.npy".format(DATASET, client_number, DP), allow_pickle=True)

if client_number != 1 :
    fid_central = [] # Under FL, the situation of server

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
    plt.plot(x_axis, y_axis, "r--", label="FID Centralized")
    # plt.plot(x_axis, y_axis, "ro-")
    plt.title("FID Centralized")
    plt.xlabel("Epoch")
    plt.ylabel("FID")
    plt.legend()
    # plt.show()
    plt.savefig('Results/{}/{}/{}/Figures/fid_centralized.png'.format(DATASET, client_number, DP), dpi=400)
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
    plt.savefig('Results/{}/{}/{}/Figures/fid_distributed.png'.format(DATASET, client_number, DP), dpi=400)
    plt.close()

    # Epsilon
    figure3 = plt.figure(figsize=(12, 8))
    y_axis = epsilon
    x_axis = np.arange(0, len(y_axis))
    plt.plot(x_axis, y_axis, "r--", label="Epsilon FL")
    plt.title("Epsilon FL")
    x_ticks = np.arange(0, len(y_axis) + 1, 10)
    plt.xticks(x_ticks)
    plt.xlabel("Epoch")
    plt.ylabel("Epsilon")
    plt.legend()
    plt.savefig('Results/{}/{}/{}/Figures/epsilon_fl.png'.format(DATASET, client_number, DP), dpi=400)
    plt.close()

    print("Saved Figures")

else:
    # FID
    figure = plt.figure(figsize=(12, 8))
    y_axis = x
    x_axis = np.arange(0, len(y_axis))
    plt.plot(x_axis, y_axis, "r--", label="FID Centralized")
    plt.title("FID Centralized")
    x_ticks = np.arange(0, len(y_axis)+1, 10)
    plt.xticks(x_ticks)
    plt.xlabel("Epoch")
    plt.ylabel("FID")
    plt.legend()
    plt.savefig('Results/{}/{}/{}/Figures/fid_centralized.png'.format(DATASET, client_number, DP), dpi=400)
    plt.close()

    # Loss
    figure2 = plt.figure(figsize=(12, 8))
    y_axis = loss
    x_axis = np.arange(0, len(y_axis))
    plt.plot(x_axis, y_axis, "r--", label="Loss Centralized")
    plt.title("Loss Centralized")
    x_ticks = np.arange(0, len(y_axis) + 1, 10)
    plt.xticks(x_ticks)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('Results/{}/{}/{}/Figures/loss_centralized.png'.format(DATASET, client_number, DP), dpi=400)
    plt.close()



    # print(x)

