import matplotlib.pyplot as plt
import numpy as np

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/bar_num - 0.2, 1.03 * height, '%s' % float(height))


### Sample data
#  IS/Reconstruction, IS/Sampling, PSNR, FID
## VAE
# MNIST
vae_mnist_normal = [3.51, 1.62, 23.78, 6.23]
vae_mnist_normal_err = [0.10, 0.08, 0.34, 0.13]
vae_mnist_dp = [3.22, 1.23, 22.99, 6.98]
vae_mnist_dp_err = [0.13, 0.08, 0.21, 0.28]

# Fashion-MNIST
vae_fm_normal = [1.98, 1.69, 20.16, 17.77]
vae_fm_normal_err = [0.10, 0.12, 0.14, 0.23]
vae_fm_dp = [1.79, 1.11, 19.99, 18.01]
vae_fm_dp_err = [0.11, 0.14, 0.24, 0.20]

# CIFAR10
vae_cifar_normal = [2.87, 3.53, 20.18, 85.36]
vae_cifar_normal_err = [0.17, 0.17, 0.36, 1.94]
vae_cifar_dp = [2.56, 4.01, 19.99, 91.19]
vae_cifar_dp_err = [0.15, 0.18, 0.35, 2.67]

# STl
vae_stl_normal = [2.48, 2.51, 18.55, 70.48]
vae_stl_normal_err = [0.13, 0.16, 0.99, 4.03]
vae_stl_dp = [1.32, 1.99, 18.11, 68.17]
vae_stl_dp_err = [0.12, 0.13, 0.19, 3.43]

## VTF
# MNIST
vtf_mnsit_normal = [3.70, 1.84, 24.36, 5.84]
vtf_mnsit_normal_err = [0.15, 0.09, 0.21, 0.23]
vtf_mnist_dp = [3.33, 1.36, 24.22, 6.62]
vtf_mnist_dp_err = [0.12, 0.09, 0.22, 0.34]

# Fasion-MNSIT
vtf_fm_normal =[2.16, 2.24, 21.59, 16.20]
vtf_fm_normal_err = [0.10, 0.13, 0.19, 0.30]
vtf_fm_dp = [2.05, 1.18, 21.52, 16.99]
vtf_fm_dp_err = [0.13, 0.11, 0.28, 0.29]

# CIFAR10
vtf_cifar_normal = [3.18, 4.27, 22.46, 81.69]
vtf_cifar_normal_err = [0.14, 0.13, 0.39, 1.00]
vtf_cifar_dp = [2.71, 4.35, 21.12, 89.03]
vtf_cifar_dp_err = [0.13, 0.11, 0.41, 2.11]

# STL
vtf_stl_normal = [2.63, 2.64, 21.52, 66.41]
vtf_stl_normal_err = [0.14, 0.11, 0.33, 3.01]
vtf_stl_dp = [1.51, 2.35, 21.30, 65.48]
vtf_stl_dp_err = [0.22, 0.20, 0.27, 2.78]

## RTF
# MNIST
rtf_mnist_normal = [3.41, 1.51, 25.76, 5.00]
rtf_mnist_normal_err = [0.12, 0.10, 0.31, 0.26]
rtf_mnist_dp = [3.56, 1.49, 24.99, 5.92]
rtf_mnist_dp_err = [0.11, 0.03, 0.30, 0.31]

# Fashion-MNIST
rtf_fm_normal = [1.92, 1.17, 22.21, 15.40]
rtf_fm_normal_err = [0.13, 0.10, 0.35, 0.23]
rtf_fm_dp = [2.31, 2.36, 22.05, 15.83]
rtf_fm_dp_err = [0.13, 0.13, 0.32, 0.34]

# CIFAR10
rtf_cifar_normal = [3.09, 4.73, 22.94, 73.34]
rtf_cifar_normal_err = [0.16, 0.06, 0.23, 1.14]
rtf_cifar_dp = [3.09, 4.73, 22.94, 73.34]
rtf_cifar_dp_err = [0.19, 0.22, 0.27, 1.34]

# STL
rtf_stl_normal = [1.51, 1.33, 23.04, 61.17]
rtf_stl_normal_err = [0.11, 0.13, 0.29, 1.98]
rtf_stl_dp = [1.85, 2.69, 23.22, 57.95]
rtf_stl_dp_err = [0.18, 0.12, 0.34, 2.31]

mnist_normal = np.array([vae_mnist_normal, vtf_mnsit_normal, rtf_mnist_normal]).T
mnist_normal_err = np.array([vae_mnist_normal_err, vtf_mnsit_normal_err, rtf_mnist_normal_err]).T
mnist_dp = np.array([vae_mnist_dp, vtf_mnist_dp, rtf_mnist_dp]).T
mnist_dp_err = np.array([vae_mnist_dp_err, vtf_mnist_dp_err, rtf_mnist_dp_err]).T

fm_normal = np.array([vae_fm_normal, vtf_fm_normal, rtf_fm_normal]).T
fm_normal_err = np.array([vae_fm_normal_err, vtf_fm_normal_err, rtf_fm_normal_err]).T
fm_dp = np.array([vae_fm_dp, vtf_fm_dp, rtf_fm_dp]).T
fm_dp_err = np.array([vae_fm_dp_err, vtf_fm_dp_err, rtf_fm_dp_err]).T

cifar_normal = np.array([vae_cifar_normal, vtf_cifar_normal, rtf_cifar_normal]).T
cifar_normal_err = np.array([vae_cifar_normal_err, vtf_cifar_normal_err, rtf_cifar_normal_err]).T
cifar_dp = np.array([vae_cifar_dp, vtf_cifar_dp, rtf_cifar_dp]).T
cifar_dp_err = np.array([vae_cifar_dp_err, vtf_cifar_dp_err, rtf_cifar_dp_err]).T

stl_normal = np.array([vae_stl_normal, vtf_stl_normal, rtf_stl_normal]).T
stl_normal_err = np.array([vae_stl_normal_err, vtf_stl_normal_err, rtf_stl_normal_err]).T
stl_dp = np.array([vae_stl_dp, vtf_stl_dp, rtf_stl_dp]).T
stl_dp_err = np.array([vae_stl_dp_err, vtf_stl_dp_err, rtf_stl_dp_err]).T


method = ["VAE", "VTF", "RTF"]
dataset = ["IS/Reconstruction", "IS/Sampling", "PSNR", "FID"]
x_axis = [0, 1, 2]
total_width = 0.8
bar_num = 2
width = total_width / bar_num

# MNIST
n_rows, n_cols = 1, 4
x = [0, 0, 0]
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5))
for col_num in range(n_cols):
    ax = axes[col_num]
    ax.set_xlim(-0.5, 3)
    error_kw = {'capsize': 4, 'linewidth': 1, 'elinewidth': 0.5, 'ecolor': 'red'}
    bar = ax.bar(x_axis, mnist_normal[col_num], yerr = mnist_normal_err[col_num], error_kw=error_kw, width=width, color='#63b2ee', label="Normal Scheme")
    bar_labels = ax.bar_label(bar, label_type="edge", fmt='%.2f')

    for bar_label in bar_labels:
        bar_x, bar_y = bar_label.get_position()
        bar_label.set_position((bar_x-4, bar_y))

    for i in range(len(x_axis)):
        x[i] = x_axis[i] + width
    bar_dp = ax.bar(x, mnist_dp[col_num], yerr = mnist_dp_err[col_num], error_kw=error_kw, width=width, color='#76da91', label="DP Scheme")
    bar_dp_labels = ax.bar_label(bar_dp, label_type="edge", fmt='%.2f')

    ax.set_title(dataset[col_num])
    x_ticks = (np.array(x_axis) + np.array(x)) / 2
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(method)

# Display the plot
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
plt.tight_layout()
plt.savefig('mnist_result.png', dpi=1200, format='png', bbox_inches='tight')
plt.close()

# Fashion-MNIST

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5))
for col_num in range(n_cols):
    ax = axes[col_num]
    ax.set_xlim(-0.5, 3)
    error_kw = {'capsize': 4, 'linewidth': 1, 'elinewidth': 0.5, 'ecolor': 'red'}
    bar = ax.bar(x_axis, fm_normal[col_num], yerr = fm_normal_err[col_num], error_kw=error_kw, width=width, color='#f8cb7f', label="Normal Scheme")
    bar_labels = ax.bar_label(bar, label_type="edge", fmt='%.2f')

    for bar_label in bar_labels:
        bar_x, bar_y = bar_label.get_position()
        bar_label.set_position((bar_x-4, bar_y))

    for i in range(len(x_axis)):
        x[i] = x_axis[i] + width
    bar_dp = ax.bar(x, fm_dp[col_num], yerr = fm_dp_err[col_num], error_kw=error_kw, width=width, color='#f89588', label="DP Scheme")
    bar_dp_labels = ax.bar_label(bar_dp, label_type="edge", fmt='%.2f')

    ax.set_title(dataset[col_num])
    x_ticks = (np.array(x_axis) + np.array(x)) / 2
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(method)

# Display the plot
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
plt.tight_layout()
plt.savefig('fm_result.png', dpi=1200, format='png', bbox_inches='tight')
plt.close()

# CIFAR10

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5))
for col_num in range(n_cols):
    ax = axes[col_num]
    ax.set_xlim(-0.5, 3)
    error_kw = {'capsize': 4, 'linewidth': 1, 'elinewidth': 0.5, 'ecolor': 'red'}
    bar = ax.bar(x_axis, cifar_normal[col_num], yerr = cifar_normal_err[col_num], error_kw=error_kw, width=width, color='#7cd6cf', label="Normal Scheme")
    bar_labels = ax.bar_label(bar, label_type="edge", fmt='%.2f')

    for bar_label in bar_labels:
        bar_x, bar_y = bar_label.get_position()
        bar_label.set_position((bar_x-4, bar_y))

    for i in range(len(x_axis)):
        x[i] = x_axis[i] + width
    bar_dp = ax.bar(x, cifar_dp[col_num], yerr = cifar_dp_err[col_num], error_kw=error_kw, width=width, color='#9192ab', label="DP Scheme")
    bar_dp_labels = ax.bar_label(bar_dp, label_type="edge", fmt='%.2f')

    ax.set_title(dataset[col_num])
    x_ticks = (np.array(x_axis) + np.array(x)) / 2
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(method)

# Display the plot
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
plt.tight_layout()
plt.savefig('cifar_result.png', dpi=1200, format='png', bbox_inches='tight')
plt.close()

# STL

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5))
for col_num in range(n_cols):
    ax = axes[col_num]
    ax.set_xlim(-0.5, 3)
    error_kw = {'capsize': 4, 'linewidth': 1, 'elinewidth': 0.5, 'ecolor': 'red'}
    bar = ax.bar(x_axis, stl_normal[col_num], yerr = stl_normal_err[col_num], error_kw=error_kw, width=width, color='#7898e1', label="Normal Scheme")
    bar_labels = ax.bar_label(bar, label_type="edge", fmt='%.2f')

    for bar_label in bar_labels:
        bar_x, bar_y = bar_label.get_position()
        bar_label.set_position((bar_x-4, bar_y))

    for i in range(len(x_axis)):
        x[i] = x_axis[i] + width
    bar_dp = ax.bar(x, stl_dp[col_num], yerr = stl_dp_err[col_num], error_kw=error_kw, width=width, color='#efa666', label="DP Scheme")
    bar_dp_labels = ax.bar_label(bar_dp, label_type="edge", fmt='%.2f')

    ax.set_title(dataset[col_num])
    x_ticks = (np.array(x_axis) + np.array(x)) / 2
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(method)

# Display the plot
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
plt.tight_layout()
plt.savefig('stl_result.png', dpi=1200, format='png', bbox_inches='tight')
plt.close()