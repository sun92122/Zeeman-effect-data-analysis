import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def dk_B_to_muB(dk_B, dk_B_err=0):
    # mu_B = 9.274e-24 J/T = h * c * dk_B
    # h = 6.62607015 × 10-34 m2 kg / s
    # c = 299792458 m/s
    return 6.62607015e-34 * 299792458 * dk_B / 2, 6.62607015e-34 * 299792458 * dk_B_err / 2


font = {'family': 'DejaVu Sans', 'weight': 'normal', 'size': 18}

# Load the provided Excel file
file_path = './IBdk.xlsx'
data = pd.read_excel(file_path)

# Display the first few rows of the data to understand its structure
# print(data.head())

d = 1 / (2 * 1.4560 * 6e-3)
I_1 = data['1_I'].dropna()
B_1 = data['1_B'].dropna() * 1e-3
B_1_error = data['1_B_err'].dropna() * 1e-3
p_1_1 = data['1_1'].dropna() * d
p_1_2 = data['1_2'].dropna() * d
I_2 = data['2_I'].dropna()
B_2 = data['2_B'].dropna() * 1e-3
B_2_error = data['2_B_err'].dropna() * 1e-3
p_2_1 = data['2_1'].dropna() * d
p_2_2 = data['2_2'].dropna() * d
I = [I_1, I_1, I_2, I_2]
B = [B_1, B_1, B_2, B_2]
B_error = [B_1_error, B_1_error, B_2_error, B_2_error]
k = [p_1_1, p_1_2, p_2_1, p_2_2]
# 1_ transverse, 2_ longitudinal
k_label = [r'$\Delta k_{p=1\perp}$', r'$\Delta k_{p=2\perp}$',
           r'$\Delta k_{p=1\parallel}$', r'$\Delta k_{p=2\parallel}$']
c = ['b', 'g', 'c', 'm']

# I

# Creating the plot
plt.figure(figsize=(8, 6))

for i in range(4):
    plt.scatter(I[i], k[i], marker='o', color=c[i], s=3, label=k_label[i])
    # fit
    fit = np.polyfit(I[i], k[i], 1, cov=True)
    print(fit)
    fit_fn = np.poly1d(fit[0])
    plt.plot(I[i], fit_fn(I[i]), '--'+c[i])
    # plt.fill_between(I[i], fit_fn(I[i]) - np.sqrt(np.diag(fit[1]))[0], fit_fn(
    #     I[i]) + np.sqrt(np.diag(fit[1]))[0], color=c[i], alpha=0.1)

# Setting the axis labels with LaTeX notation
plt.xlabel(r'$I\ (\mathrm{A})$', fontdict=font)
plt.ylabel(r'$\Delta k$', fontdict=font)

# Adding main ticks
plt.tick_params(axis='both', which='major', labelsize=18,
                top=True, right=True, direction='in')

# Removing grid lines
plt.grid(False)

# Ensuring the entire chart, including all axes and labels, is correctly visualized
plt.tight_layout()

# Adding legend
plt.legend(loc='best', fontsize=16)

# Show the plot
plt.show()

# B

# Creating the plot
plt.figure(figsize=(8, 6))

mu_B = []
mu_B_err = []
for i in range(4):
    # fit
    fit = np.polyfit(B[i], k[i], 1, cov=True)
    print(fit)
    fit_fn = np.poly1d(fit[0])

    plt.scatter(B[i], k[i], marker='o', color=c[i], s=3, label=k_label[i] + r'$,\ y={:.2f}x {:+.2f},\ R^2={:.3f}$'.format(fit[0][0], fit[0][1], np.corrcoef(B[i], k[i])[0, 1] ** 2))
    # label = k_label[i] + (function of fit parameters)
    plt.plot(B[i], fit_fn(B[i]), '--'+c[i])
    # plt.fill_between(B[i], fit_fn(B[i]) - np.sqrt(np.diag(fit[1]))[0], fit_fn(
    #     B[i]) + np.sqrt(np.diag(fit[1]))[0], color=c[i], alpha=0.1)

    # Plot B_err
    plt.errorbar(B[i], k[i], xerr=B_error[i],
                 fmt='none', ecolor=c[i], alpha=0.3)

    # output mu_B
    mu_B_temp, mu_B_err_temp = dk_B_to_muB(fit[0][0], np.sqrt(np.diag(fit[1]))[0])
    print('mu_B = {:.2e} +- {:.2e} J/T'.format(mu_B_temp, mu_B_err_temp))
    mu_B.append(mu_B_temp)
    mu_B_err.append(mu_B_err_temp)

# Setting the axis labels with LaTeX notation
plt.xlabel(r'$B\ (\mathrm{T})$', fontdict=font)
plt.ylabel(r'$\Delta k$', fontdict=font)

# Adding main ticks
plt.tick_params(axis='both', which='major', labelsize=18,
                top=True, right=True, direction='in')

# Removing grid lines
plt.grid(False)

# Ensuring the entire chart, including all axes and labels, is correctly visualized
plt.tight_layout()

# Adding legend
plt.legend(loc='upper left', fontsize=16)

# Show the plot
plt.show()

# mu_B
print('mu_B = {:.2e} +- {:.2e} J/T'.format(np.mean(mu_B), np.max(mu_B_err)))
# print('mu_B = {:.2e} +- {:.2e} J/T'.format(np.mean(mu_B), np.std(mu_B)))