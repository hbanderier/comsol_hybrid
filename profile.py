import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
from pns_v3 import extract_df
from util import realconv
plt.rc('font', size=12)
plt.rc('lines', lw=1.5)

def extract_profile(name, here):
    path = './data/' + name + '/'
    if isinstance(here, list):
        metadata = pk.load(open(path + 'metadata.pkl', 'rb'))
        mindist_all = metadata['mindist_all_z']
        here = mindist_all[here[0]][here[1]][here[2]]
    profile = np.loadtxt(path + 'profile.csv', delimiter=',', comments='%', usecols=[0, here + 1], converters={here + 1 : realconv})
    ind = np.argsort(profile[:, 0])
    profile[:, 1] = np.abs(profile[:, 1])
    profile[:, 1] = profile[:, 1] / np.nanmax(profile[:, 1])
    profile[:, 0] = (profile[:, 0] - 1e-4) * 1e6
    return profile[ind, :]

if __name__ == "__main__":
    prof_SM = extract_profile('SM', here=2)
    prof_1 = extract_profile('p3', here=1)
    prof_2 = extract_profile('p3', here=[-1, 0, 0])

    n = prof_SM.shape[0]
    x = prof_SM[:, 0]
    y = prof_SM[:, 1]
    mean = np.nansum(x * y) / n           
    sigma = np.nansum(y * (x - mean) ** 2) / n
    print(sigma)

    fig, ax1 = plt.subplots(tight_layout=True)
    ax2 = ax1.inset_axes([0.09,0.3,0.32,0.5])    
    x1, x2, y1, y2 = -100, -30, -0.002, 0.015

    pouets = [prof_SM, prof_1, prof_2]
    labels = ['Bare (0, 0)', 'Hybrid Qubit', 'Hybrid (0, 0)']
    colors = ['k-', 'b-', 'g-']
    for i in range(3):
        pouet = pouets[i]
        ax1.plot(pouet[:, 0], pouet[:, 1], colors[i], label=labels[i])
        ind = np.logical_and(pouet[:, 0] >= x1, pouet[:, 0] <= x2)
        ax2.plot(pouet[ind, 0], pouet[ind, 1], colors[i], label=labels[i])

    ax1.legend()
    ax1.grid(True)
    ax2.grid(True)
    ax2.hlines(0, x1, x2, 'k', lw=.8)
    ax1.set_xlabel('$x$ [$\mu$m]')
    ax1.set_ylabel('$u_z$, normalized')
    ax1.set_xlim([-130, 130])

    ax2.set_xlim(x1, x2)
    ax2.set_ylim(y1, y2)
    ax2.set_xticklabels([])
    ax2.set_yticks([0, 0.01])
    ax2.yaxis.label.set_color('red')
    ax2.tick_params(axis='y', colors='red')
    lw=1.5
    color = 'red'
    o1, o2 = ax1.indicate_inset_zoom(ax2, edgecolor=color, lw=lw, alpha=1)
    vis = [False, False, False, False]  # ll, ul, lr, ur
    for i in range(4):
        o2[i].set(lw=lw)
        o2[i].set(visible=vis[i])
    for key, spine in ax2.spines.items():
        spine.set_color(color)
        spine.set_linewidth(lw)
    plt.show(block=True)