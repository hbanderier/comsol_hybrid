import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import constants as co
import mph
import time as timer
from util import loadmda, savemda, realconv, extract_freqs, is_numeric, filepath
from analysis_v4 import find_modes_wrapper, create_Zref
from pns_v4 import extract_df, cleandf, adjust_p, compute_dispersive_quantities
plt.rc('font', size=12)
plt.rc('lines', lw=1)
plt.rc('lines', mew=1.4)
plt.rc('lines', markersize=6)
pd.set_option('mode.chained_assignment', None)


colors12 = {
    "L": {"LG" : "#118AB2", "HG" : "#073B4C"}, 
    "SH": {"LG" : "#FFD166", "HG" : "#EF476F"}
}
colors3 = ["#06D6A0", "#EF476F"]
markers = ['o', 's', 'D', "X", "h", "v", "^", "p", "<", ">", "8"]


def create_variables(w02=4.78120254670659e-10):
    N = 200
    X, Y = np.meshgrid(np.linspace(9e-4, 11e-4, N), np.linspace(-1e-4, 1e-4, N))
    Xc = 1e-3
    phi = np.arctan2((X - Xc), Y) + np.pi / 2
    r = np.sqrt((X - Xc) ** 2 + Y ** 2)
    rho = 2 * r * r / w02
    Xp = np.sqrt(2 / w02) * (X - Xc)
    Yp = np.sqrt(2 / w02) * Y
    return [rho, phi, Xp, Yp]


def create_and_save_mda(name, filename, sol, dset, model):
    metadata = {
        "name" : name,
        "filename" : filename,
        "comp": "comp7",
        "mesh": "mesh1",
        "sol": sol,
        "dset": dset,
    }
    javaobj = model.java
    metadata['geom_sf_o'] = str(javaobj.modelNode(metadata['comp']).sorder())
    metadata['disp_sf_o'] = int(str(javaobj.component(metadata['comp']).physics('solid')
                                    .prop('ShapeProperty').getString('order_displacement')[0]))
    metadata['efield_sf_o'] = int(str(javaobj.component(metadata['comp']).physics('emw')
                                      .prop('ShapeProperty').getString('order_electricfield')[0]))
    freqs1, L, is_sweep = extract_freqs(javaobj, model, metadata['sol'], True)
    metadata['freqs'], metadata['L'] = freqs1, L
    print(L)
    savemda(metadata)


def init_mph():
    client = mph.start()
    ti = timer.time()
    filename = "/users/Hyqu/Desktop/Final"
    model = client.load(filename)
    print(timer.time() - ti)
    names = ["EM", "SM", "EMsweep", "sweep1", "sweep2", "disp_mech", "disp_qubit"]
    sols = ["sol1", "sol2", "sol5", "sol47", "sol80", "sol4", "sol78"]
    dsets = ["dset1", "dset2", "dset11", "dset12", "dset15", "dset4", "dset13"]
    for i in range(len(names)):
        ti = timer.time()
        create_and_save_mda(names[i], filename, sols[i], dsets[i], model)
        print(timer.time() - ti)


def init_look_for():
    metadata = loadmda("SM")
    metadata["look_for"] = {
        "LG00": 6.4403,
        "LG01a": 6.4429,
        "LG01b": 6.4436,
        "HG02": 6.4454,
        "HG20a": 6.4457,
        "HG20b": 6.4463,
        "LG03": 6.4483,
        "HG12": 6.4484,
        "HG30": 6.4493,
        "HG04": 6.4507,
        "HG22": 6.4512,
        "HG40a": 6.4514,
        "HG40b": 6.4518,
        "HG32": 6.4542,
        "HG50": 6.4550,
    }
    savemda(metadata)
    metadata = loadmda("disp_mech")
    metadata["look_for"] = {
        "LG00": 6.4451,
        "LG01a": 6.4476,
        "LG01b": 6.4486,
        "HG02": 6.4497,
        "HG20": 6.4506,
        "LG03": 6.4529,
        "HG30": 6.4539,
        "HG04": 6.4554,
        "HG22": 6.4559,
        "HG40": 6.4563,
        "HG14": 6.4578,
    }
    savemda(metadata)
    metadata = loadmda("sweep1")
    metadata["look_for"] = {
        "LG00": 6.4451,
        "LG01a": 6.4476,
        "LG01b": 6.4486,
        "HG02": 6.4497,
        "HG20": 6.4506,
        "LG03": 6.4529,
        "HG30": 6.4539,
        "HG04": 6.4554,
        "HG22": 6.4559,
        "HG40": 6.4563,
        "HG14": 6.4578,
    }
    savemda(metadata)
    metadata = loadmda("sweep2")
    metadata["look_for"] = {
        "LG00": 6.4451,
        "LG01a": 6.4476,
        "LG01b": 6.4486,
        "HG02": 6.4497,
        "HG20": 6.4506,
        "LG03": 6.4529,
        "HG30": 6.4539,
        "HG04": 6.4554,
        "HG22": 6.4559,
        "HG40": 6.4563,
        "HG14": 6.4578,
    }
    savemda(metadata)


def get_bare_freqs(nameEM, nameSM):
    path = filepath(nameSM)
    metadata = loadmda(nameSM)
    bare_mech_fs = {key: metadata["freqs"][0][value] for key, value in metadata["mindist_all_z"][0].items()}
    bare_fq = loadmda(nameEM)["freqs"]
    allgs = np.loadtxt(path + 'g.csv', delimiter=',', comments='%', converters={0: realconv, 1: realconv})
    if bare_fq.shape[0] == 1:
        bare_fq = bare_fq[0, 0]
        deltaB = {key: value - bare_fq for key, value in bare_mech_fs.items()}
        gs = {key : allgs[value][1] for key, value in metadata["mindist_all_z"][0].items()}
    else:
        bare_fq = bare_fq[:, 0]
        Ls = np.round(loadmda(nameEM)["L"] * 1e9, 4)
        deltaB = pd.DataFrame(index=Ls)
        gs = pd.DataFrame(index=Ls)
        for mode, value in bare_mech_fs.items():
            deltaB.loc[:, mode] = bare_fq - value
            gs.loc[:, mode] = allgs[metadata["mindist_all_z"][0][mode]][1]
    return bare_fq, bare_mech_fs, deltaB, gs


def dispmain(): # about dispersive regime and unhybridized approaches. Produces figure 2, data in section IV.B tables and text.
    
    # find_modes_wrapper("SM", [10000], [np.zeros(loadmda("SM")["freqs"].shape[-1])])
    bare_fq, bare_mech_fs, deltaB, gs = get_bare_freqs("EM", "SM")
    name = "disp_mech"
    df, qubit_res, qures, qures2, starts, allpeas = extract_df(name)
    metadata = loadmda(name)
    # find_modes_wrapper(name, [100000], allpeas)
    df2, qubit_res, qures, qures2, starts, allpeas = extract_df("disp_qubit")
    df = df.append(df2).reset_index(drop=True)
    new_df, modes, brokenmodes = cleandf(name, df, starts, qubit_res)
    
    common_modes = [mode for mode in deltaB.keys() if mode in modes]
    common_modes.append("q")

    new_df = adjust_p(new_df, df, metadata["L"], modes)
    new_df = compute_dispersive_quantities(new_df, common_modes, selfchi=True, deltas=deltaB)

    data = list(new_df.transpose().to_dict().values())[0]
    for mode in common_modes[:-1]:
        data["bare_f" + mode] = bare_mech_fs[mode]
        data["g_" + mode + "q"] = gs[mode]

    new_df.T.to_csv('disp_df.csv')

    fig = plt.figure(figsize=(8, 7), tight_layout=True)
    numtop = 7
    gs = gridspec.GridSpec(6, numtop, wspace=0, hspace=0, height_ratios=[1.4, 1, 0.6, 1, 0.6, 1])
    axes = []
    invaxes = []
    for i in range(5):
        if i == 1 or i == 3:
            invaxes.append(fig.add_subplot(gs[i + 1, :]).set_visible(False))
        else:
            axes.append(fig.add_subplot(gs[i + 1, :]))
    lgaxes = []
    for i in range(numtop):
        lgaxes.append(fig.add_subplot(gs[0, i]))
    width12 = 0.0005
    width3 = 0.3
    modetype = {"H": "SH", "L": "L"}
    nicermode = []
    unique_top = []
    counter = 0
    variables = create_variables()

    for mode in common_modes[:-1]:
        family = mode[:2]
        m, n = mode[2], mode[3]
        typ = modetype[mode[-1]]
        nicermode.append(f'${family} ({m}, {n})_' + '{\mathrm{' + f'{typ}' + '}}$')
        axes[0].bar(data["bare_f" + mode] * 1e-9, np.abs(data["g_" + mode + "q"]) / 1000, color=colors12[typ][family], label=nicermode[counter], width=width12)
        axes[1].bar(data["f" + mode] * 1e-9, np.abs(data["expg_" + mode + "q"]) / 1000, color=colors12[typ][family], width=width12)
        axes[2].bar(counter, np.abs(data["g_" + mode + "q"]) / 1000, width=width3, color=colors3[0], align='edge')
        axes[2].bar(counter, np.abs(data["expg_" + mode + "q"]) / 1000, width=-width3, color=colors3[1], align='edge')
        this_uniquetop = f"{family}{m}{n}"
        if len(unique_top) < numtop and this_uniquetop not in unique_top:
            i = len(unique_top)
            Z = create_Zref(family == "LG", int(m), int(n), variables)
            vmax = np.max(np.abs(Z))
            lgaxes[i].imshow(Z, cmap='bwr', vmin=-vmax, vmax=vmax)
            lgaxes[i].set_title(f'{family} ({m}, {n})')
            lgaxes[i].set_xticks([])
            lgaxes[i].set_yticks([])
            unique_top.append(this_uniquetop)
        counter += 1
    axes[0].legend(ncol=4)
    handles = [plt.Rectangle((0,0),1,1, color=color) for color in colors3]
    labels = ["$g$ from unhybridized approach", "$g$ from hybridized approach"]
    axes[2].legend(handles, labels)
    axes[2].set_xticks(np.arange(counter))
    axes[2].set_xticklabels(nicermode)
    for i in range(3):
        axes[i].set_ylabel(r'$g / 2 \pi$ [kHz]')
        if i < 2 :
            axes[i].set_xlabel('$f$ [GHz]')

    plt.show(block=True)


def sweepmain(): # about the L sweep in section IV.B. Produces figure 3.
    # find_modes_wrapper("SM", [10000], [np.zeros(loadmda("SM")["freqs"].shape[-1])])
    mda = loadmda("EMsweep")
    f, L = mda["freqs"], mda["L"]
    C = np.mean(1 / (2 * np.pi * f) ** 2 / L)
    ECover2h = co.e ** 2 / co.h / 2 / C
    print(f"C={C:.4e}")
    print(f"EC/h={ECover2h:.4e}")
    ouais = []
    mmodes = []
    bbrokenmodes = []
    bare_fq, bare_mech_fs, deltaB, gs = get_bare_freqs("EMsweep", "SM")
    for name in ["sweep1", "sweep2"]:
        path = filepath(name)
        metadata = loadmda(name)
        df, qubit_res, qures, qures2, starts, allpeas = extract_df(name)
        # find_modes_wrapper(name, qures2, allpeas)
        new_df, modes, brokenmodes = cleandf(name, df, starts, qubit_res)
        mmodes.extend(modes)
        bbrokenmodes.append(brokenmodes)
        new_df = adjust_p(new_df, df, metadata["L"], modes)
        ouais.append(new_df)
    new_df = pd.concat(ouais).sort_index()
    brokenmodes = pd.concat(bbrokenmodes).sort_index().fillna(value=False).iloc[:-1,:]
    modes = np.array(mmodes)
    _, idx = np.unique(modes, return_index=True)
    modes = modes[np.sort(idx)].tolist()
    new_df = new_df.iloc[:-1, :]
    bare_fq = bare_fq[:-1]
    modetype = {"H": "SH", "L": "L"}
    nicermode = []

    # fig, [ax1, ax2] = plt.subplots(2, 1, tight_layout=True, figsize=(8, 15))
    fig = plt.figure(figsize=(8, 11))
    gs = gridspec.GridSpec(3, 1, wspace=0, hspace=0, height_ratios=[0.335, 1, 1], top=0.98, bottom=0.05, right=0.98)
    ax1 = fig.add_subplot(gs[1, :])
    ax0 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[2, :])
    # ax1.text(0.02,0.95, 'a)', fontweight='bold')
    # fig2, ax2 = plt.subplots(tight_layout=True)
    # ax2.text(0.02,0.95, 'b)', fontweight='bold')
    ax1.annotate('a)',
        xy=(3, 1), xycoords='axes fraction',
        xytext=(0.02, 0.98), textcoords='axes fraction',
        horizontalalignment='left', verticalalignment='top', 
        fontweight="bold", fontsize=15, bbox=dict(boxstyle="Square", fc="white", alpha=0.8, ec="white"))
    ax2.annotate('b)',
        xy=(3, 1), xycoords='axes fraction',
        xytext=(0.02, 0.98), textcoords='axes fraction',
        horizontalalignment='left', verticalalignment='top', 
        fontweight="bold", fontsize=15, bbox=dict(boxstyle="Square", fc="white", alpha=0.8, ec="white"))

    ax1.plot(
        bare_fq / 1e9, new_df['fq'] / 1e9, label='qubit', ls='None', 
        markerfacecolor='None', color='black', marker='+', ms=8, mew=1.8
        )
    ax2.plot(
        bare_fq / 1e9, new_df['p_adjq'], marker='+',
        label='Qubit', color='black', ls='--', ms=8, mew=1.8
    )
    k = 0
    marker_counter = {"L": {"LG": 0, "HG": 0}, "SH": {"LG": 0, "HG": 0}}
    for mode in modes[:-1]:
        family = mode[:2]
        m, n = mode[2], mode[3]
        typ = modetype[mode[-1] if mode[-1] != 'b' else mode[-2]]
        nicermode.append(f'${family} ({m}, {n})_' + '{\mathrm{' + f'{typ}' + '}}$')
        indexer = brokenmodes.loc[:, mode].to_numpy()
        if mode[-1] == 'b':
            marker_counter[typ][family] -= 1
        try:
            sf = np.nanmax(np.abs(new_df["f" + mode][indexer].values[:-1] - new_df["f" + mode][indexer].values[1:])) / (1e6)
        except:
            sf = 0
        ms = 3 + 3
        print(sf, ms)

        ax1.plot(
            bare_fq[indexer] / 1e9, new_df['f' + mode][indexer] / 1e9, label=nicermode[k], ls='None', 
            markerfacecolor='None', color=colors12[typ][family], marker=markers[marker_counter[typ][family]], ms=ms
        )
        ax2.plot(
            bare_fq[indexer]  / 1e9, new_df['p_adj' + mode][indexer], label=mode, 
            color=colors12[typ][family], ls='--', markerfacecolor='None', marker=markers[marker_counter[typ][family]], ms=ms
        )
        marker_counter[typ][family] += 1
        k += 1
    for ax in [ax1, ax2]:
        ax.grid(True)
        ax.set_xlabel('Bare qubit frequency [GHz]')
    lines, labels = ax1.get_legend_handles_labels()

    ax0.legend(lines, labels, ncol=4, bbox_to_anchor=(0., 0, 1., .5), 
               loc='lower left', mode="expand", borderaxespad=0., 
               framealpha=0, title='Legend', title_fontproperties=dict(weight="bold"))
    # ax1.set_xticks([])
    ax1.set_xticklabels([])
    ax0.set_xticks([])
    ax0.set_xticklabels([])
    ax0.set_yticks([])
    ax0.set_yticklabels([])
    
    ax1.set_ylabel('Mode frequency [GHz]')    
    ax2.set_ylabel('Energy participation ratio ($p$)')
    fig.align_ylabels([ax1, ax2])
    plt.show(block=True)


if __name__ == '__main__':
    # init_mph()
    # init_look_for()
    dispmain()
    # sweepmain()