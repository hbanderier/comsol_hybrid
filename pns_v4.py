import numpy as np
import pandas as pd
import pickle as pk
import scipy.constants as co
from util import realconv, filepath, loadmda, savemda



LOSS = {
    'sapp_delta': 5e-8,
    'AlN_delta': 1.5e-3,
    'sapp_scatt_Q': 7740905308201184.0,
    'AlN_scatt_Q' : 2.2559311082955164e+16
}


def extract_losses(df, lossusecol, losses, path):
    for key, value in losses.items():
        val = value[0]
        func = value[1]
        which = value[2]
        lossusecol1 = [lossusecol[0], lossusecol[0] + 1] if key=='p_surf' else lossusecol
        converters = {luc: realconv for luc in lossusecol1}
        df[val] = func(np.loadtxt(path + key + '.csv', delimiter=',', usecols=lossusecol1, comments='%', converters=converters))
        # df[val] = df[val] * (1./ factor[which] if val[0] == 'Q' else factor[which])
        # df['p_SM'] = factor['SM']
    if 'p_sapp' in losses.keys() and 'p_AlN' in losses.keys():
        df['Q_diel'] = 1 / (df['p_sapp'] * LOSS['sapp_delta'] + df['p_AlN'] * LOSS['AlN_delta'])
    return df


def extract_df(name, losses={}):
    path = filepath(name)
    metadata = loadmda(name)
    freqs = metadata['freqs']
    numsfreq = np.sum(freqs > 0, axis=1)
    oneL = freqs.shape[0] == 1
    if oneL:
        converters = {0 : realconv}
        usecols=(0, 1)
        lossusecol = [1]
    else:
        converters = {1: realconv}
        usecols=(0, 1, 2)
        lossusecol = [2]
    dat = np.loadtxt(path + 'pn.csv', delimiter=',', usecols=usecols, comments='%', converters=converters)
    if oneL:
        dat = np.hstack([metadata['L'][0] * np.ones((len(dat), 1)), dat])
    df = pd.DataFrame(dat, columns=['L', 'f', 'p'])
    if df['f'][0] < 10:
        df['f'] *= 1e9
    Ls = np.unique(df['L'])

    df = extract_losses(df, lossusecol, losses, path)
    
    pivoted = df.pivot(columns='L')
    cols = df.columns
    qures = pivoted['p'].idxmax()
    qubit_res = df.loc[qures].set_index('L')
    qubit_res = qubit_res.rename({c: c + 'q' for c in qubit_res.columns}, axis=1)
    starts = np.append([0], np.cumsum(numsfreq[:-1]))
    qures2 = qures.to_numpy() - starts
    allpeas = np.array([pivoted['p'].iloc[:, i].dropna().to_numpy() for i in range(len(Ls))], dtype=object)
    return df, qubit_res, qures, qures2, starts, allpeas


def cleandf(name, df, starts, qubit_res):
    metadata = loadmda(name)
    mindist_all = metadata['mindist_all_z']
    interesting_modes = [qubit_res]
    indices = {}
    modes = list(mindist_all[0].keys())
    for mode in modes:
        dont = False
        indices[mode] = [None] * len(metadata["L"])
        for l in range(len(metadata["L"])):
            if mode in mindist_all[l].keys():
                indices[mode][l] = mindist_all[l][mode]
            else:
                dont = True
        if dont:
            continue
        dfind = indices[mode] + starts
        this_df = df.loc[dfind].set_index('L')
        this_df = this_df.rename({c: c + mode for c in this_df.columns}, axis=1)
        this_df.loc[:, 'delta' + mode] = (this_df['f' + mode] - qubit_res['fq']).abs()
        interesting_modes.append(this_df)
            # df = df.drop(dfind)  # don't reset indices just yet. Should not be errors thanks to decide()
    new_df = pd.concat(interesting_modes, axis=1)
    modes.append("q")
    return new_df, modes


def adjust_p(new_df, df, Ls, modes, selfchi=False, deltas=None):
    phi0 = co.h / 2 / co.e
    factor = 1
    EJoverh = 1 / co.h / new_df.index * (phi0 / 2 / co.pi) ** 2  # [Hz]
    modes_to_change = modes
    # modes_to_change = [modes[0], modes[-1]]
    for L in Ls:
        if L < 1e-7:
            L = np.round(L * 1e9, 4) 
        here = df[df['L'] == L]
        for mode in modes:
            here.loc[:, mode] = np.abs(here['f'] - new_df.loc[L, 'f' + mode])
        here.loc[:, 'closest'] = here.loc[:, modes].idxmin(axis=1)
        for mode in modes:
            if mode in modes_to_change:
                new_df.loc[L, 'p_adj' + mode] = here['p'][here['closest'] == mode].sum()
            else:
                new_df.loc[L, 'p_adj' + mode] = new_df.loc[L, 'p' + mode]
    new_df['alpha_q'] = new_df['fq'] ** 2 * new_df['p_adjq'] ** 2 / EJoverh / 8
    for mode in modes[:-1]:  # now after redistribution
        if isinstance(deltas, dict):
            new_df.loc[:, "DeltaB_" + mode + "q"] = deltas[mode]
        elif isinstance(deltas, pd.DataFrame):
            new_df.loc[:, "DeltaB_" + mode + "q"] = deltas.loc[:, mode]
        else:
            new_df.loc[:, "DeltaB_" + mode + "q"] = - 0.00466 * 1e9 + new_df['f' + mode] - new_df['fq']  # GHz, close enough using f00-f^B_00 \approx 0.00466 GHz
        new_df.loc[:, 'chi_' + mode + 'q'] = new_df['f' + mode] * new_df['fq'] * new_df['p_adj' + mode] * new_df['p_adjq'] / EJoverh / 4 / (1 + new_df['alpha_q'] / new_df["DeltaB_" + mode + "q"])  # chi/2/pi  [Hz]
        new_df.loc[:, "expg_" + mode + "q"] = np.sqrt(new_df['chi_' + mode + 'q'] * new_df["DeltaB_" + mode + "q"] / 2 * (new_df['alpha_q'] + new_df["DeltaB_" + mode + "q"]) / new_df['alpha_q'])
        if selfchi:
            new_df.loc[:, 'alpha_' + mode] = new_df['f' + mode] * new_df['f' + mode] * new_df['p_adj' + mode] * new_df['p_adj' + mode] / EJoverh / 8  # chi/2/pi  [Hz]
    return new_df

