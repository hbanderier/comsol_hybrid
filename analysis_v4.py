from hashlib import new
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.interpolate import griddata as gd
from scipy.special import genlaguerre
from scipy.special import hermite
from util import filepath, realconv, loadmda, savemda
plt.rc('font', size=12)
plt.rc('lines', linewidth=3)
plt.rc('lines', markersize=7)
plt.ion()


def sapphire(n=120):
#  Sapphire, isotropic
    rho = 3965
    E = 400e9
    nu = 0.22
    c33 = E * (1 - nu) / (1 + nu) / (1 - 2 * nu)
    c13 = E * nu / (1 + nu) / (1 - 2 * nu)
    c44 = E / 2 / (1 + nu)
    vl = np.sqrt(c33 / rho)
    vt = np.sqrt(c44 / rho)
    gamma = np.sqrt((c13 + c44) / rho)
    chi = vl ** 2 * (vl ** 2 - vt ** 2) / (vl ** 2 * vt ** 2 - vt ** 4 + gamma ** 4)
    R2 = 5.6e-3
    t = 104.5e-6
    f0 = vl / 2 / t
    zR = np.sqrt(chi * R2 / t - 1) * t
    psi0 = np.arctan(t / zR)
    lam = vl / f0 / n
    w0 = np.sqrt(t * lam / np.pi / chi * np.sqrt(chi * R2 / t - 1))
    return lam, f0, w0, psi0


def create_Zref(LG, m, n, variables):
    if LG:
        rho, phi = variables[0], variables[1]
        p = m
        l = n
        return rho ** (l / 2) * genlaguerre(p, l)(rho) * np.exp(-rho/2) * np.cos(l *  phi)
    else:
        X, Y = variables[2], variables[3]
        return hermite(m)(X) * hermite(n)(Y) * np.exp(-(X*X+Y*Y) / 2)


def check_qubit(ind, qures, peas):
    if ind == qures:
        print('Was Qubit')
        ind = np.argsort(peas)[-2]
    return ind


def compute_dist(Z, Zref):
    norm = np.nanmax(np.abs(Zref))
    Zref = np.tensordot(Zref / norm, np.ones(Z.shape[-1]), axes=0)
    return np.nansum(np.nansum((np.abs(Z) - np.abs(Zref)) ** 2, axis=0), axis=0)


def condition(this_mindist, these_mindists, indices):
    cond = False
    if this_mindist in indices.values():
        cond = True
        print("Already in : ", this_mindist)
    return cond


def decide(mindists, indices, qures, peas, freqs):
    counter = 0
    failstate = False
    this_mindist = check_qubit(mindists[0], qures, peas)
    while condition(this_mindist, mindists, indices):
        counter += 1
        this_mindist = mindists[counter]
        print("Switching to ", this_mindist)
        this_mindist = check_qubit(this_mindist, qures, peas)
        failstate = True
    return this_mindist, failstate


def plot_compare(Zref, Z, L, f):
    fig, axes = plt.subplots(1, 2, tight_layout=True)
    axes[1].imshow(Zref, cmap='Spectral', origin='lower')
    axes[0].imshow(Z, cmap='Spectral', origin='lower')
    fig.suptitle(f'L={L*1e9:.3f} nH, f={f * 1e-9:.4f} GHz')
    axes[0].axis('off')
    axes[1].axis('off')
    return fig


def find_modes(L, freqs, data, look_for, qures, peas):
    print(f"L={L*1e9:.3f}")
    print("qures : ", qures)
    N = 300
    X, Y = np.meshgrid(np.linspace(9e-4, 11e-4, N), np.linspace(0, 1e-4, N))

    allz = data[:, 2:]
    Z = np.empty((*X.shape, len(freqs)))
    for i in range(Z.shape[-1]):
        thisZ = gd(data[:, [0, 1]], allz[:, i], (X, Y), method='linear')
        Z[:, :, i] = thisZ / np.nanmax(np.abs(thisZ))
    
    q = 49
    _, _, w0t, _ = sapphire(q)
    upper = 0.7
    lower = 0.5
    how_many = 10
    w02 = np.linspace(lower * w0t, upper * w0t, how_many)** 2

    Xc = 1e-3
    r = np.sqrt((X - Xc) ** 2 + Y ** 2)
    phi = np.arctan2((X - Xc), Y) + np.pi / 2

    indices = {}

    rho = np.tensordot(2 * r * r, 1 / w02, axes=0)
    Zref = genlaguerre(0, 0)(rho) * np.exp(-rho/2)
    norm = np.tensordot(np.ones(Zref.shape[:2]), np.nanmax(np.abs(Zref), axis=(0, 1)), axes=0)
    Zref = np.tensordot(Zref / norm, np.ones(Z.shape[-1]), axes=0)
    mod_Z = np.tensordot(np.abs(Z), np.ones(how_many), axes=0)
    mod_Z = np.moveaxis(mod_Z, (2, 3), (3, 2))

    distances = np.nansum(np.nansum((mod_Z - np.abs(Zref)) ** 2, axis=0), axis=0)
    asort = np.argsort(distances.flatten())
    coords = np.unravel_index(asort[0], distances.shape)
    i = 0
    if coords[1] == qures:
        coords = list(coords)
        coords[1] = np.argsort(peas)[-2]
        coords = tuple(coords)


    print('Found', coords)
    w02ref = w02[coords[0]]
    rho = 2 * r * r / w02ref
    Xp = np.sqrt(2 / w02ref) * (X - Xc)
    Yp = np.sqrt(2 / w02ref) * Y
    variables = [rho, phi, Xp, Yp]
    indices["LG00"] = coords[1]
    # fig = plot_compare(Zref[:, :, coords[0], coords[1]], Z[:, :, indices["LG00"]], L, freqs[indices["LG00"]])

    for key, f0 in look_for.items():
        if key == "LG00":
            continue
        # print(key)
        loweri = np.argmin(np.abs(freqs - f0 * 1e9)) - 3
        upperi = np.argmin(np.abs(freqs - f0 * 1e9)) + 3
        LG = key[:2] == "LG"
        Zref = create_Zref(LG, int(key[2]), int(key[3]), variables)
        distances = compute_dist(Z[:, :, loweri:upperi], Zref)
        mindists = loweri + np.argsort(distances)

        indices[key], failstate = decide(mindists, indices, qures, peas, freqs)
        # fig = plot_compare(Zref, Z[:, :, indices[key]], L, freqs[indices[key]])

    return indices


def apply_pol(mindist, pol):
    new_mindist = {}
    whichpol = np.where(np.argmax(pol, axis=1) == 0, "L", "SH")
    for key, value in mindist.items():
        if key == "LG00":
            new_mindist["LG00L"] = value
            continue
        newkey = key[:4] + whichpol[value]
        if newkey in new_mindist:
            newkey += "b"
        new_mindist[newkey] = value
    return new_mindist


def find_modes_wrapper(name, qures, allpeas):
    metadata = loadmda(name)
    freqs = metadata['freqs']
    path = filepath(name)
    if not os.path.isfile(path + 'uz.csv'):
        print('No file found')
        return -1
    if len(freqs.shape) > 1:
        numL = freqs.shape[0]
        numf = freqs.shape[1]
    else:
        numL = 1
        numf = len(freqs)
        freqs = np.reshape(freqs, (1, numf))
    usecols = [2, 3, 4] if numL > 1 else [1, 2, 3]
    pols = np.loadtxt(
        path + "pol.csv", 
        comments="%",
        delimiter=",", 
        usecols=usecols
    ).reshape((numL, numf, 3))
    mindist_all = np.empty(numL, dtype=dict)
    counter = 0
    for i in range(numL):
        thisnumf = int(np.sum(freqs[i, :] != 0))
        cols = [2 + counter + j for j in range(thisnumf)]
        counter += thisnumf
        cols = np.append([0, 1], cols)
        converters = {j : realconv for j in cols}
        data = np.loadtxt(path + 'uz.csv', delimiter=',', comments='%', usecols=cols, converters=converters)
        mindist_all[i]= find_modes(metadata['L'][i], freqs[i, :thisnumf].T, data, metadata['look_for'], qures[i], allpeas[i])  # L, freqs, data, look_for, qures, peas
        mindist_all[i] = apply_pol(mindist_all[i], pols[i])
    
    metadata['mindist_all_z'] = mindist_all

    savemda(metadata)
