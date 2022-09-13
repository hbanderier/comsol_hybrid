import numpy as np
import pickle as pkl

comps1 = ['XX', 'XY', 'XZ', 'YY', 'YZ', 'ZZ']
comps2 = ['11', '12', '13', '22', '23', '33']


def is_numeric(char):
    return char in [str(i) for i in range(10)]

    
def realconv(x):
    return np.real(complex(x.decode("utf-8").replace('i', 'j')))


def filepath(name):
    return './data/' + name + "/"


def savemda(metadata):
    with open(filepath(metadata["name"]) + "metadata.pkl", "wb") as handle:
        pkl.dump(metadata, handle)


def loadmda(name):
    with open(filepath(name) + "metadata.pkl", "rb") as handle:
        return pkl.load(handle)
    

def setup_int(javaobj, name, dic):
    intobj = javaobj.result().numerical(name)
    for key, value in dic.items():
        intobj.set(key, value)
    return intobj


def extract_freqs(javaobj, model, sol, and_l=False):
    info1 = javaobj.sol(sol).getSolutioninfo()
    if and_l:
        is_sweep = info1.getLevels() > 1
    if and_l and is_sweep:
        L = np.array(info1.getVals(1, [])[0])
        freqs1 = -np.array(info1.getValsImag(2, [])) / 2 / np.pi
        maxf = info1.getMaxInner(None)
        freqs2 = np.zeros((len(L), maxf))
        counter = 0
        for i in range(len(L)):
            howmany = info1.getMaxInner([i + 1])
            print(howmany)
            freqs2[i, 0:howmany] = freqs1[0, counter:(counter + howmany)]
            counter = counter + howmany
        freqs1 = freqs2
    elif and_l:
        L = np.array([float(model.parameter('L_junc')[:-4])]) * 1e-9
        freqs1 = -np.array(info1.getValsImag(0, [])).reshape(1, -1) / 2 / np.pi
    else:
        freqs1 = -np.array(info1.getValsImag(0, [])).reshape(1, -1) / 2 / np.pi
    if and_l :
        return freqs1, L, is_sweep
    else:
        return freqs1
