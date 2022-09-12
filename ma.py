import numpy as np

def ma(x, wlen):
    mean = []
    std = []
    
    tlist = []
    
    sumt = 0
    for k, el in enumerate(x):
        if len(tlist) < wlen:
            tlist.append(el)
            sumt += el
        else:
            el_out = tlist.pop(0)
            tlist.append(el)
            sumt += el - el_out
        mean.append(sumt / len(tlist))
        
        tstd = 0
        for j, ss in enumerate(tlist):
            tstd += (ss - mean[-1]) ** 2
        std.append(np.sqrt(tstd / len(tlist)))
        
    return np.array(mean), np.array(std)


def ma_np(x, wlen):
    mean = []
    std = []

    tlist = []

    for k, el in enumerate(x):
        if len(tlist) < wlen:
            tlist.append(el)
        else:
            el_out = tlist.pop(0)
            tlist.append(el)

        mean.append(np.array(tlist).mean())
        std.append((np.array(tlist) - mean[-1]).std())

    return np.array(mean), np.array(std)
