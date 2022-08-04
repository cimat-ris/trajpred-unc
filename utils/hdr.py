import numpy as np

# Sort samples with respect to their density value
def sort_sample(sample_pdf):
    # Samples from the pdf are sorted in decreasing order
    sample_pdf_zip = zip(sample_pdf, sample_pdf/np.sum(sample_pdf))
    return sorted(sample_pdf_zip, key=lambda x: x[1], reverse=True)

# Given a value on the pdf, deduce alpha
def get_alpha(orden, f_pdf):
    # Predicted HDR
    ind = np.where(np.array(orden)[:,0] >= f_pdf)[0]
    ind = 0 if ind.size == 0 else ind[-1] # Validamos que no sea el primer elemento mas grande
    alpha = np.array(orden)[:ind+1,1].sum()
    return alpha

def bs(orden,imin,imax,f_pdf):
    if imax==imin:
        return imin
    ind = (imin+imax)//2
    if ind==imax:
        return imin
    if ind==imin:
        return imax
    if orden[ind]>f_pdf:
        return bs(orden,ind,imax,f_pdf)
    return bs(orden,imin,ind,f_pdf)

# Given a value on the pdf, deduce alpha
def get_alpha_bs(orden, f_pdf):
    # Predicted HDR
    ind = bs(np.array(orden)[:,0],0,len(orden)-1,f_pdf)
    alpha = 1 - np.array(orden)[:ind+1,1].sum()
    return alpha

# Given a value of alpha, deduce the value of the density
def get_falpha(orden, alpha):
    # We find f_gamma(HDR) from the pdf samples
    orden_idx, orden_val = zip(*orden)
    ind = np.where(np.cumsum(orden_val) >= alpha)[0]
    if ind.shape[0] == 0:
        fa = orden[-1][0]
    else:
        fa = orden_idx[ind[0]]
    return fa
