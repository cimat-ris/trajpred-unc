import numpy as np

# Given a set of pdf values from samples on the pdf, and a pdf value, deduce alpha
def get_alpha(pdf_values, fa):
	# Sort samples pdf values
    sorted_pdf_values = sorted(pdf_values, reverse=True)
	# Select all samples for which the pdf value is above the one of GT
    ind = np.where(sorted_pdf_values < fa)[0]
    if ind.shape[0] > 0:
		# Probability mass above fa
        alpha_fa =  sum(sorted_pdf_values[:ind[0]])/sum(sorted_pdf_values)
    else:
		# All samples have a pdf value >= fa
        alpha_fa = 1.0
    return alpha_fa

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
