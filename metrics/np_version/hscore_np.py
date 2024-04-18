import numpy as np

def getCov(X):
    X_mean=X-np.mean(X,axis=0,keepdims=True)
    cov = np.divide(np.dot(X_mean.T, X_mean), len(X)-1) 
    return cov

def getHscore(f,Z):
    #Z=np.argmax(Z, axis=1)
    Covf=getCov(f)
    alphabetZ=list(set(Z))
    g=np.zeros_like(f)
    for z in alphabetZ:
        Ef_z=np.mean(f[Z==z, :], axis=0)
        g[Z==z]=Ef_z
    
    Covg=getCov(g)
    score=np.trace(np.dot(np.linalg.pinv(Covf,rcond=1e-15), Covg))
    return score