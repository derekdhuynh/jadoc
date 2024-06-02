import jax.scipy.linalg
import jax.experimental.sparse
from scipy.sparse import linalg
from jax.scipy.sparse import linalg
# import numpy as np
import jax.numpy as jnp
import jax.random
import time
from numba import njit,prange

def PerformJADOC(mC,mB0=None,iT=100,iTmin=10,dTol=1E-4,dTauH=1E-2,dAlpha=0.9,\
                 iS=None):
    """Joint Approximate Diagonalization under Orthogonality Constraints
    (JADOC)
    
    Authors: Ronald de Vlaming and Eric Slob
    Repository: https://www.github.com/devlaming/jadoc/
    
    Input
    ------
    mC : jnp.ndarray with shape (iK, iN, iN)
        iK Hermitian iN-by-iN matrices to jointly diagonalize
    
    mB0 : jnp.ndarray with shape (iN, iN), optional
        starting value for unitary transformation matrix such
        that mB@mC[i]@(mB.conj().T) is approximately diagonal for all i
    
    iT : int, optional
        maximum number of iterations; default=100
    
    iTmin : int, optional
        minimal number of iterations before convergence is tested; default=10
    
    dTol : float, optional
        stop if average magnitude elements gradient<dTol; default=1E-4
    
    dTauH : float, optional
        minimum value of second-order derivatives; default=1E-2
    
    dAlpha : float, optional
        regularization strength between zero and one; default=0.9
    
    iS : int, optional
        replace mC[i] by rank-iS approximation; default=None
        (set to ceil(iN/iK) under the default value)
    
    Output
    ------
    mB : jnp.ndarray with shape (iN, iN)
        unitary matrix such that mB@mC[i]@(mB.conj().T) is
        approximately diagonal for all i
    """
    print("Starting JADOC")
    (iK,iN,_)=mC.shape
    if iS is None:
        iS=(iN/iK)
        if (iS-int(iS))>0:
            iS=int(iS)+1
        else: 
            iS=int(iS)
    if iS==iN:
        print("Computing decomposition of input matrices")
    elif iS>iN:
        raise ValueError("Desired rank (iS) exceeds dimensionality" \
                         + " of input matrices (iN)")
    else: print("Computing low-dimensional approximation of input matrices")
    if mB0 is None: mB=jnp.eye(iN)
    elif mB0.shape!=(iN,iN):
        raise ValueError("Starting value transformation matrix" \
                         + " has wrong shape")
    else: mB=mB0
    bComplex=jnp.iscomplexobj(mC)
    if bComplex: mA=jnp.empty((iK,iN,iS),dtype="complex128")
    else: mA=jnp.empty((iK,iN,iS))
    print("Regularization strength = "+str(dAlpha))
    vAlphaLambda=jnp.empty(iK)
    for i in range(iK):
        mD=mC[i]-ConjT(mC[i])
        if bComplex: dMSD=(jnp.real(mD)**2).mean()+(jnp.imag(mD)**2).mean()
        else: dMSD=(mD**2).mean()
        if dMSD>jnp.finfo(float).eps:
            if bComplex:
                raise ValueError("Input matrices are not Hermitian")
            else:
                raise ValueError("Input matrices are not real symmetric")
        if iS<iN:
            # So jax doesn't have this implemented in the main package but it is in the experiemental
            # one as of v0.4.28
            # https://pytorch.org/docs/stable/generated/torch.lobpcg.html -> torch has this implemented
            # (vD, mP) = linalg.eigsh(mC[i], k=iS)
            # raise ValueError("Not supported in jax implementation (for now)")
            print(f'This the number of eigenvalues we want: {iS}')
            #zeros = jnp.zeros((iN, min(iN // 5 - 1, iS)))
            #zeros = jnp.zeros((iN, iS))
            seed = 943898
            key = jax.random.key(seed)
            initial_search_dirs = jax.random.normal(key, shape=(iN, iS), dtype=mC.dtype)
            (vD, mP, _) = jax.experimental.sparse.linalg.lobpcg_standard(mC[i], initial_search_dirs)
        else:
            (vD,mP)=jnp.linalg.eigh(mC[i])
        vD=abs(vD)
        vAlphaLambda = vAlphaLambda.at[i].set(dAlpha*((vD.sum())/iN))
        mA = mA.at[i].set(((1-dAlpha)**0.5)*mP*(jnp.sqrt(vD)[None,:]))
        if mB0 is not None: 
            mA = mA.at[i].set(jnp.dot(mB,mA[i]))
    (mP,vD,mC)=(None,None,None)
    print("Starting quasi-Newton algorithm with line search (golden section)")
    bConverged=False
    for t in range(iT):
        (dLoss,mDiags,dRMSG,mU)=ComputeLoss(mA,vAlphaLambda,bComplex,dTauH)
        if dRMSG<dTol and t>=iTmin:
            bConverged=True
            break
        dStepSize=PerformGoldenSection(mA,mU,mB,vAlphaLambda,bComplex)
        print("ITER "+str(t)+": L="+str(round(dLoss,3))+", RMSD(g)=" \
              +str(round(dRMSG,6))+", step="+str(round(dStepSize,3)))
        (mB,mA)=UpdateEstimates(mA,mU,mB,dStepSize)
    if not(bConverged):
        print("WARNING: JADOC did not converge. Reconsider data or thresholds")
    print("Returning transformation matrix B")
    return mB

def ComputeLoss(mA,vAlphaLambda,bComplex,dTauH=None,bLossOnly=False):
    if bComplex:
        mDiags=((jnp.real(mA)**2).sum(axis=2))+((jnp.imag(mA)**2).sum(axis=2))\
            +vAlphaLambda[:,None]
    else:
        mDiags=((mA**2).sum(axis=2))+vAlphaLambda[:,None]
    (iK,iN,iS)=mA.shape
    dLoss=0.5*(jnp.log(mDiags).sum())/iK
    if bLossOnly:
        return dLoss
    else:
        if bComplex:
            mF = jnp.zeros((iN,iN),dtype="complex128")
            mF = ComputeFComplex(mF,mA,mDiags,iK,iN)
        else:
            mF=jnp.zeros((iN,iN))
            mF=ComputeFReal(mF,mA,mDiags,iK,iN)
        mG=(mF-ConjT(mF))
        if bComplex:
            dRMSG=jnp.sqrt((((jnp.real(mG)**2).sum())+((jnp.imag(mG)**2).sum()))\
                          /(iN*(iN-1)))
        else:
            dRMSG=jnp.sqrt(((mG**2).sum())/(iN*(iN-1)))
        mH=(mDiags[:,:,None]/mDiags[:,None,:]).mean(axis=0)
        mH=mH+mH.T-2.0
        mH = mH.at[mH<dTauH].set(dTauH)
        mU = -mG/mH
        return dLoss,mDiags,dRMSG,mU

#@njit
@jax.jit
def ComputeFComplex(mF,mA,mDiags,iK,iN):
    for i in range(mA.shape[0]):
        vDiags=(mDiags[i]).reshape((mA.shape[1], 1))
        mF+=jnp.dot(mA[i]/vDiags,mA[i].conj().T)
    mF=mF/iK
    return mF

#@njit
@jax.jit
def ComputeFReal(mF,mA,mDiags,iK,iN):
    for i in range(mA.shape[0]):
        vDiags=(mDiags[i]).reshape((mA.shape[1], 1))
        mF+=jnp.dot(mA[i]/vDiags,mA[i].T)
    mF=mF/iK
    return mF

def PerformGoldenSection(mA,mU,mB,vAlphaLambda,bComplex):
    dTheta = 2 / (1 + (5 ** 0.5))
    iIter = 0
    iMaxIter = 15
    iGuesses = 4
    (dStepLB, dStepUB) = (0, 1)
    bLossOnlyGold = True
    (iK, iN, iS) = mA.shape
    mR = jax.scipy.linalg.expm(mU)

    if bComplex:
        mAS = jnp.empty((iGuesses, iK, iN, iS), dtype="complex128")
    else:
        mAS = jnp.empty((iGuesses, iK, iN, iS))

    mAS = mAS.at[0].set(mA.copy())
    mAS = mAS.at[1].set(RotateData(mR, mA.copy()))
    mAS = mAS.at[2].set((1 - dTheta) * mAS[1] + dTheta * mAS[0])
    mAS = mAS.at[3].set((1 - dTheta) * mAS[0] + dTheta * mAS[1])
    (mA, mR) = (None, None)

    dLoss2 = ComputeLoss(mAS[2], vAlphaLambda, bComplex, bLossOnly=bLossOnlyGold)
    dLoss3 = ComputeLoss(mAS[3], vAlphaLambda, bComplex, bLossOnly=bLossOnlyGold)

    while iIter < iMaxIter:
        if dLoss2 < dLoss3:
            mAS = mAS.at[1].set(mAS[3])
            mAS = mAS.at[3].set(mAS[2])
            dLoss3 = dLoss2
            dStepUB = dStepLB + dTheta * (dStepUB - dStepLB)
            mAS = mAS.at[2].set(mAS[1] - dTheta * (mAS[1] - mAS[0]))
            dLoss2 = ComputeLoss(mAS[2], vAlphaLambda, bComplex, bLossOnly=bLossOnlyGold)
        else:
            mAS = mAS.at[0].set(mAS[2])
            mAS = mAS.at[2].set(mAS[3])
            dLoss2 = dLoss3
            dStepLB = dStepUB - dTheta * (dStepUB - dStepLB)
            mAS = mAS.at[3].set(mAS[0] + dTheta * (mAS[1] - mAS[0]))
            dLoss3 = ComputeLoss(mAS[3], vAlphaLambda, bComplex, bLossOnly=bLossOnlyGold)
        iIter += 1

    return jnp.log(1 + (dStepLB * (jnp.exp(1) - 1)))
    #dTheta=2/(1+(5**0.5))
    #iIter=0
    #iMaxIter=15
    #iGuesses=4
    #(dStepLB,dStepUB)=(0,1)
    #bLossOnlyGold=True
    #(iK,iN,iS)=mA.shape
    #mR=jax.scipy.linalg.expm(mU)
    #if bComplex: mAS=jnp.empty((iGuesses,iK,iN,iS),dtype="complex128")
    #else: mAS=jnp.empty((iGuesses,iK,iN,iS))
    #mAS[0]=mA.copy()
    #mAS[1]=RotateData(mR,mA.copy())
    #mAS[2]=(1-dTheta)*mAS[1]+dTheta*mAS[0]
    #mAS[3]=(1-dTheta)*mAS[0]+dTheta*mAS[1]
    #(mA,mR)=(None,None)
    #dLoss2=ComputeLoss(mAS[2],vAlphaLambda,bComplex,bLossOnly=bLossOnlyGold)
    #dLoss3=ComputeLoss(mAS[3],vAlphaLambda,bComplex,bLossOnly=bLossOnlyGold)
    #while iIter<iMaxIter:
    #    if (dLoss2<dLoss3):
    #        mAS[1]=mAS[3]
    #        mAS[3]=mAS[2]
    #        dLoss3=dLoss2
    #        dStepUB=dStepLB+dTheta*(dStepUB-dStepLB)
    #        mAS[2]=mAS[1]-dTheta*(mAS[1]-mAS[0])
    #        dLoss2=ComputeLoss(mAS[2],vAlphaLambda,bComplex,\
    #                           bLossOnly=bLossOnlyGold)
    #    else:
    #        mAS[0]=mAS[2]
    #        mAS[2]=mAS[3]
    #        dLoss2=dLoss3
    #        dStepLB=dStepUB-dTheta*(dStepUB-dStepLB)
    #        mAS[3]=mAS[0]+dTheta*(mAS[1]-mAS[0])
    #        dLoss3=ComputeLoss(mAS[3],vAlphaLambda,bComplex,\
    #                           bLossOnly=bLossOnlyGold)
    #    iIter+=1
    #return jnp.log(1+(dStepLB*(jnp.exp(1)-1)))

def UpdateEstimates(mA,mU,mB,dStepSize):
    mR=jax.scipy.linalg.expm(dStepSize*mU)
    mB=jnp.dot(mR,mB)
    mA=RotateData(mR,mA)
    return mB,mA

#@njit
@jax.jit
def RotateData(mR,mData):
    iK=mData.shape[0]
    for i in range(iK):
        mData = mData.at[i].set(jnp.dot(mR,mData[i]))
    return mData

def ConjT(mA):
    if jnp.iscomplexobj(mA):
        return mA.conj().T
    else:
        return mA.T

def SimulateData(iK,iN,iR,dAlpha,bComplex=False,bPSD=True):
    if bComplex: sType1="Hermitian "
    else: sType1="real symmetric "
    if bPSD: sType2="positive (semi)-definite "
    else: sType2=""
    print("Simulating "+str(iK)+" distinct "+str(iN)+"-by-"+str(iN)+" " \
          +sType1+sType2+"matrices with alpha="+str(dAlpha) \
              +", for run "+str(iR))
    iMainSeed=15348091
    iRmax=10000
    if iR>=iRmax:
        return
    key = jax.random.key(iMainSeed)
    vSeed= jax.random.randint(key, (iRmax,), 0, iMainSeed)
    iSeed=vSeed[iR]
    new_key = jax.random.key(iSeed)
    if bComplex:
        # mX=rng.normal(size=(iN,iN))+1j*rng.normal(size=(iN,iN))
        new_key, subkey = jax.random.split(new_key)
        mX = jax.random.normal(new_key, (iN, iN)) + 1j * jax.random.normal(subkey, shape=(iN,iN))
        mC = jnp.empty((iK,iN,iN),dtype="complex128")
    else:
        mX = jax.random.normal(new_key, shape=(iN,iN))
        mC = jnp.empty((iK,iN,iN))
    for i in range(0,iK):
        new_key, subkey = jax.random.split(new_key)
        if bComplex:
            mXk = jax.random.normal(new_key, shape=(iN,iN)) +1j * jax.random.normal(subket, shape=(iN,iN))
        else:
            mXk= jax.random.normal(new_key, shape=(iN,iN))
        mXk=dAlpha*mX+(1-dAlpha)*mXk
        mR=jax.scipy.linalg.expm(mXk-ConjT(mXk))

        key, subkey = jax.random.split(subkey)
        vD = jax.random.normal(subkey, shape=(iN,))
        if bPSD:
            vD=vD**2
        # TODO: A reassignment happens here, see if we could replace it with something else
        # mC[i]=jnp.dot(mR*(vD[None,:]),ConjT(mR))
        mC = mC.at[i].set(jnp.dot(mR*(vD[None,:]),ConjT(mR)))
    return mC

def Test():
    iK=5
    iN=500
    iR=1
    dAlpha=0.9

    mC=SimulateData(iK,iN,iR,dAlpha)
    dTimeStart=time.time()

    # Make a jitted version of the function
    mB = PerformJADOC(mC,dAlpha=.95,dTol=1E-5,iT=1000, iS=int(iN // 5.1))
    print(mB)
    dTime = time.time()-dTimeStart
    print("Runtime: "+str(round(dTime,3))+" seconds")
    mD=jnp.empty((iK,iN,iN))
    for i in range(iK):
        mD = mD.at[i].set(jnp.dot(jnp.dot(mB,mC[i]),mB.T))
    dSS_C=0
    dSS_D=0
    for i in range(iK):
        mOffPre=mC[i]-jnp.diag(jnp.diag(mC[i]))
        mOffPost=mD[i]-jnp.diag(jnp.diag(mD[i]))
        dSS_C+=(mOffPre**2).sum()
        dSS_D+=(mOffPost**2).sum()
    dRMS_C=jnp.sqrt(dSS_C/(iN*(iN-1)*iK))
    dRMS_D=jnp.sqrt(dSS_D/(iN*(iN-1)*iK))
    print("Root-mean-square deviation off-diagonals before transformation: " \
          +str(round(dRMS_C,6)))
    print("Root-mean-square deviation off-diagonals after transformation: " \
          +str(round(dRMS_D,6)))
