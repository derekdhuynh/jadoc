import torch
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def PerformJADOC(mC, mB0=None, iT=100, iTmin=10, dTol=1E-4, dTauH=1E-2, dAlpha=0.9, iS=None):
    print("Starting JADOC")
    (iK, iN, _) = mC.shape
    if iS is None:
        iS = (iN // iK) + (iN % iK > 0)
    if iS == iN:
        print("Computing decomposition of input matrices")
    elif iS > iN:
        raise ValueError("Desired rank (iS) exceeds dimensionality of input matrices (iN)")
    else:
        print("Computing low-dimensional approximation of input matrices")
    if mB0 is None:
        mB = torch.eye(iN, dtype=mC.dtype).to(device)
    elif mB0.shape != (iN, iN):
        raise ValueError("Starting value transformation matrix has wrong shape")
    else:
        mB = mB0
    bComplex = torch.is_complex(mC)
    if bComplex:
        mA = torch.empty((iK, iN, iS), dtype=torch.complex128).to(device)
    else:
        mA = torch.empty((iK, iN, iS), dtype=mC.dtype).to(device)
    print("Regularization strength = " + str(dAlpha))
    vAlphaLambda = torch.empty(iK, dtype=mC.dtype).to(device)
    for i in range(iK):
        mD = mC[i] - mC[i].T.conj()
        dMSD = (mD.abs()**2).mean()
        if dMSD > torch.finfo(mC.dtype).eps:
            raise ValueError("Input matrices are not Hermitian")
        if iS < iN:
            vD, mP = torch.lobpcg(mC[i], k=iS, method='ortho')
        else:
            vD, mP = torch.linalg.eigh(mC[i])
        vD = vD.abs()
        vAlphaLambda[i] = dAlpha * (vD.sum() / iN)
        mA[i] = ((1 - dAlpha)**0.5) * mP * vD.sqrt()
        if mB0 is not None:
            mA[i] = mB @ mA[i]
    mP, vD, mC = None, None, None
    print("Starting quasi-Newton algorithm with line search (golden section)")
    bConverged = False
    for t in range(iT):
        dLoss, mDiags, dRMSG, mU = ComputeLoss(mA, vAlphaLambda, bComplex, dTauH)
        if dRMSG < dTol and t >= iTmin:
            bConverged = True
            break
        dStepSize = PerformGoldenSection(mA, mU, mB, vAlphaLambda, bComplex)
        print(f"ITER {t}: L={round(dLoss.item(), 3)}, RMSD(g)={round(dRMSG.item(), 6)}, step={round(dStepSize.item(), 3)}")
        mB, mA = UpdateEstimates(mA, mU, mB, dStepSize)
    if not bConverged:
        print("WARNING: JADOC did not converge. Reconsider data or thresholds")
    print("Returning transformation matrix B")
    return mB

def ComputeLoss(mA, vAlphaLambda, bComplex, dTauH=None, bLossOnly=False):
    if bComplex:
        mDiags = (mA.abs()**2).sum(dim=2) + vAlphaLambda[:, None]
    else:
        mDiags = (mA**2).sum(dim=2) + vAlphaLambda[:, None]
    iK, iN, iS = mA.shape
    dLoss = 0.5 * torch.log(mDiags).sum() / iK
    if bLossOnly:
        return dLoss
    else:
        if bComplex:
            mF = torch.zeros((iN, iN), dtype=torch.complex128).to(device)
            mF = ComputeFComplex(mF, mA, mDiags, iK, iN)
        else:
            mF = torch.zeros((iN, iN), dtype=mA.dtype).to(device)
            mF = ComputeFReal(mF, mA, mDiags, iK, iN)
        mG = mF - mF.T.conj()
        dRMSG = torch.sqrt((mG.abs()**2).sum() / (iN * (iN - 1)))
        mH = (mDiags[:, :, None] / mDiags[:, None, :]).mean(dim=0)
        mH = mH + mH.T - 2.0
        mH[mH < dTauH] = dTauH
        mU = -mG / mH
        return dLoss, mDiags, dRMSG, mU

def ComputeFComplex(mF, mA, mDiags, iK, iN):
    for i in range(iK):
        vDiags = mDiags[i].reshape((iN, 1))
        mF += (mA[i] / vDiags) @ mA[i].T.conj()
    mF /= iK
    return mF

def ComputeFReal(mF, mA, mDiags, iK, iN):
    for i in range(iK):
        vDiags = mDiags[i].reshape((iN, 1))
        mF += (mA[i] / vDiags) @ mA[i].T
    mF /= iK
    return mF

def PerformGoldenSection(mA, mU, mB, vAlphaLambda, bComplex):
    dTheta = 2 / (1 + torch.sqrt(torch.tensor(5.0)))
    iIter = 0
    iMaxIter = 15
    iGuesses = 4
    dStepLB, dStepUB = 0, 1
    bLossOnlyGold = True
    iK, iN, iS = mA.shape
    mR = torch.matrix_exp(mU)
    if bComplex:
        mAS = torch.empty((iGuesses, iK, iN, iS), dtype=torch.complex128).to(device)
    else:
        mAS = torch.empty((iGuesses, iK, iN, iS), dtype=mA.dtype).to(device)
    mAS[0] = mA.clone()
    mAS[1] = RotateData(mR, mA.clone())
    mAS[2] = (1 - dTheta) * mAS[1] + dTheta * mAS[0]
    mAS[3] = (1 - dTheta) * mAS[0] + dTheta * mAS[1]
    mA, mR = None, None
    dLoss2 = ComputeLoss(mAS[2], vAlphaLambda, bComplex, bLossOnly=bLossOnlyGold)
    dLoss3 = ComputeLoss(mAS[3], vAlphaLambda, bComplex, bLossOnly=bLossOnlyGold)
    while iIter < iMaxIter:
        if dLoss2 < dLoss3:
            mAS[1] = mAS[3]
            mAS[3] = mAS[2]
            dLoss3 = dLoss2
            dStepUB = dStepLB + dTheta * (dStepUB - dStepLB)
            mAS[2] = mAS[1] - dTheta * (mAS[1] - mAS[0])
            dLoss2 = ComputeLoss(mAS[2], vAlphaLambda, bComplex, bLossOnly=bLossOnlyGold)
        else:
            mAS[0] = mAS[2]
            mAS[2] = mAS[3]
            dLoss2 = dLoss3
            dStepLB = dStepUB - dTheta * (dStepUB - dStepLB)
            mAS[3] = mAS[0] + dTheta * (mAS[1] - mAS[0])
            dLoss3 = ComputeLoss(mAS[3], vAlphaLambda, bComplex, bLossOnly=bLossOnlyGold)
        iIter += 1
    return torch.log(1 + (dStepLB * (torch.exp(torch.tensor(1.0)) - 1)))

def UpdateEstimates(mA, mU, mB, dStepSize):
    mR = torch.matrix_exp(dStepSize * mU)
    mB = mR @ mB
    mA = RotateData(mR, mA)
    return mB, mA

def RotateData(mR, mData):
    iK = mData.shape[0]
    for i in range(iK):
        mData[i] = mR @ mData[i]
    return mData

def ConjT(mA):
    if torch.is_complex(mA):
        return mA.T.conj()
    else:
        return mA.T

def SimulateData(iK, iN, iR, dAlpha, bComplex=False, bPSD=True, device='cpu'):
    if bComplex:
        sType1 = "Hermitian "
    else:
        sType1 = "real symmetric "
    if bPSD:
        sType2 = "positive (semi)-definite "
    else:
        sType2 = ""
    print(f"Simulating {iK} distinct {iN}-by-{iN} {sType1}{sType2}matrices with alpha={dAlpha}, for run {iR}")
    iMainSeed = 15348091
    iRmax = 10000
    if iR >= iRmax:
        return
    torch.manual_seed(iMainSeed)
    vSeed = torch.randint(0, iMainSeed, (iRmax,))
    iSeed = vSeed[iR]
    torch.manual_seed(iSeed)
    if bComplex:
        mX = torch.randn(iN, iN, dtype=torch.complex128).to(device)
        mC = torch.empty((iK, iN, iN), dtype=torch.complex128).to(device)
    else:
        mX = torch.randn(iN, iN, dtype=torch.float64).to(device)
        mC = torch.empty((iK, iN, iN), dtype=mX.dtype).to(device)
    for i in range(iK):
        if bComplex:
            mXk = torch.randn(iN, iN, dtype=torch.complex128).to(device)
        else:
            mXk = torch.randn(iN, iN, dtype=mX.dtype).to(device)
        mXk = dAlpha * mX + (1 - dAlpha) * mXk
        mR = torch.matrix_exp(mXk - mXk.T.conj())
        vD = torch.randn(iN, dtype=mX.dtype).to(device)
        if bPSD:
            vD = vD**2
        mC[i] = mR @ torch.diag(vD) @ mR.T.conj()
    return mC

def Test(iN=500, iK=5):
    iR = 1
    dAlpha = 0.9
    mC = SimulateData(iK, iN, iR, dAlpha)
    dTimeStart = time.time()
    mB = PerformJADOC(mC, dAlpha=0.95, dTol=1E-5, iT=1000)
    dTime = time.time() - dTimeStart
    print(f"Runtime: {round(dTime, 3)} seconds")
    mD = torch.empty((iK, iN, iN))
    for i in range(iK):
        mD[i] = mB @ mC[i] @ mB.T
    dSS_C = 0
    dSS_D = 0
    for i in range(iK):
        mOffPre = mC[i] - torch.diag(torch.diag(mC[i]))
        mOffPost = mD[i] - torch.diag(torch.diag(mD[i]))
        dSS_C += (mOffPre**2).sum()
        dSS_D += (mOffPost**2).sum()
    dRMS_C = torch.sqrt(dSS_C / (iN * (iN - 1) * iK))
    dRMS_D = torch.sqrt(dSS_D / (iN * (iN - 1) * iK))
    print(f"Root-mean-square deviation off-diagonals before transformation: {round(dRMS_C.item(), 6)}")
    print(f"Root-mean-square deviation off-diagonals after transformation: {round(dRMS_D.item(), 6)}")
