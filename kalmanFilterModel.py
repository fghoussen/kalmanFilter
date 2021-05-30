#!/usr/bin/env python3

"""Kalman filter model"""

import os
import itertools
import h5py
import numpy as np
import numpy.linalg as npl

kfType = np.double

class kalmanFilterModel():
    """Kalman filter model"""

    def __init__(self, example):
        """Initialize"""

        # Initialize members.
        self.sim = {"simItNb": 0}
        self.msr = []
        self.example = example
        self.time = np.array([], dtype=kfType)
        self.states = {}
        self.outputs = {}
        self.save = {"predictor": {}, "corrector": {}}
        for key in ["simCLV", "simPrN", "simDgP"]:
            self.save["predictor"][key] = {}
        for key in ["simKGn", "simInv"]:
            self.save["corrector"][key] = {}
        self.clear()

    def clear(self):
        """Clear previous results"""

        # Clear previous measurements.
        self.msr = []

        # Clear previous time.
        self.time = np.array([], dtype=kfType)
        self.sim["simItNb"] = 0

        # Clear previous results.
        self.states.clear()
        for key in self.example.getStateKeys():
            self.states[key] = np.array([], dtype=kfType)
        self.outputs.clear()
        for key in self.example.getOutputKeys():
            self.outputs[key] = np.array([], dtype=kfType)

        # Clear previous predictor variables.
        self.save["predictor"]["simCLV"]["deltaAcc"] = {}
        self.save["predictor"]["simCLV"]["deltaAcc"]["AX"] = []
        self.save["predictor"]["simCLV"]["deltaAcc"]["AY"] = []
        self.save["predictor"]["simCLV"]["deltaAcc"]["AZ"] = []
        self.save["predictor"]["simCLV"]["roll"] = []
        self.save["predictor"]["simCLV"]["pitch"] = []
        self.save["predictor"]["simCLV"]["yaw"] = []
        self.save["predictor"]["simTEM"] = np.array([], dtype=kfType)
        for key in ["simPrN", "simDgP"]:
            for subKey in self.example.getStateKeys():
                self.save["predictor"][key][subKey] = []

        # Clear previous corrector variables.
        for key in self.example.getOutputKeys():
            self.save["corrector"]["simKGn"]["T"] = []
            self.save["corrector"]["simKGn"][key] = []
            self.save["corrector"]["simInv"][key] = {}
            self.save["corrector"]["simInv"][key]["T"] = []
            self.save["corrector"]["simInv"][key]["vecI"] = []
            self.save["corrector"]["simInv"][key]["vecZ"] = []

        # Remove H5 file.
        self.removeH5()

    def isSolved(self):
        """Check if solve has been done"""

        # Check if solve has been done.
        if len(self.time) == 0:
            return False
        return True

    def setUpSimPrm(self, sim, fcdTf):
        """Setup solver: simulation parameters"""

        # Set up solver parameters.
        for key in sim:
            if key.find("prm") == 0 or key.find("icd") == 0:
                self.sim[key] = float(sim[key].text())
        self.sim["fcdTf"] = float(fcdTf)

    def setUpMsrPrm(self, msr):
        """Setup solver: measurement parameters"""

        # Organise solver measurements.
        msrDic = {}
        for txt in msr:
            msrData = msr[txt]
            prmSigma = float(txt.split(";")[4].split()[1])
            self.organiseMsrPrm(msrData, prmSigma, msrDic)

        # Order solver measurements.
        self.msr = []
        for time in sorted(msrDic, reverse=True): # Reverse: get data poped in order (corrector).
            msrDic[time].sort(key=lambda t: t[4], reverse=True) # Small (accurate) sigma at the end.
            self.msr.append((time, msrDic[time]))

    @staticmethod
    def organiseMsrPrm(msrData, prmSigma, msrDic):
        """Organise measurements"""

        # Get measurement.
        posX, posY, posZ = None, None, None
        if msrData["msrType"] == "pos":
            posX, posY, posZ = msrData["X"], msrData["Y"], msrData["Z"]

        # Append measurement.
        for idx, time in enumerate(msrData["T"]):
            if time not in msrDic:
                msrDic[time] = []
            if msrData["msrType"] == "pos":
                msrDic[time].append(("pos", posX[idx], posY[idx], posZ[idx], prmSigma))

    def setLTI(self, matA, matB, matC, matD):
        """Set Linear Time Invariant matrices"""

        # Set matrices.
        self.sim["matA"] = matA
        self.sim["matB"] = matB
        self.sim["matC"] = matC
        self.sim["matD"] = matD
        if self.sim["prmVrb"] >= 3:
            print("  "*2+"Linear Time Invariant system:")
            self.printMat("A", self.sim["matA"])
            if self.sim["matB"] is not None:
                self.printMat("B", self.sim["matB"])
            self.printMat("C", self.sim["matC"])
            if self.sim["matD"] is not None:
                self.printMat("D", self.sim["matD"])

    def solve(self):
        """Solve based on Kalman filter"""

        # Don't solve if we have already a solution.
        if self.isSolved():
            return

        # Initialize states.
        self.removeH5()
        if self.sim["prmVrb"] >= 1:
            print("  "*2+"Initialisation:")
        time = 0.
        states = self.example.initStates(self.sim)
        vecU = self.example.computeControlLaw(states, time, self.save["predictor"])
        outputs = self.computeOutputs(states, vecU)
        matP = self.example.initStateCovariance(self.sim)
        if self.sim["prmVrb"] >= 2:
            self.printMat("X", np.transpose(states))
            self.printMat("Y", np.transpose(outputs))
            self.printMat("P", matP)
        self.savePredictor(time, states, outputs, matP)

        # Solve: https://www.kalmanfilter.net/multiSummary.html.
        prmVrb, prmDt, prmTf = self.sim["prmVrb"], self.sim["prmDt"], self.sim["fcdTf"]
        while time < prmTf:
            # Cut off time.
            if time+prmDt > prmTf:
                prmDt = prmTf-time

            # Set verbose only for some multiple of iterations.
            self.sim["simItNb"] = self.sim["simItNb"] + 1
            self.sim["prmVrb"] = -1*prmVrb
            if self.sim["simItNb"]%self.sim["prmVrbIt"] == 0:
                self.sim["prmVrb"] = prmVrb

            # Solve (predictor/corrector) with Kalman filter.
            timeDt, msrInDeltaT = self.isMsrInDeltaT(time, prmDt)
            if timeDt > 0.:
                states, matP = self.predictor(time, timeDt, states, matP)
            if msrInDeltaT:
                states, matP = self.corrector(matP, states)

            # Increase time.
            time = time+timeDt

        # Verbose.
        self.sim["prmVrb"] = prmVrb
        if self.sim["prmVrb"] >= 1:
            print("  "*2+"End: time %.3f, iteration %d" % (time, self.sim["simItNb"]))

    def isMsrInDeltaT(self, time, prmDt):
        """Check if a measure is about to happen in deltaT"""

        # Look for next measurement.
        timeDt, msrInDeltaT = prmDt, False
        nbMsr = len(self.msr)
        if nbMsr > 0:
            timeMsr = self.msr[nbMsr-1][0]
            if time <= timeMsr <= time+prmDt:
                timeDt = timeMsr-time
                msrInDeltaT = True

        return timeDt, msrInDeltaT

    def corrector(self, matP, states):
        """Solve corrector step"""

        # Look for measurement.
        msrData = self.msr.pop() # Get measurement out of the list.
        newStates, newMatP = self.computeCorrection(msrData, matP, states)

        return newStates, newMatP

    def computeCorrection(self, msrData, matP, states):
        """Compute correction"""

        # Get measurement data.
        newTime = msrData[0] # Cut off time to measurement time.
        if newTime >= self.sim["fcdTf"]:
            self.sim["prmVrb"] = np.abs(self.sim["prmVrb"])
        msrLst = msrData[1]
        if self.sim["prmVrb"] >= 1:
            print("  "*2+"Corrector: time %.3f, iteration %d" % (newTime, self.sim["simItNb"]))

        # Get measurement z_{n}.
        vecZ, matH, matR = self.getMsr(msrLst)
        self.saveH5("Z", newTime, vecZ, self.example.getOutputKeys())

        # Compute Kalman gain K_{n}.
        matK = self.computeKalmanGain(matP, matH, matR)

        # Update estimate with measurement: x_{n,n} = x_{n,n-1} + K_{n}*(z_{n} - H*x_{n,n-1}).
        #
        # The Kalman gain tells how much you need to change your estimate by given a measurement:
        #   x_{n,n} = x_{n,n-1} + K_{n}*(z_{n} - H*x_{n,n-1})
        #   x_{n,n} = (I - K_{n}*H)*x_{n,n-1} + (K_{n}*H)*x_{n,n-1} + (K_{n}*H)*v_{n}
        #   x_{n,n} = (I -  alpha )*x_{n,n-1} +   alpha  *x_{n,n-1} +     constant
        vecI = vecZ-np.dot(matH, states) # Innovation.
        newStates = states+np.dot(matK, vecI) # States correction = K_{n}*Innovation.
        if self.sim["prmVrb"] >= 2:
            self.printMat("X", np.transpose(newStates))

        # Update covariance.
        newMatP = self.updateCovariance(matK, matH, matP, matR)

        # Save corrector results.
        self.saveCorrector(newTime, (matK, vecI, vecZ))

        return newStates, newMatP

    def getMsr(self, msrLst):
        """Get measurement"""

        # Verbose on demand.
        if self.sim["prmVrb"] >= 2:
            print("  "*3+"Measurements:")
            for msrItem in msrLst: # Small (accurate) sigma at msrLst tail.
                print("  "*4+msrItem[0]+":", end="")
                print(" %.6f" % msrItem[1], end="")
                print(" %.6f" % msrItem[2], end="")
                print(" %.6f" % msrItem[3], end="")
                print(", sigma %.6f" % msrItem[4], end="")
                print("")

        # Get measurement: z_{n} = H*x_{n} + v_{n}.
        vecZ, matH, matR = self.example.getMsr(msrLst)

        # Verbose on demand.
        if self.sim["prmVrb"] >= 2:
            self.printMat("Z", np.transpose(vecZ))
        if self.sim["prmVrb"] >= 3:
            self.printMat("H", matH)
            self.printMat("R", matR)

        return vecZ, matH, matR

    def computeKalmanGain(self, matP, matH, matR):
        """Compute Kalman gain"""

        # Compute Kalman gain: K_{n} = P_{n,n-1}*Ht*(H*P_{n,n-1}*Ht + R_{n})^-1.
        matK = np.dot(matH, np.dot(matP, np.transpose(matH)))+matR
        if self.sim["prmVrb"] >= 4:
            self.printMat("H*P*Ht+R", matK)
        matK = np.dot(matP, np.dot(np.transpose(matH), npl.inv(matK)))

        # Verbose on demand.
        if self.sim["prmVrb"] >= 3:
            self.printMat("K", matK)

        return matK # https://www.kalmanfilter.net/kalmanGain.html.

    def updateCovariance(self, matK, matH, matP, matR):
        """Update covariance"""

        # Update covariance using Joseph's formula (better numerical stability):
        # P_{n,n} = (I-K_{n}*H)*P_{n,n-1}*(I-K_{n}*H)t + K_{n}*R*K_{n}t.
        nbOfSts = self.example.getNbOfStates()
        matImKH = np.identity(nbOfSts, dtype=kfType)-np.dot(matK, matH)
        if self.sim["prmVrb"] >= 4:
            self.printMat("I-KH", matImKH)
        newMatP = np.dot(matImKH, np.dot(matP, np.transpose(matImKH)))
        newMatP = newMatP+np.dot(matK, np.dot(matR, np.transpose(matK)))

        # Verbose on demand.
        if self.sim["prmVrb"] >= 3:
            self.printMat("P", newMatP)

        return newMatP

    def predictor(self, newTime, timeDt, states, matP):
        """Solve predictor step"""

        # Predict states.
        if self.sim["prmVrb"] >= 1:
            print("  "*2+"Predictor: time %.3f, iteration %d" % (newTime, self.sim["simItNb"]))
        newStates, matF, matQ = self.predictStates(timeDt, newTime, states)

        # Outputs equation: y_{n+1,n+1} = C*x_{n+1,n+1} + D*u_{n+1,n+1}.
        newVecU = self.example.computeControlLaw(newStates, newTime, self.save["predictor"])
        newOutputs = self.computeOutputs(newStates, newVecU)
        if self.sim["prmVrb"] >= 2:
            self.printMat("Y", np.transpose(newOutputs))

        # Save simulation results.
        self.savePredictor(newTime, newStates, newOutputs, matP)

        # Extrapolate uncertainty.
        newMatP = self.predictCovariance(matP, matF, matQ)

        return newStates, newMatP

    def computeOutputs(self, states, vecU):
        """Compute outputs"""

        # Outputs equation: y_{n+1} = C*x_{n} + D*u_{n}.
        outputs = np.dot(self.sim["matC"], states)
        if self.sim["matD"] is not None:
            outputs = outputs+np.dot(self.sim["matD"], vecU)

        return outputs

    def predictStates(self, timeDt, newTime, states):
        """Predict states"""

        # Compute F_{n,n}: see https://www.kalmanfilter.net/modeling.html.
        nbOfSts = self.example.getNbOfStates()
        matF = np.identity(nbOfSts, dtype=kfType)
        if self.sim["prmVrb"] >= 4:
            self.printMat("F order 0", matF)
        taylorExpLTM = 0.
        for idx in range(1, int(self.sim["prmExpOrd"])+1):
            fac = np.math.factorial(idx)
            taylorExp = npl.matrix_power(timeDt*self.sim["matA"], idx)/fac
            taylorExpLTM = np.amax(np.abs(taylorExp))
            matF = matF+taylorExp
            if self.sim["prmVrb"] >= 4:
                self.printMat("F order %d"%idx, matF)
        if self.sim["prmVrb"] >= 3:
            self.printMat("F", matF)
        self.save["predictor"]["simTEM"] = np.append(self.save["predictor"]["simTEM"], taylorExpLTM)

        # Compute G_{n,n}: see https://www.kalmanfilter.net/modeling.html.
        matG = None
        if self.sim["matB"] is not None:
            matG = np.dot(timeDt*matF, self.sim["matB"])
            if self.sim["prmVrb"] >= 3:
                self.printMat("G", matG)

        # Compute process noise w_{n,n}.
        matQ, vecW = self.getProcessNoise(matG)

        # Compute control law u_{n,n}.
        vecU = self.example.computeControlLaw(states, newTime)
        if self.sim["prmVrb"] >= 2:
            self.printMat("U", np.transpose(vecU))

        # Predictor equation: x_{n+1,n} = F*x_{n,n} + G*u_{n,n} + w_{n,n}.
        newStates = np.dot(matF, states)
        if matG is not None:
            newStates = newStates+np.dot(matG, vecU)
        if vecW is not None:
            newStates = newStates+vecW
        if self.sim["prmVrb"] >= 2:
            self.printMat("X", np.transpose(newStates))

        assert newStates.shape == (nbOfSts, 1), "states - bad dimension"
        return newStates, matF, matQ

    def getProcessNoise(self, matG):
        """Get process noise"""

        # Check if process noise if used.
        if matG is None:
            return None, None

        # Compute process noise matrix: Q_{n,n} = G_{n,n}*sigma^2*G_{n,n}t.
        varQ = self.sim["prmProNseSig"]*self.sim["prmProNseSig"]
        matQ = varQ*np.dot(matG, np.transpose(matG)) # https://www.kalmanfilter.net/covextrap.html.
        if self.sim["prmVrb"] >= 3:
            self.printMat("Q", matQ)

        # Get random noise: w_{n,n} must be such that w_{n,n}*w_{n,n}t = Q_{n,n}.
        nbOfSts = self.example.getNbOfStates()
        vecW = np.zeros((nbOfSts, 1), dtype=kfType)
        for idx in range(nbOfSts):
            vecW[idx] = np.sqrt(matQ[idx, idx])

        # Verbose on demand.
        if self.sim["prmVrb"] >= 2:
            self.printMat("W", np.transpose(vecW))

        # Save process noise.
        self.saveProcessNoise(vecW)

        return matQ, vecW

    def predictCovariance(self, matP, matF, matQ):
        """Predict covariance"""

        # Covariance equation: P_{n+1,n} = F_{n,n}*P_{n,n}*F_{n,n}t + Q_{n,n}.
        newMatP = np.dot(matF, np.dot(matP, np.transpose(matF)))
        if matQ is not None:
            newMatP = newMatP+matQ
        if self.sim["prmVrb"] >= 3:
            self.printMat("P", newMatP)

        return newMatP

    @staticmethod
    def removeH5():
        """Remove H5 file"""

        # Remove H5 file.
        fileName = "kalmanFilterModel.h5"
        if os.path.exists(fileName):
            os.remove(fileName)

    def saveH5(self, name, time, data, header):
        """Save data to H5 file"""

        # Initialize file.
        fileName = "kalmanFilterModel.h5"
        mode = "a" if os.path.exists(fileName) else "w"
        fh5 = h5py.File(fileName, mode)
        for hdr in ["T"] + header:
            key = "%s/%s"%(name, hdr)
            if key not in fh5:
                nbTimeIt = int(self.sim["fcdTf"]/self.sim["prmDt"]) + 1
                dsSize = nbTimeIt + len(self.msr)
                fh5.create_dataset(key, data=np.zeros(shape=(dsSize, 1), dtype=kfType),
                                   chunks=True, maxshape=(dsSize,1))
        simIdx = self.sim["simItNb"]

        # Save data.
        key = "%s/%s"%(name, "T")
        fh5[key][simIdx] = time
        lsData = list(itertools.chain(*data))
        for idx, hdr in enumerate(header):
            key = "%s/%s"%(name, hdr)
            fh5[key][simIdx] = lsData[idx]
        fh5.close()

    def savePredictor(self, time, newStates, newOutputs, matP):
        """Save predictor results"""

        # Save time.
        self.time = np.append(self.time, time)

        # Save states and outputs.
        keys = self.example.getStateKeys()
        for idx, key in enumerate(keys):
            self.states[key] = np.append(self.states[key], newStates[idx])
        keys = self.example.getOutputKeys()
        for idx, key in enumerate(keys):
            self.outputs[key] = np.append(self.outputs[key], newOutputs[idx])

        # Save diagonal terms of covariance.
        keys = self.example.getStateKeys()
        for idx, key in enumerate(keys):
            self.save["predictor"]["simDgP"][key].append(matP[idx, idx])

        # Save data in H5 file.
        self.saveH5("X", time, newStates, self.example.getStateKeys())
        self.saveH5("Y", time, newOutputs, self.example.getOutputKeys())

    def saveCorrector(self, time, vecKIZ):
        """Save corrector results"""

        # Save time.
        self.save["corrector"]["simKGn"]["T"].append(time)
        keys = self.example.getOutputKeys()
        for key in keys:
            self.save["corrector"]["simInv"][key]["T"].append(time)

        # Save Kalman gain, innovation, measurement and states.
        matK, vecI, vecZ = vecKIZ[0], vecKIZ[1], vecKIZ[2]
        colK = matK.sum(axis=0) # Sum of each column of matK.
        for idx, key in enumerate(keys):
            self.save["corrector"]["simKGn"][key].append(colK[idx])
            self.save["corrector"]["simInv"][key]["vecI"].append(vecI[idx])
            self.save["corrector"]["simInv"][key]["vecZ"].append(vecZ[idx])

    def saveProcessNoise(self, vecW):
        """Save process noise"""

        # Save process noise.
        keys = self.example.getStateKeys()
        for idx, key in enumerate(keys):
            self.save["predictor"]["simPrN"][key].append(vecW[idx])

    @staticmethod
    def printMat(msg, mat, indent=3, fmt=".6f"):
        """Pretty print matrice"""

        # Pretty print matrice.
        print("  "*indent+msg+":")
        colMax = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
        for row in mat:
            print("  "*(indent+1), end="")
            for idx, val in enumerate(row):
                print(("{:"+str(colMax[idx])+fmt+"}").format(val), end=" ")
            print("")
