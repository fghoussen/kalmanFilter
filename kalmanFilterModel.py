#!/usr/bin/env python3

"""Kalman filter model"""

import numpy as np
import numpy.linalg as npl

class kalmanFilterModel():
    """Kalman filter model"""

    def __init__(self, example):
        """Initialize"""

        # Initialize members.
        self.sim = {}
        self.msr = []
        self.mat = {}
        self.example = example
        self.time = np.array([], dtype=float)
        self.outputs = {}
        self.save = {"predictor": {}, "corrector": {}}
        for key in ["simCLV", "simFrc", "simPrN", "simDgP"]:
            self.save["predictor"][key] = {}
        for key in ["simDgK", "simInv"]:
            self.save["corrector"][key] = {}
        self.clear()

    def clear(self):
        """Clear previous results"""

        # Clear previous measurements.
        self.msr = []

        # Clear previous time.
        self.time = np.array([], dtype=float)

        # Clear previous results.
        self.outputs.clear()
        keys = self.example.getOutputKeys()
        for key in keys:
            self.outputs[key] = np.array([], dtype=float)

        # Clear previous predictor variables.
        self.save["predictor"]["simCLV"]["FoM"] = {}
        self.save["predictor"]["simCLV"]["FoM"]["X"] = []
        self.save["predictor"]["simCLV"]["FoM"]["Y"] = []
        self.save["predictor"]["simCLV"]["FoM"]["Z"] = []
        self.save["predictor"]["simCLV"]["d(FoM)/dt"] = {}
        self.save["predictor"]["simCLV"]["d(FoM)/dt"]["X"] = []
        self.save["predictor"]["simCLV"]["d(FoM)/dt"]["Y"] = []
        self.save["predictor"]["simCLV"]["d(FoM)/dt"]["Z"] = []
        self.save["predictor"]["simCLV"]["roll"] = []
        self.save["predictor"]["simCLV"]["pitch"] = []
        self.save["predictor"]["simCLV"]["yaw"] = []
        self.save["predictor"]["simTEM"] = np.array([])
        for key in ["thrForce", "dpgForce"]:
            self.save["predictor"]["simFrc"][key] = {}
            self.save["predictor"]["simFrc"][key]["X"] = []
            self.save["predictor"]["simFrc"][key]["Y"] = []
            self.save["predictor"]["simFrc"][key]["Z"] = []
        for key in ["simPrN", "simDgP"]:
            for subKey in self.example.getStateKeys():
                self.save["predictor"][key][subKey] = []

        # Clear previous corrector variables.
        for key in self.example.getStateKeys():
            self.save["corrector"]["simDgK"]["T"] = []
            self.save["corrector"]["simDgK"][key] = []
            self.save["corrector"]["simInv"][key] = {}
            self.save["corrector"]["simInv"][key]["T"] = []
            self.save["corrector"]["simInv"][key]["vecI"] = []
            self.save["corrector"]["simInv"][key]["vecZ"] = []
            self.save["corrector"]["simInv"][key]["state"] = []

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

        # Compute default measurement covariance matrix (needed to avoid singular K matrix).
        self.computeDefaultMsrCovariance()

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
        if msrData["msrType"] == "x":
            posX, posY, posZ = msrData["X"], msrData["Y"], msrData["Z"]
        eqnVX, eqnVY, eqnVZ = None, None, None
        if msrData["msrType"] == "v":
            eqnVX, eqnVY, eqnVZ = msrData["VX"], msrData["VY"], msrData["VZ"]
        eqnAX, eqnAY, eqnAZ = None, None, None
        if msrData["msrType"] == "a":
            eqnAX, eqnAY, eqnAZ = msrData["AX"], msrData["AY"], msrData["AZ"]

        # Append measurement.
        for idx, time in enumerate(msrData["T"]):
            if time not in msrDic:
                msrDic[time] = []
            if msrData["msrType"] == "x":
                msrDic[time].append(("x", posX[idx], posY[idx], posZ[idx], prmSigma))
            if msrData["msrType"] == "v":
                msrDic[time].append(("v", eqnVX[idx], eqnVY[idx], eqnVZ[idx], prmSigma))
            if msrData["msrType"] == "a":
                msrDic[time].append(("a", eqnAX[idx], eqnAY[idx], eqnAZ[idx], prmSigma))

    def computeDefaultMsrCovariance(self):
        """Compute default measurement covariance matrix"""

        # Compute default measurement covariance matrix.
        prmN = self.example.getLTISystemSize()
        matR = np.zeros((prmN, prmN), dtype=float)
        matR[0, 0] = np.power(self.sim["icdSigX0"], 2)
        matR[1, 1] = np.power(self.sim["icdSigVX0"], 2)
        matR[2, 2] = np.power(self.sim["icdSigAX0"], 2)
        matR[3, 3] = np.power(self.sim["icdSigY0"], 2)
        matR[4, 4] = np.power(self.sim["icdSigVY0"], 2)
        matR[5, 5] = np.power(self.sim["icdSigAY0"], 2)
        matR[6, 6] = np.power(self.sim["icdSigZ0"], 2)
        matR[7, 7] = np.power(self.sim["icdSigVZ0"], 2)
        matR[8, 8] = np.power(self.sim["icdSigAZ0"], 2)
        self.mat["R"] = matR # Save for later use: restart from it to avoid singular matrix.

    def setLTI(self, matA, matB, matC, matD):
        """Set Linear Time Invariant matrices"""

        # Set matrices.
        self.mat["A"] = matA
        self.mat["B"] = matB
        self.mat["C"] = matC
        self.mat["D"] = matD
        if self.sim["prmVrb"] >= 3:
            print("  "*2+"Linear Time Invariant system:")
            self.printMat("A", self.mat["A"])
            if self.mat["B"] is not None:
                self.printMat("B", self.mat["B"])
            self.printMat("C", self.mat["C"])
            if self.mat["D"] is not None:
                self.printMat("D", self.mat["D"])

    def solve(self):
        """Solve based on Kalman filter"""

        # Don't solve if we have already a solution.
        if self.isSolved():
            return

        # Initialize states.
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
        self.savePredictor(time, outputs, matP)

        # Solve: https://www.kalmanfilter.net/multiSummary.html.
        prmVrb, self.sim["simItNb"] = self.sim["prmVrb"], 0
        prmDt, prmTf = self.sim["prmDt"], self.sim["fcdTf"]
        while time < prmTf:
            # Cut off time.
            if time+prmDt > prmTf:
                prmDt = prmTf-time

            # Set verbose only for some multiple of iterations.
            self.sim["simItNb"] = self.sim["simItNb"] + 1
            self.sim["prmVrb"] = prmVrb if self.sim["simItNb"]%self.sim["prmVrbIt"] == 0 else 0

            # Solve (= corrector + predictor) with Kalman filter.
            newTime, timeDt, states, matP = self.corrector(time, prmDt, matP, states)
            states, matP = self.predictor(newTime, timeDt, states, matP)

            # Increase time.
            time = time+timeDt

        # Verbose.
        self.sim["prmVrb"] = prmVrb
        if self.sim["prmVrb"] >= 1:
            print("  "*2+"End: time %.3f, iteration %d" % (time, self.sim["simItNb"]))

    def corrector(self, time, prmDt, matP, states):
        """Solve corrector step"""

        # Check if no more measurement: nothing to do.
        newTime = time+prmDt
        newStates = states
        newMatP = matP
        nbMsr = len(self.msr)
        if nbMsr == 0:
            return newTime, newTime-time, newStates, newMatP

        # Look for measurement.
        timeMsr = self.msr[nbMsr-1][0]
        if time <= timeMsr <= newTime:
            msrData = self.msr.pop() # Get measurement out of the list.
            newTime, newStates, newMatP = self.computeCorrection(msrData, matP, states)

        return newTime, newTime-time, newStates, newMatP

    def computeCorrection(self, msrData, matP, states):
        """Compute correction"""

        newTime = msrData[0] # Cut off time to measurement time.
        msrLst = msrData[1]
        if self.sim["prmVrb"] >= 1:
            print("  "*2+"Corrector: time %.3f, iteration %d" % (newTime, self.sim["simItNb"]))

        # Get measurement z_{n}.
        vecZ, matH, msrFlags = self.getMsr(msrLst)

        # Compute Kalman gain K_{n}.
        matR, matK = self.computeKalmanGain(msrLst, matP, matH)

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
        self.saveCorrector(newTime, msrFlags, (matK, vecI, vecZ, states))

        return newTime, newStates, newMatP

    def getMsr(self, msrLst):
        """Get measurement"""

        # Get measurement: z_{n} = H*x_{n} + v_{n}.
        prmN = self.example.getLTISystemSize()
        vecZ = np.zeros((prmN, 1), dtype=float)
        matH = np.zeros((prmN, prmN), dtype=float)
        msrFlags = []
        if self.sim["prmVrb"] >= 2:
            print("  "*3+"Measurements:")
        for msrItem in msrLst: # Small (accurate) sigma at msrLst tail.
            # Print out current measurement.
            if self.sim["prmVrb"] >= 2:
                print("  "*4+msrItem[0]+":", end="")
                print(" %.6f" % msrItem[1], end="")
                print(" %.6f" % msrItem[2], end="")
                print(" %.6f" % msrItem[3], end="")
                print(", sigma %.6f" % msrItem[4], end="")
                print("")

            # Recover most accurate measurement: inaccurate sigma (msrLst head) are rewritten.
            self.example.getMsr(msrItem, vecZ, matH, msrFlags)

        # Verbose on demand.
        if self.sim["prmVrb"] >= 2:
            self.printMat("Z", np.transpose(vecZ))
        if self.sim["prmVrb"] >= 3:
            self.printMat("H", matH)

        return vecZ, matH, msrFlags

    def computeKalmanGain(self, msrLst, matP, matH):
        """Compute Kalman gain"""

        # Compute measurement covariance.
        matR = self.computeMsrCovariance(msrLst)

        # Compute Kalman gain: K_{n} = P_{n,n-1}*Ht*(H*P_{n,n-1}*Ht + R_{n})^-1.
        matK = np.dot(matH, np.dot(matP, np.transpose(matH)))+matR
        if self.sim["prmVrb"] >= 4:
            self.printMat("H*P*Ht+R", matK)
        matK = np.dot(matP, np.dot(np.transpose(matH), npl.inv(matK)))

        # Verbose on demand.
        if self.sim["prmVrb"] >= 3:
            self.printMat("K", matK)

        return matR, matK # https://www.kalmanfilter.net/kalmanGain.html.

    def computeMsrCovariance(self, msrLst):
        """Compute measurement covariance"""

        # Get measurement covariance.
        matR = self.mat["R"] # Start from default matrix (needed to avoid singular K matrix).
        for msrItem in msrLst: # Small (accurate) sigma at msrLst tail.
            msrType = msrItem[0]
            prmSigma = msrItem[4]
            if msrType == "x":
                matR[0, 0] = prmSigma*prmSigma
                matR[3, 3] = prmSigma*prmSigma
                matR[6, 6] = prmSigma*prmSigma
            if msrType == "v":
                matR[1, 1] = prmSigma*prmSigma
                matR[4, 4] = prmSigma*prmSigma
                matR[7, 7] = prmSigma*prmSigma
            if msrType == "a":
                matR[2, 2] = prmSigma*prmSigma
                matR[5, 5] = prmSigma*prmSigma
                matR[8, 8] = prmSigma*prmSigma

        # Verbose on demand.
        if self.sim["prmVrb"] >= 3:
            self.printMat("R", matR)

        return matR

    def updateCovariance(self, matK, matH, matP, matR):
        """Update covariance"""

        # Update covariance using Joseph's formula (better numerical stability):
        # P_{n,n} = (I-K_{n}*H)*P_{n,n-1}*(I-K_{n}*H)t + K_{n}*R*K_{n}t.
        prmN = self.example.getLTISystemSize()
        matImKH = np.identity(prmN, dtype=float)-np.dot(matK, matH)
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
        self.savePredictor(newTime, newOutputs, matP)

        # Extrapolate uncertainty.
        newMatP = self.predictCovariance(matP, matF, matQ)

        return newStates, newMatP

    def computeOutputs(self, states, vecU):
        """Compute outputs"""

        # Outputs equation: y_{n+1} = C*x_{n} + D*u_{n}.
        outputs = np.dot(self.mat["C"], states)
        if self.mat["D"] is not None:
            outputs = outputs+np.dot(self.mat["D"], vecU)

        assert outputs.shape == states.shape, "outputs - bad dimension"
        return outputs

    def predictStates(self, timeDt, newTime, states):
        """Predict states"""

        # Compute F_{n,n}.
        prmN = self.example.getLTISystemSize()
        matF = np.identity(prmN, dtype=float)
        taylorExpLTM = 0.
        for idx in range(1, int(self.sim["prmExpOrd"])+1):
            fac = np.math.factorial(idx)
            taylorExp = npl.matrix_power(timeDt*self.mat["A"], idx)/fac
            taylorExpLTM = np.amax(np.abs(taylorExp))
            matF = matF+taylorExp
        if self.sim["prmVrb"] >= 3:
            self.printMat("F", matF)
        self.save["predictor"]["simTEM"] = np.append(self.save["predictor"]["simTEM"], taylorExpLTM)

        # Compute G_{n,n}.
        matG = None
        if self.mat["B"] is not None:
            matG = np.dot(timeDt*matF, self.mat["B"])
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

        assert newStates.shape == (prmN, 1), "states - bad dimension"
        return newStates, matF, matQ

    def getProcessNoise(self, matG):
        """Get process noise"""

        # Check if process noise if used.
        if matG is None:
            return None, None

        # Compute process noise matrix: Q_{n,n} = G_{n,n}*sigma^2*G_{n,n}t.
        varQ = self.sim["prmProNseSig"]*self.sim["prmProNseSig"]
        matQ = matG*varQ*np.transpose(matG) # https://www.kalmanfilter.net/covextrap.html.
        if self.sim["prmVrb"] >= 3:
            self.printMat("Q", matQ)

        # Get random noise: w_{n,n} must be such that w_{n,n}*w_{n,n}t = Q_{n,n}.
        prmN = self.example.getLTISystemSize()
        vecW = np.zeros((prmN, 1), dtype=float)
        for idx in range(prmN):
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

    def savePredictor(self, time, newOutputs, matP):
        """Save predictor results"""

        # Save time.
        self.time = np.append(self.time, time)

        # Save states and outputs.
        keys = self.example.getOutputKeys()
        for idx, key in enumerate(keys):
            self.outputs[key] = np.append(self.outputs[key], newOutputs[idx])

        # Save diagonal terms of covariance.
        keys = self.example.getStateKeys()
        for idx, key in enumerate(keys):
            self.save["predictor"]["simDgP"][key].append(matP[idx, idx])

    def saveCorrector(self, time, msrFlags, vecKIZX):
        """Save corrector results"""

        # Save time.
        self.save["corrector"]["simDgK"]["T"].append(time)
        keys = self.example.getStateKeys()
        for key in keys:
            if key in msrFlags:
                self.save["corrector"]["simInv"][key]["T"].append(time)

        # Save Kalman gain, innovation, measurement and states.
        matK, vecI, vecZ, states = vecKIZX[0], vecKIZX[1], vecKIZX[2], vecKIZX[3]
        for idx, key in enumerate(keys):
            self.save["corrector"]["simDgK"][key].append(matK[idx, idx])
            if key in msrFlags:
                self.save["corrector"]["simInv"][key]["vecI"].append(vecI[idx])
                self.save["corrector"]["simInv"][key]["vecZ"].append(vecZ[idx])
                self.save["corrector"]["simInv"][key]["state"].append(states[idx])

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
