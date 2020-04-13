#!/usr/bin/env python3

"""Kalman filter MVC (Model-View-Controller)"""

import sys
import math
import numpy as np
import numpy.linalg as npl
import scipy.linalg as spl
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from scipy.interpolate import lagrange
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget
from PyQt5.QtWidgets import QLabel, QComboBox, QPushButton, QHBoxLayout, QVBoxLayout
from PyQt5.QtWidgets import QGroupBox, QGridLayout, QLineEdit
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QListWidget, QCheckBox
from PyQt5.QtCore import Qt

matplotlib.use("Qt5Agg")

class mpl2DCanvas(FigureCanvasQTAgg):
    """Matplotlib 2D canvas to be embedded in Qt widget"""

    def __init__(self, parent=None):
        """Initialize"""

        # Initialize.
        fig = plt.figure()
        super(mpl2DCanvas, self).__init__(fig)
        self.setParent(parent)
        self.fig = fig
        self.nrows = 0
        self.ncols = 0
        self.axes = []

    def setUp(self, nrows=1, ncols=1):
        """Set up"""

        # Set up.
        self.nrows = nrows
        self.ncols = ncols
        self.axes = []
        for idx in range(nrows*ncols):
            self.axes.append(self.fig.add_subplot(nrows, ncols, idx+1))

class mpl3DCanvas(FigureCanvasQTAgg):
    """Matplotlib 3D canvas to be embedded in Qt widget"""

    def __init__(self, parent=None):
        """Initialize"""

        # Initialize.
        fig = plt.figure()
        super(mpl3DCanvas, self).__init__(fig)
        self.setParent(parent)
        self.axes = fig.add_subplot(111, projection=Axes3D.name)

class viewer2DGUI(QMainWindow):
    """Kalman filter 2D viewer"""

    def __init__(self, *args, **kwargs):
        """Initialize"""

        # Initialize.
        super(viewer2DGUI, self).__init__(*args, **kwargs)
        self.mcvs = mpl2DCanvas(self)
        self.closed = False
        self.nrows = 0
        self.ncols = 0
        self.toolbar = NavigationToolbar(self.mcvs, self)

        # Set window as non modal.
        self.setWindowModality(Qt.NonModal)

        # Set the layout
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.mcvs)

        # Build the GUI.
        vwrGUI = QWidget(self)
        vwrGUI.setLayout(layout)
        self.setCentralWidget(vwrGUI)

    def setUp(self, nrows=1, ncols=1):
        """Set up"""

        # Set up.
        self.nrows = nrows
        self.ncols = ncols
        self.mcvs.setUp(nrows, ncols)

    def getAxis(self, idx=0):
        """Get viewer axis"""

        # Return viewer axis.
        if idx < 0 or idx >= self.nrows*self.ncols:
            return None
        return self.mcvs.axes[idx]

    def draw(self):
        """Force draw of the scene"""

        # Draw scene.
        self.mcvs.draw()

    def closeEvent(self, event):
        """Callback on closing window"""

        # Mark window as closed.
        self.closed = True
        event.accept()

class viewer3DGUI(QMainWindow):
    """Kalman filter 3D viewer"""

    def __init__(self, *args, **kwargs):
        """Initialize"""

        # Initialize.
        super(viewer3DGUI, self).__init__(*args, **kwargs)
        self.mcvs = mpl3DCanvas(self)
        self.closed = False
        self.toolbar = NavigationToolbar(self.mcvs, self)

        # Set window as non modal.
        self.setWindowModality(Qt.NonModal)

        # Set the layout
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.mcvs)

        # Build the GUI.
        vwrGUI = QWidget(self)
        vwrGUI.setLayout(layout)
        self.setCentralWidget(vwrGUI)

    def getAxis(self):
        """Get viewer axis"""

        # Return viewer axis.
        return self.mcvs.axes

    def draw(self):
        """Force draw of the scene"""

        # Draw scene.
        self.mcvs.draw()

    def closeEvent(self, event):
        """Callback on closing window"""

        # Mark window as closed.
        self.closed = True
        event.accept()

class kalmanFilterModel():
    """Kalman filter model"""

    def __init__(self, example):
        """Initialize"""

        # Initialize members.
        self.sim = {"ctlHV": {}, "matP": {}, "matK": {}}
        self.msr = []
        self.example = example
        self.time = []
        self.states = {}
        self.outputs = {}
        self.mat = {}
        self.clear()

    def clear(self):
        """Clear previous results"""

        # Clear previous measurements.
        self.msr = []

        # Clear previous time.
        self.time = []

        # Clear previous results.
        self.states.clear()
        keys = self.example.getStateKeys()
        for key in keys:
            self.states[key] = []
        self.outputs.clear()
        keys = self.example.getOutputKeys()
        for key in keys:
            self.outputs[key] = []

        # Clear previous control law hidden variables.
        self.sim["ctlHV"]["FoM"] = {}
        self.sim["ctlHV"]["FoM"]["X"] = []
        self.sim["ctlHV"]["FoM"]["Y"] = []
        self.sim["ctlHV"]["FoM"]["Z"] = []
        self.sim["ctlHV"]["d(FoM)/dt"] = {}
        self.sim["ctlHV"]["d(FoM)/dt"]["X"] = []
        self.sim["ctlHV"]["d(FoM)/dt"]["Y"] = []
        self.sim["ctlHV"]["d(FoM)/dt"]["Z"] = []
        self.sim["ctlHV"]["roll"] = []
        self.sim["ctlHV"]["pitch"] = []
        self.sim["ctlHV"]["yaw"] = []
        self.sim["taylorExpLTM"] = np.array([])

        # Clear previous covariance and Kalman gain variables.
        for key in ["matP", "matK"]:
            self.sim[key]["T"] = []
            self.sim[key]["X"] = []
            self.sim[key]["VX"] = []
            self.sim[key]["AX"] = []
            self.sim[key]["Y"] = []
            self.sim[key]["VY"] = []
            self.sim[key]["AY"] = []
            self.sim[key]["Z"] = []
            self.sim[key]["VZ"] = []
            self.sim[key]["AZ"] = []

    def isSolved(self):
        """Check if solve has been done"""

        # Check if solve has been done.
        if len(self.time) == 0:
            return False
        return True

    def setUpSimPrm(self, sim, cdfTf):
        """Setup solver: simulation parameters"""

        # Set up solver parameters.
        for key in sim:
            if key.find("prm") == 0 or key.find("cdi") == 0:
                self.sim[key] = float(sim[key].text())
        self.sim["cdfTf"] = float(cdfTf)

        # Compute default measurement covariance matrix (needed to avoid singular K matrix).
        self.computeDefaultMeasurementCovariance()

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
            prmSig = "sigma "+str(prmSigma)
            if msrData["msrType"] == "x":
                msrDic[time].append(("x", posX[idx], posY[idx], posZ[idx], prmSig))
            if msrData["msrType"] == "v":
                msrDic[time].append(("v", eqnVX[idx], eqnVY[idx], eqnVZ[idx], prmSig))
            if msrData["msrType"] == "a":
                msrDic[time].append(("a", eqnAX[idx], eqnAY[idx], eqnAZ[idx], prmSig))

    def computeDefaultMeasurementCovariance(self):
        """Compute default measurement covariance matrix"""

        # Compute default measurement covariance matrix.
        prmN = self.example.getLTISystemSize()
        matR = np.zeros((prmN, prmN), dtype=float)
        matR[0, 0] = np.power(self.sim["cdiSigX0"], 2)
        matR[1, 1] = np.power(self.sim["cdiSigVX0"], 2)
        matR[2, 2] = np.power(self.sim["cdiSigAX0"], 2)
        matR[3, 3] = np.power(self.sim["cdiSigY0"], 2)
        matR[4, 4] = np.power(self.sim["cdiSigVY0"], 2)
        matR[5, 5] = np.power(self.sim["cdiSigAY0"], 2)
        matR[6, 6] = np.power(self.sim["cdiSigZ0"], 2)
        matR[7, 7] = np.power(self.sim["cdiSigVZ0"], 2)
        matR[8, 8] = np.power(self.sim["cdiSigAZ0"], 2)
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
        time = 0.
        states = self.example.initStates(self.sim)
        matU = self.example.computeControlLaw(states, self.sim)
        outputs = self.computeOutputs(states, matU)
        matP = self.example.initStateCovariance(self.sim)
        if self.sim["prmVrb"] >= 1:
            print("  "*2+"Initialisation:")
        if self.sim["prmVrb"] >= 2:
            self.printMat("Initialisation - X", np.transpose(states))
            self.printMat("Initialisation - Y", np.transpose(outputs))
            self.printMat("Initialisation - P", matP)
        self.saveXY(time, states, outputs)
        self.saveP(time, matP)

        # Solve: https://www.kalmanfilter.net/multiSummary.html.
        prmDt, prmTf = self.sim["prmDt"], self.sim["cdfTf"]
        while time < prmTf:
            # Cut off time.
            if time+prmDt > prmTf:
                prmDt = prmTf-time

            # Solve (= corrector + predictor) with Kalman filter.
            newTime, timeDt, states, matP = self.corrector(time, prmDt, matP, states)
            states, matP = self.predictor(newTime, timeDt, states, matP)

            # Increase time.
            time = time+timeDt

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
            print("  "*2+"Corrector: time %.3f" % newTime)

        # Get measurement z_{n}.
        matZ, matH = self.getMeasurement(msrLst)
        if self.sim["prmVrb"] >= 2:
            self.printMat("Corrector - Z", np.transpose(matZ))
        if self.sim["prmVrb"] >= 3:
            self.printMat("Corrector - H", matH)

        # Compute Kalman gain K_{n}.
        matK = self.computeKalmanGain(msrLst, matP, matH)
        if self.sim["prmVrb"] >= 3:
            self.printMat("Corrector - K", matK)
        self.saveK(newTime, matK)

        # Update estimate with measurement: x_{n,n} = x_{n,n-1} + K_{n}*(z_{n} - H*x_{n,n-1}).
        matI = matZ-np.dot(matH, states) # Innovation.
        newStates = states+np.dot(matK, matI) # States correction = K_{n}*Innovation.
        if self.sim["prmVrb"] >= 2:
            self.printMat("Corrector - X", np.transpose(newStates))

        # Update covariance.
        newMatP = self.updateCovariance(matK, matH, matP)
        if self.sim["prmVrb"] >= 3:
            self.printMat("Corrector - P", newMatP)

        return newTime, newStates, newMatP

    def getMeasurement(self, msrLst):
        """Get measurement"""

        # Get measurement: z_{n} = H*x_{n} + v_{n}.
        prmN = self.example.getLTISystemSize()
        matZ = np.zeros((prmN, 1), dtype=float)
        matH = np.zeros((prmN, prmN), dtype=float)
        for msrItem in msrLst:
            if self.sim["prmVrb"] >= 2:
                print("  "*3+msrItem[0]+":", end="")
                print(" %.6f" % msrItem[1], end="")
                print(" %.6f" % msrItem[2], end="")
                print(" %.6f" % msrItem[3], end="")
                print(" %s" % msrItem[4], end="")
                print("")
            if msrItem[0] == "x":
                matZ[0, 0] = msrItem[1] # X.
                matZ[3, 0] = msrItem[2] # Y.
                matZ[6, 0] = msrItem[3] # Z.
                matH[0, 0] = 1.
                matH[3, 3] = 1.
                matH[6, 6] = 1.
            if msrItem[0] == "v":
                matZ[1, 0] = msrItem[1] # VX.
                matZ[4, 0] = msrItem[2] # VY.
                matZ[7, 0] = msrItem[3] # VZ.
                matH[1, 1] = 1.
                matH[4, 4] = 1.
                matH[7, 7] = 1.
            if msrItem[0] == "a":
                matZ[2, 0] = msrItem[1] # AX.
                matZ[5, 0] = msrItem[2] # AY.
                matZ[8, 0] = msrItem[3] # AZ.
                matH[2, 2] = 1.
                matH[5, 5] = 1.
                matH[8, 8] = 1.

        return matZ, matH

    def computeKalmanGain(self, msrLst, matP, matH):
        """Compute Kalman gain"""

        # Compute measurement covariance.
        matR = self.computeMeasurementCovariance(msrLst)
        if self.sim["prmVrb"] >= 3:
            self.printMat("Corrector - R", matR)

        # Compute Kalman gain: K_{n} = P_{n,n-1}*Ht*(H*P_{n,n-1}*Ht + R_{n})^-1.
        matK = np.dot(matH, np.dot(matP, np.transpose(matH)))+matR
        if self.sim["prmVrb"] >= 4:
            self.printMat("Corrector - H*P*Ht+R", matK)
        matK = np.dot(matP, np.dot(np.transpose(matH), npl.inv(matK)))
        if self.sim["prmVrb"] >= 3:
            self.printMat("Corrector - K", matK)

        return matK # https://www.kalmanfilter.net/kalmanGain.html.

    def computeMeasurementCovariance(self, msrLst):
        """Compute measurement covariance"""

        # Get measurement covariance.
        matR = self.mat["R"] # Start from default matrix (needed to avoid singular K matrix).
        for msrItem in msrLst:
            msrType = msrItem[0]
            prmSigma = float(msrItem[4].split()[1])
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

        return matR

    def updateCovariance(self, matK, matH, matP):
        """Update covariance"""

        # Update covariance: P_{n,n} = (I-K_{n}*H)*P_{n,n-1}.
        _, matL, matU = spl.lu(matK) # K_{n} = L*U (numerically unstable: use LU for stability).
        if self.sim["prmVrb"] >= 4:
            self.printMat("Corrector - L such that K = L*U", matL)
            self.printMat("Corrector - U such that K = L*U", matU)
        newMatP = np.dot(matL, np.dot(matU, matH)) # K_{n}*H = L*U*H.
        prmN = self.example.getLTISystemSize()
        newMatP = np.identity(prmN, dtype=float)-newMatP # I-K_{n}*H.
        newMatP = np.dot(newMatP, matP)

        return newMatP # https://www.kalmanfilter.net/simpCovUpdate.html.

    def predictor(self, newTime, timeDt, states, matP):
        """Solve predictor step"""

        # Predict states.
        if self.sim["prmVrb"] >= 1:
            print("  "*2+"Iteration: time %.3f" % newTime)
        newStates, matF, matG = self.predictStates(timeDt, states)

        # Outputs equation: y_{n+1,n+1} = C*x_{n+1,n+1} + D*u_{n+1,n+1}.
        newMatU = self.example.computeControlLaw(newStates, self.sim)
        newOutputs = self.computeOutputs(newStates, newMatU)
        if self.sim["prmVrb"] >= 2:
            self.printMat("Predictor - Y", np.transpose(newOutputs))

        # Save simulation results.
        self.saveXY(newTime, newStates, newOutputs)
        self.saveP(newTime, matP)

        # Extrapolate uncertainty.
        newMatP = self.predictCovariance(matP, matF, matG)

        return newStates, newMatP

    def predictStates(self, timeDt, states):
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
            msg = "Predictor - F"
            self.printMat(msg, matF)
        self.sim["taylorExpLTM"] = np.append(self.sim["taylorExpLTM"], taylorExpLTM)

        # Compute G_{n,n}.
        matG = None
        if self.mat["B"] is not None:
            matG = np.dot(timeDt*matF, self.mat["B"])
            if self.sim["prmVrb"] >= 3:
                self.printMat("Predictor - G", matG)

        # Compute process noise w_{n,n}.
        matW = self.getProcessNoise(states)
        if self.sim["prmVrb"] >= 2:
            self.printMat("Predictor - W", np.transpose(matW))

        # Compute control law u_{n,n}.
        matU = self.example.computeControlLaw(states, self.sim, save=False)
        if self.sim["prmVrb"] >= 2:
            self.printMat("Predictor - U", np.transpose(matU))

        # Predictor equation: x_{n+1,n} = F*x_{n,n} + G*u_{n,n} + w_{n,n}.
        newStates = np.dot(matF, states)
        if matG is not None:
            newStates = newStates+np.dot(matG, matU)
        newStates = newStates+matW
        if self.sim["prmVrb"] >= 2:
            self.printMat("Predictor - X", np.transpose(newStates))

        return newStates, matF, matG

    def predictCovariance(self, matP, matF, matG):
        """Predict covariance"""

        # Compute process noise matrix: Q_{n,n} = G_{n,n}*sigma^2*G_{n,n}t.
        varQ = self.sim["prmProNseSig"]*self.sim["prmProNseSig"]
        matQ = matG*varQ*np.transpose(matG) # https://www.kalmanfilter.net/covextrap.html.
        if self.sim["prmVrb"] >= 3:
            self.printMat("Predictor - Q", matQ)

        # Covariance equation: P_{n+1,n} = F_{n,n}*P_{n,n}*F_{n,n}t + Q_{n,n}.
        newMatP = np.dot(matF, np.dot(matP, np.transpose(matF)))+matQ
        if self.sim["prmVrb"] >= 3:
            self.printMat("Predictor - P", newMatP)

        return newMatP

    def saveXY(self, time, newStates, newOutputs):
        """Save simulation states and outputs"""

        # Save time.
        self.time.append(time)

        # Save states and outputs.
        keys = self.example.getStateKeys()
        for idx, key in enumerate(keys):
            self.states[key].append(newStates[idx, 0])
        keys = self.example.getOutputKeys()
        for idx, key in enumerate(keys):
            self.outputs[key].append(newOutputs[idx, 0])

    def saveP(self, time, matP):
        """Save covariance"""

        # Save covariance.
        self.sim["matP"]["T"].append(time)
        keys = self.example.getStateKeys()
        for idx, key in enumerate(keys):
            self.sim["matP"][key].append(matP[idx, idx])

    def saveK(self, time, matK):
        """Save Kalman gain"""

        # Save Kalman gain.
        self.sim["matK"]["T"].append(time)
        keys = self.example.getStateKeys()
        for idx, key in enumerate(keys):
            self.sim["matK"][key].append(matK[idx, idx])

    def getProcessNoise(self, states):
        """Get process noise"""

        # Get random noise.
        prmMu, prmSigma = states, self.sim["prmProNseSig"]
        noisyStates = np.random.normal(prmMu, prmSigma)
        matW = noisyStates-states

        return matW

    def computeOutputs(self, states, matU):
        """Compute outputs"""

        # Outputs equation: y_{n+1} = C*x_{n} + D*u_{n}.
        outputs = np.dot(self.mat["C"], states)
        if self.mat["D"] is not None:
            outputs = outputs+np.dot(self.mat["D"], matU)

        return outputs

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

class planeTrackingExample:
    """Plane tracking example"""

    def __init__(self, ctrGUI):
        """Initialize"""

        # Initialize members.
        self.ctrGUI = ctrGUI
        self.slt = {"sltId": ""}
        self.msr = {"sltId": "", "msrId": ""}
        self.sim = {"sltId": "", "msrId": "", "simId": ""}
        self.vwr = {"2D": {}, "3D": None}
        self.vwr["2D"]["tzp"] = None
        self.vwr["2D"]["ctlHV"] = None
        self.vwr["2D"]["simOV"] = None
        self.vwr["2D"]["timSc"] = None
        self.vwr["2D"]["matP"] = None
        self.vwr["2D"]["matK"] = None
        self.kfm = kalmanFilterModel(self)

    @staticmethod
    def getName():
        """Return example name"""

        # Return name.
        return "plane tracking"

    def createViewer(self):
        """Create viewer"""

        # Create viewer.
        if not self.vwr["3D"] or self.vwr["3D"].closed:
            self.vwr["3D"] = viewer3DGUI(self.ctrGUI)
        self.vwr["3D"].setWindowTitle("Kalman filter viewer")
        self.vwr["3D"].show()

        return self.vwr["3D"]

    def updateViewer(self):
        """Update viewer"""

        # Update viewer.
        self.clearViewer()
        self.updateViewerSlt()
        self.updateViewerMsr()
        self.updateViewerSim()

        # 3D viewer: order and show legend.
        axis = self.vwr["3D"].getAxis()
        handles, labels = axis.get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        axis.legend(handles, labels)

        # Force viewer redraw.
        self.vwr["3D"].draw()

    def clearViewer(self, vwrId="all"):
        """Clear viewer"""

        # Clear the viewer.
        if vwrId in ("all", "tzp"):
            if self.vwr["2D"]["tzp"]:
                axis = self.vwr["2D"]["tzp"].getAxis()
                axis.cla()
                axis.set_xlabel("t")
                axis.set_ylabel("z")
                self.vwr["2D"]["tzp"].draw()
        for key in ["simOV", "ctlHV", "matP", "matK"]:
            if vwrId in ("all", key):
                if self.vwr["2D"][key]:
                    for idx in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
                        axis = self.vwr["2D"][key].getAxis(idx)
                        axis.set_xlabel("t")
                        axis.cla()
                    self.vwr["2D"][key].draw()
        if vwrId in ("all", "timSc"):
            if self.vwr["2D"]["timSc"]:
                for idx in [0, 1]:
                    axis = self.vwr["2D"]["timSc"].getAxis(idx)
                    axis.set_xlabel("t")
                    axis.cla()
                self.vwr["2D"]["timSc"].draw()
        if vwrId in ("all", "3D"):
            axis = self.vwr["3D"].getAxis()
            axis.cla()
            axis.set_xlabel("x")
            axis.set_ylabel("y")
            axis.set_zlabel("z")
            self.vwr["3D"].draw()

    def updateViewerSlt(self):
        """Update viewer: solution"""

        # Plot only if checked.
        if not self.vwr["ckbSlt"].isChecked():
            return
        print("Update analytic solution")

        # Compute solution if needed.
        okSlt = 1 if self.slt["sltId"] == self.getSltId() else 0
        if not okSlt:
            self.computeSlt()

        # Plot solution.
        print("  "*1+"Plot analytic solution")
        self.updateViewerSltX()
        self.updateViewerSltV()
        self.updateViewerSltA()
        if self.vwr["2D"]["tzp"] and not self.vwr["2D"]["tzp"].closed:
            self.onPltTZPBtnClick()

        # Track solution features.
        self.slt["sltId"] = self.getSltId()

    def computeSlt(self):
        """Compute solution"""

        # Time.
        prmTf = float(self.slt["cdfTf"].text())
        vwrNbPt = float(self.slt["vwrNbPt"].text())
        eqnT = np.linspace(0., prmTf, vwrNbPt)

        # Compute solution.
        print("  "*1+"Compute analytic solution")
        eqnX, eqnY, eqnZ = self.getDisplEquations(eqnT)
        eqnVX, eqnVY, eqnVZ = self.getVelocEquations(eqnT)
        eqnAX, eqnAY, eqnAZ = self.getAccelEquations(eqnT)

        # Update V0/A0 indicators.
        self.slt["indVX0"].setText("%.3f" % eqnVX[0])
        self.slt["indVY0"].setText("%.3f" % eqnVY[0])
        self.slt["indVZ0"].setText("%.3f" % eqnVZ[0])
        self.slt["indAX0"].setText("%.3f" % eqnAX[0])
        self.slt["indAY0"].setText("%.3f" % eqnAY[0])
        self.slt["indAZ0"].setText("%.3f" % eqnAZ[0])

        # Save analytic solution.
        self.slt["T"] = eqnT
        self.slt["X"] = eqnX
        self.slt["Y"] = eqnY
        self.slt["Z"] = eqnZ
        self.slt["VX"] = eqnVX
        self.slt["VY"] = eqnVY
        self.slt["VZ"] = eqnVZ
        self.slt["AX"] = eqnAX
        self.slt["AY"] = eqnAY
        self.slt["AZ"] = eqnAZ

    def updateViewerSltX(self):
        """Update viewer: plot displacement of the solution"""

        # Plot solution: displacement.
        eqnX, eqnY, eqnZ = self.slt["X"], self.slt["Y"], self.slt["Z"]
        vwrLnWd = float(self.slt["vwrLnWd"].text())
        if vwrLnWd == 0.:
            return
        vwrPosMks = float(self.slt["vwrPosMks"].text())
        clr = (0., 0., 1.) # Blue.
        axis = self.vwr["3D"].getAxis()
        axis.plot3D(eqnX, eqnY, eqnZ, lw=vwrLnWd, color=clr,
                    label="flight path: x", marker="o", ms=vwrPosMks)

    def updateViewerSltV(self):
        """Update viewer: plot velocity of the solution"""

        # Plot solution: velocity.
        eqnX, eqnY, eqnZ = self.slt["X"], self.slt["Y"], self.slt["Z"]
        eqnVX, eqnVY, eqnVZ = self.slt["VX"], self.slt["VY"], self.slt["VZ"]
        clr = (0., 0.75, 1.) # Skyblue.
        vwrVelLgh = float(self.slt["vwrVelLgh"].text())
        if vwrVelLgh == 0.:
            return
        vwrVelNrm = self.slt["vwrVelNrm"].isChecked()
        axis = self.vwr["3D"].getAxis()
        axis.quiver3D(eqnX, eqnY, eqnZ, eqnVX, eqnVY, eqnVZ, color=clr,
                      length=vwrVelLgh, normalize=vwrVelNrm, label="flight path: v")

    def updateViewerSltA(self):
        """Update viewer: plot acceleration of the solution"""

        # Plot solution: acceleration.
        eqnX, eqnY, eqnZ = self.slt["X"], self.slt["Y"], self.slt["Z"]
        eqnAX, eqnAY, eqnAZ = self.slt["AX"], self.slt["AY"], self.slt["AZ"]
        clr = (0.25, 0., 0.5) # Indigo.
        vwrAccLgh = float(self.slt["vwrAccLgh"].text())
        if vwrAccLgh == 0.:
            return
        vwrAccNrm = self.slt["vwrAccNrm"].isChecked()
        axis = self.vwr["3D"].getAxis()
        axis.quiver3D(eqnX, eqnY, eqnZ, eqnAX, eqnAY, eqnAZ, colors=clr,
                      length=vwrAccLgh, normalize=vwrAccNrm, label="flight path: a")

    def getSltId(self):
        """Get solution identity (track solution features)"""

        # Get solution identity.
        sltId = ""
        for key in self.slt:
            fpeFound = 1 if key.find("fpe") == 0 else 0
            cdiFound = 1 if key.find("cdi") == 0 else 0
            cdfFound = 1 if key.find("cdf") == 0 else 0
            if fpeFound or cdiFound or cdfFound:
                sltId += ":"+self.slt[key].text()

        return sltId

    def updateViewerMsr(self):
        """Update viewer: measurements"""

        # Plot only if checked.
        if not self.vwr["ckbMsr"].isChecked():
            return
        print("Update measurements")

        # Compute measurements if needed.
        okSlt = 1 if self.msr["sltId"] == self.slt["sltId"] else 0
        okMsr = 1 if self.msr["msrId"] == self.getMsrId() else 0
        if not okSlt or not okMsr:
            if not okSlt:
                self.msr["datMsr"].clear()
            self.computeMsr()
        self.msr["sltId"] = self.slt["sltId"]

        # Plot measurements.
        print("  "*1+"Plot measurements")
        for txt in self.msr["datMsr"]:
            msrData = self.msr["datMsr"][txt]
            self.viewMsrData(msrData)

        # Track measurement features.
        self.msr["msrId"] = self.getMsrId()

    def computeMsr(self):
        """Compute measurements"""

        # Compute measurements.
        print("  "*1+"Compute measurements from analytic solution")
        for idx in range(self.msr["lstMsr"].count()):
            # Skip unused items.
            txt = self.msr["lstMsr"].item(idx).text()
            if txt == "":
                continue

            # Create measure data if needed.
            if txt not in self.msr["datMsr"]:
                print("  "*2+"Measurement: "+txt)
                self.msr["datMsr"][txt] = self.getMsrData(txt)

    def getMsrData(self, txt):
        """Get measure data"""

        # Get data measurements.
        msrType = txt.split(";")[0].split()[1]
        msrData = {"msrType": msrType}
        if msrType == "x":
            self.getMsrDataX(txt, msrData)
        if msrType == "v":
            self.getMsrDataV(txt, msrData)
        if msrType == "a":
            self.getMsrDataA(txt, msrData)

        return msrData

    def getMsrDataX(self, txt, msrData):
        """Get measure data: displacement"""

        # Time.
        prmT0 = float(txt.split(";")[1].split()[1])
        prmTf = float(txt.split(";")[2].split()[1])
        prmDt = float(txt.split(";")[3].split()[1])
        prmNbPt = (prmTf-prmT0)/prmDt
        eqnT = np.linspace(prmT0, prmTf, prmNbPt)
        msrData["T"] = eqnT

        # Data.
        eqnX, eqnY, eqnZ = self.getDisplEquations(eqnT)
        prmSigma = float(txt.split(";")[4].split()[1])
        msrData["X"] = self.addNoise(eqnX, prmSigma)
        msrData["Y"] = self.addNoise(eqnY, prmSigma)
        msrData["Z"] = self.addNoise(eqnZ, prmSigma)

    def getMsrDataV(self, txt, msrData):
        """Get measure data: velocity"""

        # Time.
        prmT0 = float(txt.split(";")[1].split()[1])
        prmTf = float(txt.split(";")[2].split()[1])
        prmDt = float(txt.split(";")[3].split()[1])
        prmNbPt = (prmTf-prmT0)/prmDt
        eqnT = np.linspace(prmT0, prmTf, prmNbPt)
        msrData["T"] = eqnT

        # Data.
        eqnX, eqnY, eqnZ = self.getDisplEquations(eqnT)
        msrData["X"] = eqnX
        msrData["Y"] = eqnY
        msrData["Z"] = eqnZ
        eqnVX, eqnVY, eqnVZ = self.getVelocEquations(eqnT)
        prmSigma = float(txt.split(";")[4].split()[1])
        msrData["VX"] = self.addNoise(eqnVX, prmSigma)
        msrData["VY"] = self.addNoise(eqnVY, prmSigma)
        msrData["VZ"] = self.addNoise(eqnVZ, prmSigma)

    def getMsrDataA(self, txt, msrData):
        """Get measure data: acceleration"""

        # Time.
        prmT0 = float(txt.split(";")[1].split()[1])
        prmTf = float(txt.split(";")[2].split()[1])
        prmDt = float(txt.split(";")[3].split()[1])
        prmNbPt = (prmTf-prmT0)/prmDt
        eqnT = np.linspace(prmT0, prmTf, prmNbPt)
        msrData["T"] = eqnT

        # Data.
        eqnX, eqnY, eqnZ = self.getDisplEquations(eqnT)
        msrData["X"] = eqnX
        msrData["Y"] = eqnY
        msrData["Z"] = eqnZ
        eqnAX, eqnAY, eqnAZ = self.getAccelEquations(eqnT)
        prmSigma = float(txt.split(";")[4].split()[1])
        msrData["AX"] = self.addNoise(eqnAX, prmSigma)
        msrData["AY"] = self.addNoise(eqnAY, prmSigma)
        msrData["AZ"] = self.addNoise(eqnAZ, prmSigma)

    @staticmethod
    def addNoise(eqn, prmSigma):
        """Add (gaussian) noise"""

        # Add noise to data: v_{n} such that z_{n} = H*x_{n} + v_{n}.
        prmMu = eqn
        noisyEqn = np.random.normal(prmMu, prmSigma)

        return noisyEqn

    def getDisplEquations(self, eqnT):
        """Get displacement equations"""

        # Get displacement equations.
        eqnX = self.getXEquation(eqnT)
        eqnY = self.getYEquation(eqnT)
        eqnZ = self.getZEquation(eqnT)

        return eqnX, eqnY, eqnZ

    def getVelocEquations(self, eqnT):
        """Get velocity equations"""

        # Get velocity equations.
        eqnVX = self.getVXEquation(eqnT)
        eqnVY = self.getVYEquation(eqnT)
        eqnVZ = self.getVZEquation(eqnT)

        return eqnVX, eqnVY, eqnVZ

    def getAccelEquations(self, eqnT):
        """Get acceleration equations"""

        # Get acceleration equations.
        eqnAX = self.getAXEquation(eqnT)
        eqnAY = self.getAYEquation(eqnT)
        eqnAZ = self.getAZEquation(eqnT)

        return eqnAX, eqnAY, eqnAZ

    def getXEquation(self, eqnT):
        """Get X equation: displacement"""

        # Get X equation: displacement.
        prmV0 = float(self.slt["cdiX0"].text())
        prmA = float(self.slt["fpeAx"].text())
        prmPhi = float(self.slt["fpePhix"].text())
        prmT = float(self.slt["fpeTx"].text())
        prmB = prmV0-prmA*np.cos(prmPhi)
        omega = 2.*math.pi/prmT
        eqnX = prmA*np.cos(omega*eqnT+prmPhi)+prmB

        return eqnX

    def getVXEquation(self, eqnT):
        """Get X equation: velocity"""

        # Get X equation: velocity.
        prmA = float(self.slt["fpeAx"].text())
        prmPhi = float(self.slt["fpePhix"].text())
        prmT = float(self.slt["fpeTx"].text())
        omega = 2.*math.pi/prmT
        eqnVX = -1.*prmA*omega*np.sin(omega*eqnT+prmPhi)

        return eqnVX

    def getAXEquation(self, eqnT):
        """Get X equation: acceleration"""

        # Get X equation: acceleration.
        prmA = float(self.slt["fpeAx"].text())
        prmPhi = float(self.slt["fpePhix"].text())
        prmT = float(self.slt["fpeTx"].text())
        omega = 2.*math.pi/prmT
        eqnAX = -1.*prmA*omega*omega*np.cos(omega*eqnT+prmPhi)

        return eqnAX

    def getYEquation(self, eqnT):
        """Get Y equation: displacement"""

        # Get Y equation: displacement.
        prmV0 = float(self.slt["cdiY0"].text())
        prmA = float(self.slt["fpeAy"].text())
        prmPhi = float(self.slt["fpePhiy"].text())
        prmT = float(self.slt["fpeTy"].text())
        prmB = prmV0-prmA*np.sin(prmPhi)
        omega = 2.*math.pi/prmT
        eqnY = prmA*np.sin(omega*eqnT+prmPhi)+prmB

        return eqnY

    def getVYEquation(self, eqnT):
        """Get Y equation: velocity"""

        # Get Y equation: velocity.
        prmA = float(self.slt["fpeAy"].text())
        prmPhi = float(self.slt["fpePhiy"].text())
        prmT = float(self.slt["fpeTy"].text())
        omega = 2.*math.pi/prmT
        eqnVY = prmA*omega*np.cos(omega*eqnT+prmPhi)

        return eqnVY

    def getAYEquation(self, eqnT):
        """Get Y equation: acceleration"""

        # Get Y equation: acceleration.
        prmA = float(self.slt["fpeAy"].text())
        prmPhi = float(self.slt["fpePhiy"].text())
        prmT = float(self.slt["fpeTy"].text())
        omega = 2.*math.pi/prmT
        eqnAY = -1.*prmA*omega*omega*np.sin(omega*eqnT+prmPhi)

        return eqnAY

    def getZEquation(self, eqnT):
        """Get Z equation"""

        # Get Z equation: displacement.
        poly = self.getZPoly()
        eqnZ = poly(eqnT)

        return eqnZ

    def getVZEquation(self, eqnT):
        """Get Z equation: velocity"""

        # Get Z equation: velocity.
        poly = self.getZPoly()
        dpoly = np.polyder(poly, m=1)
        eqnVZ = dpoly(eqnT)

        return eqnVZ

    def getAZEquation(self, eqnT):
        """Get Z equation: acceleration"""

        # Get Z equation: acceleration.
        poly = self.getZPoly()
        dpoly = np.polyder(poly, m=2)
        eqnAZ = dpoly(eqnT)

        return eqnAZ

    def getZPoly(self):
        """Get Z polynomial"""

        # Get polynomial.
        prmTi, prmZi = self.getZPolyPts()
        poly = lagrange(prmTi, prmZi)

        return poly

    def getZPolyPts(self):
        """Get Z polynomial lagrange points"""

        # Get polynomial points.
        prmZ0 = float(self.slt["cdiZ0"].text())
        prmTiZi = self.slt["fpeTiZi"].text()
        prmTi = np.array([0.], dtype=float)
        prmZi = np.array([prmZ0], dtype=float)
        for tokTiZi in prmTiZi.split(","):
            tokTi, tokZi = tokTiZi.split()
            prmTi = np.append(prmTi, float(tokTi))
            prmZi = np.append(prmZi, float(tokZi))

        return prmTi, prmZi

    def viewMsrData(self, msrData):
        """View measure data"""

        # View data measurements.
        if msrData["msrType"] == "x":
            self.viewMsrDataX(msrData)
        if msrData["msrType"] == "v":
            self.viewMsrDataV(msrData)
        if msrData["msrType"] == "a":
            self.viewMsrDataA(msrData)

    def viewMsrDataX(self, msrData):
        """View measure data: displacement"""

        # View measure data: displacement.
        posX = msrData["X"]
        posY = msrData["Y"]
        posZ = msrData["Z"]
        vwrPosMks = float(self.msr["vwrPosMks"].text())
        if vwrPosMks == 0.:
            return
        axis = self.vwr["3D"].getAxis()
        axis.scatter3D(posX, posY, posZ, c="r", marker="^", alpha=1, s=vwrPosMks,
                       label="measure: x")

    def viewMsrDataV(self, msrData):
        """View measure data: velocity"""

        # View measure data: velocity.
        posX = msrData["X"]
        posY = msrData["Y"]
        posZ = msrData["Z"]
        eqnVX = msrData["VX"]
        eqnVY = msrData["VY"]
        eqnVZ = msrData["VZ"]
        clr = (1., 0.65, 0.) # Orange.
        vwrVelLgh = float(self.msr["vwrVelLgh"].text())
        if vwrVelLgh == 0.:
            return
        vwrVelNrm = self.msr["vwrVelNrm"].isChecked()
        axis = self.vwr["3D"].getAxis()
        axis.quiver3D(posX, posY, posZ, eqnVX, eqnVY, eqnVZ,
                      colors=clr, length=vwrVelLgh, normalize=vwrVelNrm,
                      label="measure: v")

    def viewMsrDataA(self, msrData):
        """View measure data: acceleration"""

        # View measure data: acceleration.
        posX = msrData["X"]
        posY = msrData["Y"]
        posZ = msrData["Z"]
        eqnAX = msrData["AX"]
        eqnAY = msrData["AY"]
        eqnAZ = msrData["AZ"]
        clr = (0.6, 0.3, 0.) # Brown.
        vwrAccLgh = float(self.msr["vwrAccLgh"].text())
        if vwrAccLgh == 0.:
            return
        vwrAccNrm = self.msr["vwrAccNrm"].isChecked()
        axis = self.vwr["3D"].getAxis()
        axis.quiver3D(posX, posY, posZ, eqnAX, eqnAY, eqnAZ,
                      colors=clr, length=vwrAccLgh, normalize=vwrAccNrm,
                      label="measure: a")

    def getMsrId(self):
        """Get measurements identity (track measurement features)"""

        # Get measurements identity.
        msrId = ""
        for idx in range(self.msr["lstMsr"].count()):
            txt = self.msr["lstMsr"].item(idx).text()
            if txt == "":
                continue
            msrId += ":"+txt

        return msrId

    def updateViewerSim(self):
        """Update viewer: simulation"""

        # Plot only if checked.
        if not self.vwr["ckbSim"].isChecked():
            return
        print("Update simulation")

        # Clean solver results if analytic solution or measurements have changed.
        okSlt = 1 if self.sim["sltId"] == self.slt["sltId"] else 0
        okMsr = 1 if self.sim["msrId"] == self.msr["msrId"] else 0
        okSim = 1 if self.sim["simId"] == self.getSimId() else 0
        if not okSlt or not okMsr or not okSim:
            self.computeSim()
        self.sim["sltId"] = self.slt["sltId"]
        self.sim["msrId"] = self.msr["msrId"]

        # Plot solver results.
        print("  "*1+"Plot simulation results")
        self.updateViewerSimX()
        self.updateViewerSimV()
        self.updateViewerSimA()
        if self.vwr["2D"]["simOV"] and not self.vwr["2D"]["simOV"].closed:
            self.onPltSOVBtnClick()
        if self.vwr["2D"]["ctlHV"] and not self.vwr["2D"]["ctlHV"].closed:
            self.onPltCHVBtnClick()
        if self.vwr["2D"]["timSc"] and not self.vwr["2D"]["timSc"].closed:
            self.onPltTScBtnClick()
        if self.vwr["2D"]["matP"] and not self.vwr["2D"]["matP"].closed:
            self.onPltCovBtnClick()
        if self.vwr["2D"]["matK"] and not self.vwr["2D"]["matK"].closed:
            self.onPltSKGBtnClick()

        # Track simulation features.
        self.sim["simId"] = self.getSimId()

    def computeSim(self):
        """Compute simulation"""

        # Solve based on Kalman filter.
        print("  "*1+"Run simulation based on Kalman filter")
        self.kfm.clear()
        self.kfm.setUpSimPrm(self.sim, self.slt["cdfTf"].text())
        self.kfm.setUpMsrPrm(self.msr["datMsr"])
        matA, matB, matC, matD = self.getLTISystem()
        self.kfm.setLTI(matA, matB, matC, matD)
        self.kfm.solve()

    def getSimId(self):
        """Get simulation identity (track simulation features)"""

        # Get simulation identity.
        simId = ""
        for key in self.sim:
            if key == "prmVrb":
                continue
            prmFound = 1 if key.find("prm") == 0 else 0
            cdiFound = 1 if key.find("cdi") == 0 else 0
            ctlFound = 1 if key.find("ctl") == 0 else 0
            if prmFound or cdiFound or ctlFound:
                simId += ":"+self.sim[key].text()

        return simId

    def updateViewerSimX(self):
        """Update viewer: plot displacement of the simulation"""

        # Plot simulation: displacement.
        eqnX, eqnY, eqnZ = self.kfm.outputs["X"], self.kfm.outputs["Y"], self.kfm.outputs["Z"]
        vwrLnWd = float(self.sim["vwrLnWd"].text())
        if vwrLnWd == 0.:
            return
        vwrPosMks = float(self.sim["vwrPosMks"].text())
        clr = (0., 0.5, 0.) # Green.
        axis = self.vwr["3D"].getAxis()
        axis.plot3D(eqnX, eqnY, eqnZ, lw=vwrLnWd, color=clr,
                    label="simulation: x", marker="o", ms=vwrPosMks)

    def updateViewerSimV(self):
        """Update viewer: plot velocity of the simulation"""

        # Plot simulation: velocity.
        eqnX, eqnY, eqnZ = self.kfm.outputs["X"], self.kfm.outputs["Y"], self.kfm.outputs["Z"]
        eqnVX, eqnVY, eqnVZ = self.kfm.outputs["VX"], self.kfm.outputs["VY"], self.kfm.outputs["VZ"]
        clr = (0., 1., 0.) # Lime green.
        vwrVelLgh = float(self.sim["vwrVelLgh"].text())
        if vwrVelLgh == 0.:
            return
        vwrVelNrm = self.sim["vwrVelNrm"].isChecked()
        axis = self.vwr["3D"].getAxis()
        axis.quiver3D(eqnX, eqnY, eqnZ, eqnVX, eqnVY, eqnVZ, color=clr,
                      length=vwrVelLgh, normalize=vwrVelNrm, label="simulation: v")

    def updateViewerSimA(self):
        """Update viewer: plot acceleration of the simulation"""

        # Plot simulation: acceleration.
        eqnX, eqnY, eqnZ = self.kfm.outputs["X"], self.kfm.outputs["Y"], self.kfm.outputs["Z"]
        eqnAX, eqnAY, eqnAZ = self.kfm.outputs["AX"], self.kfm.outputs["AY"], self.kfm.outputs["AZ"]
        clr = (0., 0.2, 0.) # Dark green.
        vwrAccLgh = float(self.sim["vwrAccLgh"].text())
        if vwrAccLgh == 0.:
            return
        vwrAccNrm = self.sim["vwrAccNrm"].isChecked()
        axis = self.vwr["3D"].getAxis()
        axis.quiver3D(eqnX, eqnY, eqnZ, eqnAX, eqnAY, eqnAZ, colors=clr,
                      length=vwrAccLgh, normalize=vwrAccNrm, label="simulation: a")

    def throwError(self, eId, txt):
        """Throw an error message"""

        # Create error message box.
        msg = QMessageBox(self.ctrGUI)
        msg.setIcon(QMessageBox.Critical)
        msg.setText("Error")
        msg.setText("Error"+" - "+eId+": "+txt)
        msg.exec_()

    def createSltGUI(self):
        """Create solution GUI"""

        # Create group box.
        sltGUI = QGroupBox(self.ctrGUI)
        sltGUI.setTitle("Analytic solution: targeting real flight path")
        sltGUI.setAlignment(Qt.AlignHCenter)

        # Store analytic parameters.
        self.slt["fpeAx"] = QLineEdit("1.", self.ctrGUI)
        self.slt["fpeAy"] = QLineEdit("2.", self.ctrGUI)
        self.slt["fpeTx"] = QLineEdit("1.", self.ctrGUI)
        self.slt["fpeTy"] = QLineEdit("1.", self.ctrGUI)
        self.slt["fpePhix"] = QLineEdit("0.", self.ctrGUI)
        self.slt["fpePhiy"] = QLineEdit("0.", self.ctrGUI)
        self.slt["fpeTiZi"] = QLineEdit("0.1 0.1, 0.5 1.2, 1.5 0.8, 1.9 0.1, 2. 0.", self.ctrGUI)
        self.slt["cdiX0"] = QLineEdit("0.", self.ctrGUI)
        self.slt["cdiY0"] = QLineEdit("0.", self.ctrGUI)
        self.slt["cdiZ0"] = QLineEdit("0.", self.ctrGUI)
        self.slt["indVX0"] = QLineEdit("N.A.", self.ctrGUI)
        self.slt["indVY0"] = QLineEdit("N.A.", self.ctrGUI)
        self.slt["indVZ0"] = QLineEdit("N.A.", self.ctrGUI)
        self.slt["indAX0"] = QLineEdit("N.A.", self.ctrGUI)
        self.slt["indAY0"] = QLineEdit("N.A.", self.ctrGUI)
        self.slt["indAZ0"] = QLineEdit("N.A.", self.ctrGUI)
        self.slt["cdfTf"] = QLineEdit("2.", self.ctrGUI)
        self.slt["vwrNbPt"] = QLineEdit("50", self.ctrGUI)
        self.slt["vwrLnWd"] = QLineEdit("1.", self.ctrGUI)
        self.slt["vwrPosMks"] = QLineEdit("5", self.ctrGUI)
        self.slt["vwrVelLgh"] = QLineEdit("0.01", self.ctrGUI)
        self.slt["vwrVelNrm"] = QCheckBox("Normalize", self.ctrGUI)
        self.slt["vwrAccLgh"] = QLineEdit("0.001", self.ctrGUI)
        self.slt["vwrAccNrm"] = QCheckBox("Normalize", self.ctrGUI)

        self.slt["indVX0"].setEnabled(False)
        self.slt["indVY0"].setEnabled(False)
        self.slt["indVZ0"].setEnabled(False)
        self.slt["indAX0"].setEnabled(False)
        self.slt["indAY0"].setEnabled(False)
        self.slt["indAZ0"].setEnabled(False)

        # Fill solution GUI.
        self.fillSltGUI(sltGUI)

        return sltGUI

    def fillSltGUI(self, sltGUI):
        """Fill solution GUI"""

        # Create group box.
        gpbXi = self.fillSltGUIXi(sltGUI)
        gpbX0 = self.fillSltGUIX0(sltGUI)
        gpbTf = self.fillSltGUITf(sltGUI)
        gpbVwr = self.fillSltGUIVwr(sltGUI)

        # Set group box layout.
        sltLay = QHBoxLayout(sltGUI)
        sltLay.addWidget(gpbXi)
        sltLay.addWidget(gpbX0)
        sltLay.addWidget(gpbTf)
        sltLay.addWidget(gpbVwr)
        sltGUI.setLayout(sltLay)

    def fillSltGUIXi(self, sltGUI):
        """Fill solution GUI : flight path equation"""

        # Create analytic parameters GUI: flight path equation.
        gdlXi = QGridLayout(sltGUI)
        title = "x(t) = a<sub>x</sub>cos(2&pi;/T<sub>x</sub>*t+&phi;<sub>x</sub>)"
        gdlXi.addWidget(QLabel(title, sltGUI), 0, 0, 1, 6)
        gdlXi.addWidget(QLabel("a<sub>x</sub>", sltGUI), 1, 0)
        gdlXi.addWidget(self.slt["fpeAx"], 1, 1)
        gdlXi.addWidget(QLabel("T<sub>x</sub>", sltGUI), 1, 2)
        gdlXi.addWidget(self.slt["fpeTx"], 1, 3)
        gdlXi.addWidget(QLabel("&phi;<sub>x</sub>", sltGUI), 1, 4)
        gdlXi.addWidget(self.slt["fpePhix"], 1, 5)
        title = "y(t) = a<sub>y</sub>sin(2&pi;/T<sub>y</sub>*t+&phi;<sub>y</sub>)"
        gdlXi.addWidget(QLabel(title, sltGUI), 2, 0, 1, 6)
        gdlXi.addWidget(QLabel("a<sub>y</sub>", sltGUI), 3, 0)
        gdlXi.addWidget(self.slt["fpeAy"], 3, 1)
        gdlXi.addWidget(QLabel("T<sub>y</sub>", sltGUI), 3, 2)
        gdlXi.addWidget(self.slt["fpeTy"], 3, 3)
        gdlXi.addWidget(QLabel("&phi;<sub>y</sub>", sltGUI), 3, 4)
        gdlXi.addWidget(self.slt["fpePhiy"], 3, 5)
        title = "z<sub>1</sub> = z(t<sub>1</sub>), z<sub>2</sub> = z(t<sub>2</sub>), "
        title += "z<sub>3</sub> = z(t<sub>3</sub>), ... - "
        title += "z(t) = z<sub>1</sub>t+z<sub>2</sub>t<sup>2</sup>+z<sub>3</sub>t<sup>3</sup>+..."
        gdlXi.addWidget(QLabel(title, sltGUI), 4, 0, 1, 6)
        gdlXi.addWidget(QLabel("t<sub>i</sub> z<sub>i</sub>", sltGUI), 5, 0)
        gdlXi.addWidget(self.slt["fpeTiZi"], 5, 1, 1, 4)
        pltTZPBtn = QPushButton("Plot z(t)", sltGUI)
        pltTZPBtn.clicked.connect(self.onPltTZPBtnClick)
        gdlXi.addWidget(pltTZPBtn, 5, 5)

        # Set group box layout.
        gpbXi = QGroupBox(sltGUI)
        gpbXi.setTitle("Flight path equation")
        gpbXi.setAlignment(Qt.AlignHCenter)
        gpbXi.setLayout(gdlXi)

        return gpbXi

    def onPltTZPBtnClick(self):
        """Callback on plotting lagrange T-Z polynomial"""

        # Create or retrieve viewer.
        if not self.vwr["2D"]["tzp"] or self.vwr["2D"]["tzp"].closed:
            self.vwr["2D"]["tzp"] = viewer2DGUI(self.ctrGUI)
            self.vwr["2D"]["tzp"].setUp()
            self.vwr["2D"]["tzp"].setWindowTitle("Flight path equation: Lagrange T-Z polynomial")
            self.vwr["2D"]["tzp"].show()

        # Clear the viewer.
        self.clearViewer(vwrId="tzp")

        # Time.
        prmTf = float(self.slt["cdfTf"].text())
        vwrNbPt = float(self.slt["vwrNbPt"].text())
        eqnT = np.linspace(0., prmTf, vwrNbPt)

        # Compute lagrange Z polynomial.
        poly = self.getZPoly()
        eqnZ = poly(eqnT)

        # Plot lagrange Z polynomial.
        axis = self.vwr["2D"]["tzp"].getAxis()
        vwrLnWd = float(self.slt["vwrLnWd"].text())
        if vwrLnWd > 0.:
            vwrPosMks = float(self.slt["vwrPosMks"].text())
            clr = (0., 0., 1.) # Blue.
            axis.plot(eqnT, eqnZ, color=clr, label="z", marker="o", lw=vwrLnWd, ms=vwrPosMks)
        prmTi, prmZi = self.getZPolyPts()
        axis.scatter(prmTi, prmZi, c="r", marker="X", label="interpolation point")
        axis.legend()

        # Draw scene.
        self.vwr["2D"]["tzp"].draw()

    def fillSltGUIX0(self, sltGUI):
        """Fill solution GUI : initial conditions"""

        # Create analytic parameters GUI: initial conditions.
        gdlX0 = QGridLayout(sltGUI)
        title = "x(t = 0) = x<sub>0</sub>"
        gdlX0.addWidget(QLabel(title, sltGUI), 0, 0, 1, 2)
        gdlX0.addWidget(QLabel("x<sub>0</sub>", sltGUI), 1, 0)
        gdlX0.addWidget(self.slt["cdiX0"], 1, 1)
        title = "y(t = 0) = y<sub>0</sub>"
        gdlX0.addWidget(QLabel(title, sltGUI), 2, 0, 1, 2)
        gdlX0.addWidget(QLabel("y<sub>0</sub>", sltGUI), 3, 0)
        gdlX0.addWidget(self.slt["cdiY0"], 3, 1)
        title = "z(t = 0) = z<sub>0</sub>"
        gdlX0.addWidget(QLabel(title, sltGUI), 4, 0, 1, 2)
        gdlX0.addWidget(QLabel("z<sub>0</sub>", sltGUI), 5, 0)
        gdlX0.addWidget(self.slt["cdiZ0"], 5, 1)
        title = "V<sub>x</sub>(t = 0) = V<sub>x0</sub>"
        gdlX0.addWidget(QLabel(title, sltGUI), 0, 3, 1, 2)
        gdlX0.addWidget(QLabel("V<sub>x0</sub>", sltGUI), 1, 3)
        gdlX0.addWidget(self.slt["indVX0"], 1, 4)
        title = "V<sub>y</sub>(t = 0) = V<sub>y0</sub>"
        gdlX0.addWidget(QLabel(title, sltGUI), 2, 3, 1, 2)
        gdlX0.addWidget(QLabel("V<sub>y0</sub>", sltGUI), 3, 3)
        gdlX0.addWidget(self.slt["indVY0"], 3, 4)
        title = "V<sub>z</sub>(t = 0) = V<sub>z0</sub>"
        gdlX0.addWidget(QLabel(title, sltGUI), 4, 3, 1, 2)
        gdlX0.addWidget(QLabel("V<sub>z0</sub>", sltGUI), 5, 3)
        gdlX0.addWidget(self.slt["indVZ0"], 5, 4)
        title = "A<sub>x</sub>(t = 0) = A<sub>x0</sub>"
        gdlX0.addWidget(QLabel(title, sltGUI), 0, 6, 1, 2)
        gdlX0.addWidget(QLabel("A<sub>x0</sub>", sltGUI), 1, 6)
        gdlX0.addWidget(self.slt["indAX0"], 1, 7)
        title = "A<sub>y</sub>(t = 0) = A<sub>y0</sub>"
        gdlX0.addWidget(QLabel(title, sltGUI), 2, 6, 1, 2)
        gdlX0.addWidget(QLabel("A<sub>y0</sub>", sltGUI), 3, 6)
        gdlX0.addWidget(self.slt["indAY0"], 3, 7)
        title = "A<sub>z</sub>(t = 0) = A<sub>z0</sub>"
        gdlX0.addWidget(QLabel(title, sltGUI), 4, 6, 1, 2)
        gdlX0.addWidget(QLabel("A<sub>z0</sub>", sltGUI), 5, 6)
        gdlX0.addWidget(self.slt["indAZ0"], 5, 7)

        # Set group box layout.
        gpbX0 = QGroupBox(sltGUI)
        gpbX0.setTitle("Initial conditions")
        gpbX0.setAlignment(Qt.AlignHCenter)
        gpbX0.setLayout(gdlX0)

        return gpbX0

    def fillSltGUITf(self, sltGUI):
        """Fill solution GUI : final conditions"""

        # Create analytic parameters GUI: final conditions.
        gdlTf = QGridLayout(sltGUI)
        gdlTf.addWidget(QLabel("t<sub>f</sub>", sltGUI), 0, 0)
        gdlTf.addWidget(self.slt["cdfTf"], 0, 1)

        # Set group box layout.
        gpbTf = QGroupBox(sltGUI)
        gpbTf.setTitle("Final conditions")
        gpbTf.setAlignment(Qt.AlignHCenter)
        gpbTf.setLayout(gdlTf)

        return gpbTf

    def fillSltGUIVwr(self, sltGUI):
        """Fill solution GUI : viewer parameters"""

        # Create analytic parameters GUI: plot parameters.
        gdlVwr = QGridLayout(sltGUI)
        gdlVwr.addWidget(QLabel("Position:", sltGUI), 0, 0)
        gdlVwr.addWidget(QLabel("line width", sltGUI), 0, 1)
        gdlVwr.addWidget(self.slt["vwrLnWd"], 0, 2)
        gdlVwr.addWidget(QLabel("nb points", sltGUI), 0, 3)
        gdlVwr.addWidget(self.slt["vwrNbPt"], 0, 4)
        gdlVwr.addWidget(QLabel("marker size", sltGUI), 0, 5)
        gdlVwr.addWidget(self.slt["vwrPosMks"], 0, 6)
        gdlVwr.addWidget(QLabel("Velocity:", sltGUI), 1, 0)
        gdlVwr.addWidget(QLabel("length", sltGUI), 1, 1)
        gdlVwr.addWidget(self.slt["vwrVelLgh"], 1, 2)
        gdlVwr.addWidget(self.slt["vwrVelNrm"], 1, 3)
        gdlVwr.addWidget(QLabel("Acceleration:", sltGUI), 2, 0)
        gdlVwr.addWidget(QLabel("length", sltGUI), 2, 1)
        gdlVwr.addWidget(self.slt["vwrAccLgh"], 2, 2)
        gdlVwr.addWidget(self.slt["vwrAccNrm"], 2, 3)

        # Set group box layout.
        gpbVwr = QGroupBox(sltGUI)
        gpbVwr.setTitle("Viewer options")
        gpbVwr.setAlignment(Qt.AlignHCenter)
        gpbVwr.setLayout(gdlVwr)

        return gpbVwr

    def createMsrGUI(self):
        """Create measurement GUI"""

        # Create group box.
        msrGUI = QGroupBox(self.ctrGUI)
        msrGUI.setTitle("Measurements: data received from sensors")
        msrGUI.setAlignment(Qt.AlignHCenter)

        # Store measurement parameters.
        self.msr["addType"] = QComboBox(self.ctrGUI)
        for msr in ["x", "v", "a"]:
            self.msr["addType"].addItem(msr)
        self.msr["addT0"] = QLineEdit("0.05", self.ctrGUI)
        finalTime = self.slt["cdfTf"].text()
        self.msr["addTf"] = QLineEdit(str(float(finalTime)*0.95), self.ctrGUI)
        self.msr["addDt"] = QLineEdit("0.2", self.ctrGUI)
        self.msr["addSigma"] = QLineEdit("0.1", self.ctrGUI)
        self.msr["lstMsr"] = QListWidget(self.ctrGUI)
        self.msr["datMsr"] = {}
        self.msr["vwrPosMks"] = QLineEdit("15", self.ctrGUI)
        self.msr["vwrVelLgh"] = QLineEdit("0.1", self.ctrGUI)
        self.msr["vwrVelNrm"] = QCheckBox("Normalize", self.ctrGUI)
        self.msr["vwrAccLgh"] = QLineEdit("0.01", self.ctrGUI)
        self.msr["vwrAccNrm"] = QCheckBox("Normalize", self.ctrGUI)

        # Fill measurement GUI.
        self.fillMsrGUI(msrGUI)

        # Initialize the measurement list with GPS measurements (x, v).
        self.onAddMsrBtnClick() # Adding "x" measurement.
        self.msr["addType"].setCurrentIndex(1) # Set combo to "v" after adding "x" measurement.
        self.msr["addSigma"].setText("0.2") # Better accuracy for "x" compared to "v".
        self.onAddMsrBtnClick() # Adding "v" measurement.

        # Initialize the measurement list with accelerometer measurements (a).
        self.msr["addType"].setCurrentIndex(2) # Set combo to "a" after adding "v" measurement.
        self.msr["addDt"].setText("0.05") # Sensors (shipped on plane) provide more data than GPS.
        self.msr["addSigma"].setText("0.1")
        self.onAddMsrBtnClick() # Adding "a" measurement.

        # Reset measurement list options.
        self.msr["addType"].setCurrentIndex(0)
        self.msr["addDt"].setText("0.2")
        self.msr["addSigma"].setText("0.1")

        return msrGUI

    def fillMsrGUI(self, msrGUI):
        """Fill measurement GUI"""

        # Create group box.
        gpbAdd = self.fillMsrGUIAddMsr(msrGUI)
        gpbLst = self.fillMsrGUILstMsr(msrGUI)
        gpbVwr = self.fillMsrGUIVwrMsr(msrGUI)

        # Set group box layout.
        msrLay = QHBoxLayout(msrGUI)
        msrLay.addWidget(gpbAdd)
        msrLay.addWidget(gpbLst)
        msrLay.addWidget(gpbVwr)
        msrGUI.setLayout(msrLay)

    def fillMsrGUIAddMsr(self, msrGUI):
        """Fill measurement GUI: add measurements"""

        # Create measurement parameters GUI: add measurements.
        gdlAdd = QGridLayout(msrGUI)
        gdlAdd.addWidget(QLabel("Type:", msrGUI), 0, 0)
        gdlAdd.addWidget(self.msr["addType"], 0, 1)
        gdlAdd.addWidget(QLabel("t<sub>0</sub>:", msrGUI), 0, 2)
        gdlAdd.addWidget(self.msr["addT0"], 0, 3)
        gdlAdd.addWidget(QLabel("t<sub>f</sub>:", msrGUI), 0, 4)
        gdlAdd.addWidget(self.msr["addTf"], 0, 5)
        addMsrBtn = QPushButton("Add measure", msrGUI)
        addMsrBtn.setToolTip("Add measure to list")
        addMsrBtn.clicked.connect(self.onAddMsrBtnClick)
        gdlAdd.addWidget(addMsrBtn, 1, 0, 1, 2)
        gdlAdd.addWidget(QLabel("<em>&Delta;t</em>:", msrGUI), 1, 2)
        gdlAdd.addWidget(self.msr["addDt"], 1, 3)
        gdlAdd.addWidget(QLabel("<em>&sigma;</em>:", msrGUI), 1, 4)
        gdlAdd.addWidget(self.msr["addSigma"], 1, 5)

        # Set group box layout.
        gpbAdd = QGroupBox(msrGUI)
        gpbAdd.setTitle("Add measurements")
        gpbAdd.setAlignment(Qt.AlignHCenter)
        gpbAdd.setLayout(gdlAdd)

        return gpbAdd

    def onAddMsrBtnClick(self):
        """Callback on adding measure in list"""

        # Create new item.
        item = "type "+self.msr["addType"].currentText()
        item += "; T0 "+self.msr["addT0"].text()
        item += "; Tf "+self.msr["addTf"].text()
        item += "; Dt "+self.msr["addDt"].text()
        item += "; sigma "+self.msr["addSigma"].text()

        # Add new item.
        added = False
        for idx in range(self.msr["lstMsr"].count()):
            if self.msr["lstMsr"].item(idx).text() == "": # Unused item.
                self.msr["lstMsr"].item(idx).setText(item)
                added = True
                break
        if not added:
            self.msr["lstMsr"].addItem(item)

    def fillMsrGUILstMsr(self, msrGUI):
        """Fill measurement GUI: list measurements"""

        # Create measurement parameters GUI: list measurements.
        gdlLst = QGridLayout(msrGUI)
        gdlLst.addWidget(self.msr["lstMsr"], 0, 0)
        rmvMsrBtn = QPushButton("Remove selected measure", msrGUI)
        rmvMsrBtn.setToolTip("Remove selected measure from list")
        rmvMsrBtn.clicked.connect(self.onRmvMsrBtnClick)
        gdlLst.addWidget(rmvMsrBtn, 1, 0)

        # Set group box layout.
        gpbLst = QGroupBox(msrGUI)
        gpbLst.setTitle("List measurements")
        gpbLst.setAlignment(Qt.AlignHCenter)
        gpbLst.setLayout(gdlLst)

        return gpbLst

    def onRmvMsrBtnClick(self):
        """Callback on removing selected measure from list"""

        # Remove item.
        items = self.msr["lstMsr"].selectedItems()
        if len(items) == 0:
            eId = "measurement list"
            self.throwError(eId, "select a measure to remove from the list")
            return
        for item in items:
            for idx in range(self.msr["lstMsr"].count()):
                if item == self.msr["lstMsr"].item(idx):
                    self.msr["lstMsr"].item(idx).setText("")
                    self.msr["lstMsr"].sortItems(Qt.DescendingOrder)
                    break

    def fillMsrGUIVwrMsr(self, msrGUI):
        """Fill measurement GUI: measurement viewer options"""

        # Create measurement parameters GUI: measurement viewer options.
        gdlVwr = QGridLayout(msrGUI)
        gdlVwr.addWidget(QLabel("Position:", msrGUI), 0, 0)
        gdlVwr.addWidget(QLabel("marker size", msrGUI), 0, 1)
        gdlVwr.addWidget(self.msr["vwrPosMks"], 0, 2)
        gdlVwr.addWidget(QLabel("Velocity:", msrGUI), 1, 0)
        gdlVwr.addWidget(QLabel("length", msrGUI), 1, 1)
        gdlVwr.addWidget(self.msr["vwrVelLgh"], 1, 2)
        gdlVwr.addWidget(self.msr["vwrVelNrm"], 1, 3)
        gdlVwr.addWidget(QLabel("Acceleration:", msrGUI), 2, 0)
        gdlVwr.addWidget(QLabel("length", msrGUI), 2, 1)
        gdlVwr.addWidget(self.msr["vwrAccLgh"], 2, 2)
        gdlVwr.addWidget(self.msr["vwrAccNrm"], 2, 3)

        # Set group box layout.
        gpbVwr = QGroupBox(msrGUI)
        gpbVwr.setTitle("Viewer options")
        gpbVwr.setAlignment(Qt.AlignHCenter)
        gpbVwr.setLayout(gdlVwr)

        return gpbVwr

    def createSimGUI(self):
        """Create simulation GUI"""

        # Create group box.
        simGUI = QGroupBox(self.ctrGUI)
        simGUI.setTitle("Simulation")
        simGUI.setAlignment(Qt.AlignHCenter)

        # Store simulation parameters.
        self.sim["prmM"] = QLineEdit("1000.", self.ctrGUI)
        self.sim["prmC"] = QLineEdit("1.", self.ctrGUI)
        self.sim["prmDt"] = QLineEdit("0.03", self.ctrGUI)
        self.sim["prmExpOrd"] = QLineEdit("3", self.ctrGUI)
        self.sim["prmProNseSig"] = QLineEdit("0.05", self.ctrGUI)
        self.sim["prmVrb"] = QLineEdit("1", self.ctrGUI)
        self.sim["cdiX0"] = QLineEdit("0.1", self.ctrGUI)
        self.sim["cdiY0"] = QLineEdit("0.1", self.ctrGUI)
        self.sim["cdiZ0"] = QLineEdit("0.1", self.ctrGUI)
        self.sim["cdiSigX0"] = QLineEdit("0.2", self.ctrGUI)
        self.sim["cdiSigY0"] = QLineEdit("0.2", self.ctrGUI)
        self.sim["cdiSigZ0"] = QLineEdit("0.2", self.ctrGUI)
        self.sim["cdiVX0"] = QLineEdit("0.0", self.ctrGUI)
        self.sim["cdiVY0"] = QLineEdit("12.5", self.ctrGUI)
        self.sim["cdiVZ0"] = QLineEdit("0.2", self.ctrGUI)
        self.sim["cdiSigVX0"] = QLineEdit("0.2", self.ctrGUI)
        self.sim["cdiSigVY0"] = QLineEdit("0.2", self.ctrGUI)
        self.sim["cdiSigVZ0"] = QLineEdit("0.2", self.ctrGUI)
        self.sim["cdiAX0"] = QLineEdit("-39.5", self.ctrGUI)
        self.sim["cdiAY0"] = QLineEdit("0.", self.ctrGUI)
        self.sim["cdiAZ0"] = QLineEdit("19.7", self.ctrGUI)
        self.sim["cdiSigAX0"] = QLineEdit("0.2", self.ctrGUI)
        self.sim["cdiSigAY0"] = QLineEdit("0.2", self.ctrGUI)
        self.sim["cdiSigAZ0"] = QLineEdit("0.2", self.ctrGUI)
        self.sim["ctlRolMax"] = QLineEdit("30.", self.ctrGUI)
        self.sim["ctlPtcMax"] = QLineEdit("5.", self.ctrGUI)
        self.sim["ctlYawMax"] = QLineEdit("30.", self.ctrGUI)
        self.sim["vwrLnWd"] = QLineEdit("1.", self.ctrGUI)
        self.sim["vwrPosMks"] = QLineEdit("5", self.ctrGUI)
        self.sim["vwrVelLgh"] = QLineEdit("0.01", self.ctrGUI)
        self.sim["vwrVelNrm"] = QCheckBox("Normalize", self.ctrGUI)
        self.sim["vwrAccLgh"] = QLineEdit("0.001", self.ctrGUI)
        self.sim["vwrAccNrm"] = QCheckBox("Normalize", self.ctrGUI)

        # Fill simulation GUI.
        self.fillSimGUI(simGUI)

        return simGUI

    def fillSimGUI(self, simGUI):
        """Fill simulation GUI"""

        # Create group box.
        gpbPrm = self.fillSimGUIPrm(simGUI)
        gpbX0 = self.fillSimGUIX0(simGUI)
        gpbFCL = self.fillSimGUIFCL(simGUI)
        gpbVwr = self.fillSimGUIVwr(simGUI)
        gpbPpg = self.fillSimGUIPostPro(simGUI)

        # Set group box layout.
        simSubLay1 = QHBoxLayout()
        simSubLay1.addWidget(gpbPrm)
        simSubLay1.addWidget(gpbX0)
        simSubLay1.addWidget(gpbFCL)
        simSubLay1.addWidget(gpbVwr)
        simSubLay2 = QHBoxLayout()
        simSubLay2.addWidget(gpbPpg)
        simRootLay = QVBoxLayout(simGUI)
        simRootLay.addLayout(simSubLay1)
        simRootLay.addLayout(simSubLay2)
        simGUI.setLayout(simRootLay)

    def fillSimGUIPrm(self, simGUI):
        """Fill simulation GUI: parameters"""

        # Create simulation GUI: simulation parameters.
        gdlPrm = QGridLayout(simGUI)
        gdlPrm.addWidget(QLabel("Coefficients:", simGUI), 0, 0)
        gdlPrm.addWidget(QLabel("mass", simGUI), 0, 1)
        gdlPrm.addWidget(self.sim["prmM"], 0, 2)
        gdlPrm.addWidget(QLabel("damping", simGUI), 0, 3)
        gdlPrm.addWidget(self.sim["prmC"], 0, 4)
        gdlPrm.addWidget(QLabel("Time:", simGUI), 1, 0)
        gdlPrm.addWidget(QLabel("<em>&Delta;t</em>", simGUI), 1, 1)
        gdlPrm.addWidget(self.sim["prmDt"], 1, 2)
        gdlPrm.addWidget(QLabel("Taylor expansion order", simGUI), 1, 3)
        gdlPrm.addWidget(self.sim["prmExpOrd"], 1, 4)
        gdlPrm.addWidget(QLabel("Noise:", simGUI), 2, 0)
        gdlPrm.addWidget(QLabel("Process &sigma;<sub>pn</sub>", simGUI), 2, 1)
        gdlPrm.addWidget(self.sim["prmProNseSig"], 2, 2)
        gdlPrm.addWidget(QLabel("Solver:", simGUI), 3, 0)
        gdlPrm.addWidget(QLabel("verbose level", simGUI), 3, 1)
        gdlPrm.addWidget(self.sim["prmVrb"], 3, 2)

        # Set group box layout.
        gpbPrm = QGroupBox(simGUI)
        gpbPrm.setTitle("Simulation parameters")
        gpbPrm.setAlignment(Qt.AlignHCenter)
        gpbPrm.setLayout(gdlPrm)

        return gpbPrm

    def fillSimGUIX0(self, simGUI):
        """Fill simulation GUI : initial conditions"""

        # Create simulation GUI: initial conditions.
        gdlX0 = QGridLayout(simGUI)
        self.fillSimGUIX0Gdl(simGUI, gdlX0)
        self.fillSimGUIV0Gdl(simGUI, gdlX0)
        self.fillSimGUIA0Gdl(simGUI, gdlX0)

        # Set group box layout.
        gpbX0 = QGroupBox(simGUI)
        gpbX0.setTitle("Initial conditions")
        gpbX0.setAlignment(Qt.AlignHCenter)
        gpbX0.setLayout(gdlX0)

        return gpbX0

    def fillSimGUIX0Gdl(self, simGUI, gdlX0):
        """Fill simulation GUI : grid layout of initial conditions (X0)"""

        # Create simulation GUI: grid layout of initial conditions (X0).
        title = "x(t = 0) = x<sub>0</sub> &plusmn; &sigma;<sub>x0</sub>"
        gdlX0.addWidget(QLabel(title, simGUI), 0, 0, 1, 4)
        gdlX0.addWidget(QLabel("x<sub>0</sub>", simGUI), 1, 0)
        gdlX0.addWidget(self.sim["cdiX0"], 1, 1)
        gdlX0.addWidget(QLabel("&sigma;<sub>x0</sub>", simGUI), 1, 2)
        gdlX0.addWidget(self.sim["cdiSigX0"], 1, 3)
        title = "y(t = 0) = y<sub>0</sub> &plusmn; &sigma;<sub>y0</sub>"
        gdlX0.addWidget(QLabel(title, simGUI), 2, 0, 1, 4)
        gdlX0.addWidget(QLabel("y<sub>0</sub>", simGUI), 3, 0)
        gdlX0.addWidget(self.sim["cdiY0"], 3, 1)
        gdlX0.addWidget(QLabel("&sigma;<sub>y0</sub>", simGUI), 3, 2)
        gdlX0.addWidget(self.sim["cdiSigY0"], 3, 3)
        title = "z(t = 0) = z<sub>0</sub> &plusmn; &sigma;<sub>z0</sub>"
        gdlX0.addWidget(QLabel(title, simGUI), 4, 0, 1, 4)
        gdlX0.addWidget(QLabel("z<sub>0</sub>", simGUI), 5, 0)
        gdlX0.addWidget(self.sim["cdiZ0"], 5, 1)
        gdlX0.addWidget(QLabel("&sigma;<sub>z0</sub>", simGUI), 5, 2)
        gdlX0.addWidget(self.sim["cdiSigZ0"], 5, 3)

    def fillSimGUIV0Gdl(self, simGUI, gdlX0):
        """Fill simulation GUI : grid layout of initial conditions (V0)"""

        # Create simulation GUI: grid layout of initial conditions (V0).
        title = "V<sub>x</sub>(t = 0) = V<sub>x0</sub> &plusmn; &sigma;<sub>Vx0</sub>"
        gdlX0.addWidget(QLabel(title, simGUI), 0, 6, 1, 4)
        gdlX0.addWidget(QLabel("V<sub>x0</sub>", simGUI), 1, 6)
        gdlX0.addWidget(self.sim["cdiVX0"], 1, 7)
        gdlX0.addWidget(QLabel("&sigma;<sub>Vx0</sub>", simGUI), 1, 8)
        gdlX0.addWidget(self.sim["cdiSigVX0"], 1, 9)
        title = "V<sub>y</sub>(t = 0) = V<sub>y0</sub> &plusmn; &sigma;<sub>Vy0</sub>"
        gdlX0.addWidget(QLabel(title, simGUI), 2, 6, 1, 4)
        gdlX0.addWidget(QLabel("V<sub>y0</sub>", simGUI), 3, 6)
        gdlX0.addWidget(self.sim["cdiVY0"], 3, 7)
        gdlX0.addWidget(QLabel("&sigma;<sub>Vy0</sub>", simGUI), 3, 8)
        gdlX0.addWidget(self.sim["cdiSigVY0"], 3, 9)
        title = "V<sub>z</sub>(t = 0) = V<sub>z0</sub> &plusmn; &sigma;<sub>Vz0</sub>"
        gdlX0.addWidget(QLabel(title, simGUI), 4, 6, 1, 4)
        gdlX0.addWidget(QLabel("V<sub>z0</sub>", simGUI), 5, 6)
        gdlX0.addWidget(self.sim["cdiVZ0"], 5, 7)
        gdlX0.addWidget(QLabel("&sigma;<sub>Vz0</sub>", simGUI), 5, 8)
        gdlX0.addWidget(self.sim["cdiSigVZ0"], 5, 9)

    def fillSimGUIA0Gdl(self, simGUI, gdlX0):
        """Fill simulation GUI : grid layout of initial conditions (A0)"""

        # Create simulation GUI: grid layout of initial conditions (A0).
        title = "A<sub>x</sub>(t = 0) = A<sub>x0</sub> &plusmn; &sigma;<sub>Ax0</sub>"
        gdlX0.addWidget(QLabel(title, simGUI), 0, 10, 1, 4)
        gdlX0.addWidget(QLabel("A<sub>x0</sub>", simGUI), 1, 10)
        gdlX0.addWidget(self.sim["cdiAX0"], 1, 11)
        gdlX0.addWidget(QLabel("&sigma;<sub>Ax0</sub>", simGUI), 1, 12)
        gdlX0.addWidget(self.sim["cdiSigAX0"], 1, 13)
        title = "A<sub>y</sub>(t = 0) = A<sub>y0</sub> &plusmn; &sigma;<sub>Ay0</sub>"
        gdlX0.addWidget(QLabel(title, simGUI), 2, 10, 1, 4)
        gdlX0.addWidget(QLabel("A<sub>y0</sub>", simGUI), 3, 10)
        gdlX0.addWidget(self.sim["cdiAY0"], 3, 11)
        gdlX0.addWidget(QLabel("&sigma;<sub>Ay0</sub>", simGUI), 3, 12)
        gdlX0.addWidget(self.sim["cdiSigAY0"], 3, 13)
        title = "A<sub>z</sub>(t = 0) = A<sub>z0</sub> &plusmn; &sigma;<sub>Az0</sub>"
        gdlX0.addWidget(QLabel(title, simGUI), 4, 10, 1, 4)
        gdlX0.addWidget(QLabel("A<sub>z0</sub>", simGUI), 5, 10)
        gdlX0.addWidget(self.sim["cdiAZ0"], 5, 11)
        gdlX0.addWidget(QLabel("&sigma;<sub>Az0</sub>", simGUI), 5, 12)
        gdlX0.addWidget(self.sim["cdiSigAZ0"], 5, 13)

    def fillSimGUIFCL(self, simGUI):
        """Fill simulation GUI: simulation flight control law"""

        # Create simulation GUI: simulation flight control law.
        gdlFCL = QGridLayout(simGUI)
        gdlFCL.addWidget(QLabel("Variation max:", simGUI), 0, 0, 1, 2)
        gdlFCL.addWidget(QLabel("Roll:", simGUI), 1, 0)
        gdlFCL.addWidget(self.sim["ctlRolMax"], 1, 1)
        gdlFCL.addWidget(QLabel("Pitch:", simGUI), 2, 0)
        gdlFCL.addWidget(self.sim["ctlPtcMax"], 2, 1)
        gdlFCL.addWidget(QLabel("Yaw:", simGUI), 3, 0)
        gdlFCL.addWidget(self.sim["ctlYawMax"], 3, 1)

        # Set group box layout.
        gpbFCL = QGroupBox(simGUI)
        gpbFCL.setTitle("Control law")
        gpbFCL.setAlignment(Qt.AlignHCenter)
        gpbFCL.setLayout(gdlFCL)

        return gpbFCL

    def fillSimGUIVwr(self, simGUI):
        """Fill simulation GUI: viewer"""

        # Create simulation GUI: simulation viewer options.
        gdlVwr = QGridLayout(simGUI)
        gdlVwr.addWidget(QLabel("Position:", simGUI), 0, 0)
        gdlVwr.addWidget(QLabel("line width", simGUI), 0, 1)
        gdlVwr.addWidget(self.sim["vwrLnWd"], 0, 2)
        gdlVwr.addWidget(QLabel("marker size", simGUI), 0, 3)
        gdlVwr.addWidget(self.sim["vwrPosMks"], 0, 4)
        gdlVwr.addWidget(QLabel("Velocity:", simGUI), 1, 0)
        gdlVwr.addWidget(QLabel("length", simGUI), 1, 1)
        gdlVwr.addWidget(self.sim["vwrVelLgh"], 1, 2)
        gdlVwr.addWidget(self.sim["vwrVelNrm"], 1, 3)
        gdlVwr.addWidget(QLabel("Acceleration:", simGUI), 2, 0)
        gdlVwr.addWidget(QLabel("length", simGUI), 2, 1)
        gdlVwr.addWidget(self.sim["vwrAccLgh"], 2, 2)
        gdlVwr.addWidget(self.sim["vwrAccNrm"], 2, 3)

        # Set group box layout.
        gpbVwr = QGroupBox(simGUI)
        gpbVwr.setTitle("Viewer options")
        gpbVwr.setAlignment(Qt.AlignHCenter)
        gpbVwr.setLayout(gdlVwr)

        return gpbVwr

    def fillSimGUIPostPro(self, simGUI):
        """Fill simulation GUI: post processing"""

        # Create push buttons.
        pltSOVBtn = QPushButton("Output variables", simGUI)
        pltSOVBtn.clicked.connect(self.onPltSOVBtnClick)
        pltCHVBtn = QPushButton("Control law variables", simGUI)
        pltCHVBtn.clicked.connect(self.onPltCHVBtnClick)
        pltTscBtn = QPushButton("Time scheme", simGUI)
        pltTscBtn.clicked.connect(self.onPltTScBtnClick)
        pltCovBtn = QPushButton("Covariance", simGUI)
        pltCovBtn.clicked.connect(self.onPltCovBtnClick)
        pltSKGBtn = QPushButton("Kalman gain", simGUI)
        pltSKGBtn.clicked.connect(self.onPltSKGBtnClick)

        # Create simulation GUI: simulation post processing.
        gdlPpg = QGridLayout(simGUI)
        gdlPpg.addWidget(pltSOVBtn, 0, 0)
        gdlPpg.addWidget(pltCHVBtn, 0, 1)
        gdlPpg.addWidget(pltTscBtn, 0, 2)
        gdlPpg.addWidget(pltCovBtn, 0, 3)
        gdlPpg.addWidget(pltSKGBtn, 0, 4)

        # Set group box layout.
        gpbPpg = QGroupBox(simGUI)
        gpbPpg.setTitle("Post processing options")
        gpbPpg.setAlignment(Qt.AlignHCenter)
        gpbPpg.setLayout(gdlPpg)

        return gpbPpg

    def onPltSOVBtnClick(self):
        """Callback on plotting output variables of simulation"""

        # Create or retrieve viewer.
        if not self.vwr["2D"]["simOV"] or self.vwr["2D"]["simOV"].closed:
            self.vwr["2D"]["simOV"] = viewer2DGUI(self.ctrGUI)
            self.vwr["2D"]["simOV"].setUp(nrows=3, ncols=3)
            self.vwr["2D"]["simOV"].setWindowTitle("Simulation: outputs")
            self.vwr["2D"]["simOV"].show()

        # Clear the viewer.
        self.clearViewer(vwrId="simOV")

        # Plot simulation output variables.
        self.plotSimulationOutputVariables()

        # Draw scene.
        self.vwr["2D"]["simOV"].draw()

    def plotSimulationOutputVariables(self):
        """Plot simulation output variables"""

        # Don't plot if there's nothing to plot.
        if not self.kfm.isSolved():
            return

        # Plot simulation output variables.
        self.plotSimulationOutputVariablesX()
        self.plotSimulationOutputVariablesV()
        self.plotSimulationOutputVariablesA()

    def plotSimulationOutputVariablesX(self):
        """Plot simulation output variables: X"""

        # Plot simulation output variables.
        axis = self.vwr["2D"]["simOV"].getAxis(0)
        axis.plot(self.slt["T"], self.slt["X"], label="slt: X",
                  marker="o", ms=3, c="b")
        axis.plot(self.kfm.time, self.kfm.outputs["X"], label="sim: X",
                  marker="o", ms=3, c="g")
        vwrPosMks = float(self.msr["vwrPosMks"].text())
        if vwrPosMks > 0:
            eqnT, posX = np.array([]), np.array([])
            for txt in self.msr["datMsr"]:
                msrData = self.msr["datMsr"][txt]
                if msrData["msrType"] == "x":
                    eqnT = np.append(eqnT, msrData["T"])
                    posX = np.append(posX, msrData["X"])
            axis.scatter(eqnT, posX, c="r", marker="^", alpha=1, s=vwrPosMks, label="msr: X")
        axis.set_xlabel("t")
        axis.set_ylabel("X")
        axis.legend()
        axis = self.vwr["2D"]["simOV"].getAxis(1)
        axis.plot(self.slt["T"], self.slt["Y"], label="slt: Y",
                  marker="o", ms=3, c="b")
        axis.plot(self.kfm.time, self.kfm.outputs["Y"], label="sim: Y",
                  marker="o", ms=3, c="g")
        if vwrPosMks > 0:
            eqnT, posY = np.array([]), np.array([])
            for txt in self.msr["datMsr"]:
                msrData = self.msr["datMsr"][txt]
                if msrData["msrType"] == "x":
                    eqnT = np.append(eqnT, msrData["T"])
                    posY = np.append(posY, msrData["Y"])
            axis.scatter(eqnT, posY, c="r", marker="^", alpha=1, s=vwrPosMks, label="msr: Y")
        axis.set_xlabel("t")
        axis.set_ylabel("Y")
        axis.legend()
        axis = self.vwr["2D"]["simOV"].getAxis(2)
        axis.plot(self.slt["T"], self.slt["Z"], label="slt: Z",
                  marker="o", ms=3, c="b")
        axis.plot(self.kfm.time, self.kfm.outputs["Z"], label="sim: Z",
                  marker="o", ms=3, c="g")
        if vwrPosMks > 0:
            eqnT, posZ = np.array([]), np.array([])
            for txt in self.msr["datMsr"]:
                msrData = self.msr["datMsr"][txt]
                if msrData["msrType"] == "x":
                    eqnT = np.append(eqnT, msrData["T"])
                    posZ = np.append(posZ, msrData["Z"])
            axis.scatter(eqnT, posZ, c="r", marker="^", alpha=1, s=vwrPosMks, label="msr: Z")
        axis.set_xlabel("t")
        axis.set_ylabel("Z")
        axis.legend()

    def plotSimulationOutputVariablesV(self):
        """Plot simulation output variables: V"""

        # Plot simulation output variables.
        axis = self.vwr["2D"]["simOV"].getAxis(3)
        axis.plot(self.slt["T"], self.slt["VX"], label="slt: VX",
                  marker="o", ms=3, c="b")
        axis.plot(self.kfm.time, self.kfm.outputs["VX"], label="sim: VX",
                  marker="o", ms=3, c="g")
        vwrPosMks = float(self.msr["vwrPosMks"].text())
        if vwrPosMks > 0:
            eqnT, eqnVX = np.array([]), np.array([])
            for txt in self.msr["datMsr"]:
                msrData = self.msr["datMsr"][txt]
                if msrData["msrType"] == "v":
                    eqnT = np.append(eqnT, msrData["T"])
                    eqnVX = np.append(eqnVX, msrData["VX"])
            axis.scatter(eqnT, eqnVX, c="r", marker="o", alpha=1, s=vwrPosMks, label="msr: VX")
        axis.set_xlabel("t")
        axis.set_ylabel("VX")
        axis.legend()
        axis = self.vwr["2D"]["simOV"].getAxis(4)
        axis.plot(self.slt["T"], self.slt["VY"], label="slt: VY",
                  marker="o", ms=3, c="b")
        axis.plot(self.kfm.time, self.kfm.outputs["VY"], label="sim: VY",
                  marker="o", ms=3, c="g")
        if vwrPosMks > 0:
            eqnT, eqnVY = np.array([]), np.array([])
            for txt in self.msr["datMsr"]:
                msrData = self.msr["datMsr"][txt]
                if msrData["msrType"] == "v":
                    eqnT = np.append(eqnT, msrData["T"])
                    eqnVY = np.append(eqnVY, msrData["VY"])
            axis.scatter(eqnT, eqnVY, c="r", marker="o", alpha=1, s=vwrPosMks, label="msr: VY")
        axis.set_xlabel("t")
        axis.set_ylabel("VY")
        axis.legend()
        axis = self.vwr["2D"]["simOV"].getAxis(5)
        axis.plot(self.slt["T"], self.slt["VZ"], label="slt: VZ",
                  marker="o", ms=3, c="b")
        axis.plot(self.kfm.time, self.kfm.outputs["VZ"], label="sim: VZ",
                  marker="o", ms=3, c="g")
        if vwrPosMks > 0:
            eqnT, eqnVZ = np.array([]), np.array([])
            for txt in self.msr["datMsr"]:
                msrData = self.msr["datMsr"][txt]
                if msrData["msrType"] == "v":
                    eqnT = np.append(eqnT, msrData["T"])
                    eqnVZ = np.append(eqnVZ, msrData["VZ"])
            axis.scatter(eqnT, eqnVZ, c="r", marker="o", alpha=1, s=vwrPosMks, label="msr: VZ")
        axis.set_xlabel("t")
        axis.set_ylabel("VZ")
        axis.legend()

    def plotSimulationOutputVariablesA(self):
        """Plot simulation output variables: A"""

        # Plot simulation output variables.
        axis = self.vwr["2D"]["simOV"].getAxis(6)
        axis.plot(self.slt["T"], self.slt["AX"], label="slt: AX",
                  marker="o", ms=3, c="b")
        axis.plot(self.kfm.time, self.kfm.outputs["AX"], label="sim: AX",
                  marker="o", ms=3, c="g")
        vwrPosMks = float(self.msr["vwrPosMks"].text())
        if vwrPosMks > 0:
            eqnT, eqnAX = np.array([]), np.array([])
            for txt in self.msr["datMsr"]:
                msrData = self.msr["datMsr"][txt]
                if msrData["msrType"] == "a":
                    eqnT = np.append(eqnT, msrData["T"])
                    eqnAX = np.append(eqnAX, msrData["AX"])
            axis.scatter(eqnT, eqnAX, c="r", marker="s", alpha=1, s=vwrPosMks, label="msr: AX")
        axis.set_xlabel("t")
        axis.set_ylabel("AX")
        axis.legend()
        axis = self.vwr["2D"]["simOV"].getAxis(7)
        axis.plot(self.slt["T"], self.slt["AY"], label="slt: AY",
                  marker="o", ms=3, c="b")
        axis.plot(self.kfm.time, self.kfm.outputs["AY"], label="sim: AY",
                  marker="o", ms=3, c="g")
        if vwrPosMks > 0:
            eqnT, eqnAY = np.array([]), np.array([])
            for txt in self.msr["datMsr"]:
                msrData = self.msr["datMsr"][txt]
                if msrData["msrType"] == "a":
                    eqnT = np.append(eqnT, msrData["T"])
                    eqnAY = np.append(eqnAY, msrData["AY"])
            axis.scatter(eqnT, eqnAY, c="r", marker="s", alpha=1, s=vwrPosMks, label="msr: AY")
        axis.set_xlabel("t")
        axis.set_ylabel("AY")
        axis.legend()
        axis = self.vwr["2D"]["simOV"].getAxis(8)
        axis.plot(self.slt["T"], self.slt["AZ"], label="slt: AZ",
                  marker="o", ms=3, c="b")
        axis.plot(self.kfm.time, self.kfm.outputs["AZ"], label="sim: AZ",
                  marker="o", ms=3, c="g")
        if vwrPosMks > 0:
            eqnT, eqnAZ = np.array([]), np.array([])
            for txt in self.msr["datMsr"]:
                msrData = self.msr["datMsr"][txt]
                if msrData["msrType"] == "a":
                    eqnT = np.append(eqnT, msrData["T"])
                    eqnAZ = np.append(eqnAZ, msrData["AZ"])
            axis.scatter(eqnT, eqnAZ, c="r", marker="s", alpha=1, s=vwrPosMks, label="msr: AZ")
        axis.set_xlabel("t")
        axis.set_ylabel("AZ")
        axis.legend()

    def onPltCHVBtnClick(self):
        """Callback on plotting hidden variables of control law"""

        # Create or retrieve viewer.
        if not self.vwr["2D"]["ctlHV"] or self.vwr["2D"]["ctlHV"].closed:
            self.vwr["2D"]["ctlHV"] = viewer2DGUI(self.ctrGUI)
            self.vwr["2D"]["ctlHV"].setUp(nrows=3, ncols=3)
            self.vwr["2D"]["ctlHV"].setWindowTitle("Simulation: control law")
            self.vwr["2D"]["ctlHV"].show()

        # Clear the viewer.
        self.clearViewer(vwrId="ctlHV")

        # Plot hidden variables.
        self.plotControlLawHiddenVariables()

        # Draw scene.
        self.vwr["2D"]["ctlHV"].draw()

    def plotControlLawHiddenVariables(self):
        """Plot control law hidden variables"""

        # Don't plot if there's nothing to plot.
        if not self.kfm.isSolved():
            return

        # Plot control law hidden variables.
        time = self.kfm.time
        axis = self.vwr["2D"]["ctlHV"].getAxis(0)
        axis.plot(time, self.kfm.sim["ctlHV"]["FoM"]["X"], label="F/m - X", marker="o", ms=3)
        axis.set_xlabel("t")
        axis.set_ylabel("F/m")
        axis.legend()
        axis = self.vwr["2D"]["ctlHV"].getAxis(1)
        axis.plot(time, self.kfm.sim["ctlHV"]["FoM"]["Y"], label="F/m - Y", marker="o", ms=3)
        axis.set_xlabel("t")
        axis.set_ylabel("F/m")
        axis.legend()
        axis = self.vwr["2D"]["ctlHV"].getAxis(2)
        axis.plot(time, self.kfm.sim["ctlHV"]["FoM"]["Z"], label="F/m - Z", marker="o", ms=3)
        axis.set_xlabel("t")
        axis.set_ylabel("F/m")
        axis.legend()
        axis = self.vwr["2D"]["ctlHV"].getAxis(3)
        axis.plot(time, self.kfm.sim["ctlHV"]["d(FoM)/dt"]["X"], label="d(F/m)/dt - X",
                  marker="o", ms=3)
        axis.set_xlabel("t")
        axis.set_ylabel("d(F/m)/dt")
        axis.legend()
        axis = self.vwr["2D"]["ctlHV"].getAxis(4)
        axis.plot(time, self.kfm.sim["ctlHV"]["d(FoM)/dt"]["Y"], label="d(F/m)/dt - Y",
                  marker="o", ms=3)
        axis.set_xlabel("t")
        axis.set_ylabel("d(F/m)/dt")
        axis.legend()
        axis = self.vwr["2D"]["ctlHV"].getAxis(5)
        axis.plot(time, self.kfm.sim["ctlHV"]["d(FoM)/dt"]["Z"], label="d(F/m)/dt - Z",
                  marker="o", ms=3)
        axis.set_xlabel("t")
        axis.set_ylabel("d(F/m)/dt")
        axis.legend()
        axis = self.vwr["2D"]["ctlHV"].getAxis(6)
        axis.plot(time, self.kfm.sim["ctlHV"]["roll"], label="roll", marker="o", ms=3)
        axis.set_xlabel("t")
        axis.set_ylabel("roll")
        axis.legend()
        axis = self.vwr["2D"]["ctlHV"].getAxis(7)
        axis.plot(time, self.kfm.sim["ctlHV"]["pitch"], label="pitch", marker="o", ms=3)
        axis.set_xlabel("t")
        axis.set_ylabel("pitch")
        axis.legend()
        axis = self.vwr["2D"]["ctlHV"].getAxis(8)
        axis.plot(time, self.kfm.sim["ctlHV"]["yaw"], label="yaw", marker="o", ms=3)
        axis.set_xlabel("t")
        axis.set_ylabel("yaw")
        axis.legend()

    def onPltTScBtnClick(self):
        """Callback on plotting time scheme variables"""

        # Create or retrieve viewer.
        if not self.vwr["2D"]["timSc"] or self.vwr["2D"]["timSc"].closed:
            self.vwr["2D"]["timSc"] = viewer2DGUI(self.ctrGUI)
            self.vwr["2D"]["timSc"].setUp(nrows=1, ncols=2)
            self.vwr["2D"]["timSc"].setWindowTitle("Simulation: time scheme")
            self.vwr["2D"]["timSc"].show()

        # Clear the viewer.
        self.clearViewer(vwrId="timSc")

        # Plot hidden variables.
        self.plotTimeSchemeVariables()

        # Draw scene.
        self.vwr["2D"]["timSc"].draw()

    def plotTimeSchemeVariables(self):
        """Plot time scheme variables"""

        # Don't plot if there's nothing to plot.
        if not self.kfm.isSolved():
            return

        # Plot time scheme variables.
        time = self.kfm.time
        axis = self.vwr["2D"]["timSc"].getAxis(0)
        cfl = np.array([], dtype=float)
        for idx in range(1, len(self.kfm.time)):
            deltaT = self.kfm.time[idx]-self.kfm.time[idx-1]
            deltaX = self.kfm.outputs["X"][idx]-self.kfm.outputs["X"][idx-1]
            deltaY = self.kfm.outputs["Y"][idx]-self.kfm.outputs["Y"][idx-1]
            deltaZ = self.kfm.outputs["Z"][idx]-self.kfm.outputs["Z"][idx-1]
            deltaVX = self.kfm.outputs["VX"][idx]-self.kfm.outputs["VX"][idx-1]
            deltaVY = self.kfm.outputs["VY"][idx]-self.kfm.outputs["VY"][idx-1]
            deltaVZ = self.kfm.outputs["VZ"][idx]-self.kfm.outputs["VZ"][idx-1]
            dist = math.sqrt(deltaX*deltaX+deltaY*deltaY+deltaZ*deltaZ)
            speed = math.sqrt(deltaVX*deltaVX+deltaVY*deltaVY+deltaVZ*deltaVZ)
            cfl = np.append(cfl, speed*deltaT/dist)
        axis.plot(time[1:], cfl, label="CFL", marker="o", ms=3)
        axis.set_xlabel("t")
        axis.set_ylabel("CFL")
        axis.legend()
        axis = self.vwr["2D"]["timSc"].getAxis(1)
        title = "Taylor expansion (exponential): last term magnitude"
        axis.plot(time[1:], self.kfm.sim["taylorExpLTM"], label=title, marker="o", ms=3)
        axis.set_xlabel("t")
        axis.set_ylabel("magnitude")
        axis.legend()

    def onPltCovBtnClick(self):
        """Callback on plotting covariance diagonal terms"""

        # Create or retrieve viewer.
        if not self.vwr["2D"]["matP"] or self.vwr["2D"]["matP"].closed:
            self.vwr["2D"]["matP"] = viewer2DGUI(self.ctrGUI)
            self.vwr["2D"]["matP"].setUp(nrows=3, ncols=3)
            self.vwr["2D"]["matP"].setWindowTitle("Simulation: covariance")
            self.vwr["2D"]["matP"].show()

        # Clear the viewer.
        self.clearViewer(vwrId="matP")

        # Plot hidden variables.
        self.plotSimulationCovarianceVariables()

        # Draw scene.
        self.vwr["2D"]["matP"].draw()

    def plotSimulationCovarianceVariables(self):
        """Plot covariance diagonal terms"""

        # Don't plot if there's nothing to plot.
        if not self.kfm.isSolved():
            return

        # Plot simulation covariance variables.
        time = self.kfm.sim["matP"]["T"]
        axis = self.vwr["2D"]["matP"].getAxis(0)
        axis.plot(time, self.kfm.sim["matP"]["X"], label="$P_{xx}$", marker="o", ms=3)
        axis.set_xlabel("t")
        axis.set_ylabel("$P_{xx}$")
        axis.legend()
        axis = self.vwr["2D"]["matP"].getAxis(1)
        axis.plot(time, self.kfm.sim["matP"]["Y"], label="$P_{yy}$", marker="o", ms=3)
        axis.set_xlabel("t")
        axis.set_ylabel("$P_{yy}$")
        axis.legend()
        axis = self.vwr["2D"]["matP"].getAxis(2)
        axis.plot(time, self.kfm.sim["matP"]["Z"], label="$P_{zz}$", marker="o", ms=3)
        axis.set_xlabel("t")
        axis.set_ylabel("$P_{zz}$")
        axis.legend()
        axis = self.vwr["2D"]["matP"].getAxis(3)
        axis.plot(time, self.kfm.sim["matP"]["VX"], label="$P_{vxvx}$", marker="o", ms=3)
        axis.set_xlabel("t")
        axis.set_ylabel("$P_{vxvx}$")
        axis.legend()
        axis = self.vwr["2D"]["matP"].getAxis(4)
        axis.plot(time, self.kfm.sim["matP"]["VY"], label="$P_{vyvy}$", marker="o", ms=3)
        axis.set_xlabel("t")
        axis.set_ylabel("$P_{vyvy}$")
        axis.legend()
        axis = self.vwr["2D"]["matP"].getAxis(5)
        axis.plot(time, self.kfm.sim["matP"]["VZ"], label="$P_{vzvz}$", marker="o", ms=3)
        axis.set_xlabel("t")
        axis.set_ylabel("$P_{vzvz}$")
        axis.legend()
        axis = self.vwr["2D"]["matP"].getAxis(6)
        axis.plot(time, self.kfm.sim["matP"]["AX"], label="$P_{axax}$", marker="o", ms=3)
        axis.set_xlabel("t")
        axis.set_ylabel("$P_{axax}$")
        axis.legend()
        axis = self.vwr["2D"]["matP"].getAxis(7)
        axis.plot(time, self.kfm.sim["matP"]["AY"], label="$P_{ayay}$", marker="o", ms=3)
        axis.set_xlabel("t")
        axis.set_ylabel("$P_{ayay}$")
        axis.legend()
        axis = self.vwr["2D"]["matP"].getAxis(8)
        axis.plot(time, self.kfm.sim["matP"]["AZ"], label="$P_{azaz}$", marker="o", ms=3)
        axis.set_xlabel("t")
        axis.set_ylabel("$P_{azaz}$")
        axis.legend()

    def onPltSKGBtnClick(self):
        """Callback on plotting Kalman gain diagonal terms"""

        # Create or retrieve viewer.
        if not self.vwr["2D"]["matK"] or self.vwr["2D"]["matK"].closed:
            self.vwr["2D"]["matK"] = viewer2DGUI(self.ctrGUI)
            self.vwr["2D"]["matK"].setUp(nrows=3, ncols=3)
            self.vwr["2D"]["matK"].setWindowTitle("Simulation: Kalman gain")
            self.vwr["2D"]["matK"].show()

        # Clear the viewer.
        self.clearViewer(vwrId="matK")

        # Plot hidden variables.
        self.plotSimulationKalmanGainVariables()

        # Draw scene.
        self.vwr["2D"]["matK"].draw()

    def plotSimulationKalmanGainVariables(self):
        """Plot Kalman gain diagonal terms"""

        # Don't plot if there's nothing to plot.
        if not self.kfm.isSolved():
            return

        # Plot simulation Kalman gain variables.
        time = self.kfm.sim["matK"]["T"]
        axis = self.vwr["2D"]["matK"].getAxis(0)
        axis.plot(time, self.kfm.sim["matK"]["X"], label="$K_{xx}$", marker="o", ms=3)
        axis.set_xlabel("t")
        axis.set_ylabel("$K_{xx}$")
        axis.legend()
        axis = self.vwr["2D"]["matK"].getAxis(1)
        axis.plot(time, self.kfm.sim["matK"]["Y"], label="$K_{yy}$", marker="o", ms=3)
        axis.set_xlabel("t")
        axis.set_ylabel("$K_{yy}$")
        axis.legend()
        axis = self.vwr["2D"]["matK"].getAxis(2)
        axis.plot(time, self.kfm.sim["matK"]["Z"], label="$K_{zz}$", marker="o", ms=3)
        axis.set_xlabel("t")
        axis.set_ylabel("$K_{zz}$")
        axis.legend()
        axis = self.vwr["2D"]["matK"].getAxis(3)
        axis.plot(time, self.kfm.sim["matK"]["VX"], label="$K_{vxvx}$", marker="o", ms=3)
        axis.set_xlabel("t")
        axis.set_ylabel("$K_{vxvx}$")
        axis.legend()
        axis = self.vwr["2D"]["matK"].getAxis(4)
        axis.plot(time, self.kfm.sim["matK"]["VY"], label="$K_{vyvy}$", marker="o", ms=3)
        axis.set_xlabel("t")
        axis.set_ylabel("$K_{vyvy}$")
        axis.legend()
        axis = self.vwr["2D"]["matK"].getAxis(5)
        axis.plot(time, self.kfm.sim["matK"]["VZ"], label="$K_{vzvz}$", marker="o", ms=3)
        axis.set_xlabel("t")
        axis.set_ylabel("$K_{vzvz}$")
        axis.legend()
        axis = self.vwr["2D"]["matK"].getAxis(6)
        axis.plot(time, self.kfm.sim["matK"]["AX"], label="$K_{axax}$", marker="o", ms=3)
        axis.set_xlabel("t")
        axis.set_ylabel("$K_{axax}$")
        axis.legend()
        axis = self.vwr["2D"]["matK"].getAxis(7)
        axis.plot(time, self.kfm.sim["matK"]["AY"], label="$K_{ayay}$", marker="o", ms=3)
        axis.set_xlabel("t")
        axis.set_ylabel("$K_{ayay}$")
        axis.legend()
        axis = self.vwr["2D"]["matK"].getAxis(8)
        axis.plot(time, self.kfm.sim["matK"]["AZ"], label="$K_{azaz}$", marker="o", ms=3)
        axis.set_xlabel("t")
        axis.set_ylabel("$K_{azaz}$")
        axis.legend()

    def createVwrGUI(self, gdlVwr):
        """Create viewer GUI"""

        # Store viewer parameters.
        self.vwr["ckbSlt"] = QCheckBox("Analytic solution (blue-like)", self.ctrGUI)
        self.vwr["ckbMsr"] = QCheckBox("Measurements (red-like)", self.ctrGUI)
        self.vwr["ckbSim"] = QCheckBox("Simulation (green-like)", self.ctrGUI)
        self.vwr["ckbSlt"].setChecked(True)
        self.vwr["ckbMsr"].setChecked(True)
        self.vwr["ckbSim"].setChecked(True)

        # Create viewer parameters.
        gdlVwr.addWidget(self.vwr["ckbSlt"], 0, 0)
        gdlVwr.addWidget(self.vwr["ckbMsr"], 0, 1)
        gdlVwr.addWidget(self.vwr["ckbSim"], 0, 2)

        return 1, 3

    def checkValidity(self):
        """Check example validity"""

        # Check validity.
        if not self.checkValiditySlt():
            return False
        if not self.checkValidityMsr():
            return False
        if not self.checkValiditySim():
            return False

        return True

    def checkValiditySlt(self):
        """Check example validity: analytic solution"""

        # Check analytic solution validity.
        eId = "analytic solution"
        prmTiZi = self.slt["fpeTiZi"].text()
        for tokTiZi in prmTiZi.split(","):
            if len(tokTiZi.split()) != 2:
                self.throwError(eId, "each t<sub>i</sub> must match a z<sub>i</sub>")
                return False
            tokTi = tokTiZi.split()[0]
            if np.abs(float(tokTi)) < 1.e-6:
                self.throwError(eId, "t<sub>i</sub> must be superior than 0.")
                return False
        if float(self.slt["cdfTf"].text()) <= 0.:
            self.throwError(eId, "t<sub>f</sub> must be superior than 0.")
            return False

        return self.checkValiditySltVwr()

    def checkValiditySltVwr(self):
        """Check example validity: viewer options of analytic solution"""

        # Check viewer options of analytic solution.
        eId = "analytic solution"
        if float(self.slt["vwrNbPt"].text()) < 0:
            self.throwError(eId, "number of points must be superior than 0.")
            return False
        if float(self.slt["vwrLnWd"].text()) < 0:
            self.throwError(eId, "line width must be superior than 0.")
            return False
        if float(self.slt["vwrPosMks"].text()) < 0:
            self.throwError(eId, "position marker size must be superior than 0.")
            return False

        return True

    def checkValidityMsr(self):
        """Check example validity: measurements"""

        # Check measurements validity.
        eId = "measurements"
        for idx in range(self.msr["lstMsr"].count()):
            txt = self.msr["lstMsr"].item(idx).text()
            if txt == "":
                continue
            prmT0 = float(txt.split(";")[1].split()[1])
            if prmT0 < 0.:
                msg = "t<sub>0</sub> must be superior than 0."
                self.throwError(eId, "list item "+str(idx+1)+", "+msg)
                return False
            prmTf = float(txt.split(";")[2].split()[1])
            if prmT0 > prmTf:
                msg = "t<sub>f</sub> must be superior than t<sub>0</sub>"
                self.throwError(eId, "list item "+str(idx+1)+", "+msg)
                return False
            prmDt = float(txt.split(";")[3].split()[1])
            if prmDt <= 0.:
                msg = "<em>&Delta;t</em> must be superior than 0."
                self.throwError(eId, "list item "+str(idx+1)+", "+msg)
                return False
            prmSigma = float(txt.split(";")[4].split()[1])
            if prmSigma <= 0.:
                msg = "<em>&sigma;</em> must be superior than 0."
                self.throwError(eId, "list item "+str(idx+1)+", "+msg)
                return False

        return self.checkValidityMsrVwr()

    def checkValidityMsrVwr(self):
        """Check example validity: viewer options of measurements"""

        # Check viewer options of measurements.
        eId = "measurements"
        if float(self.msr["vwrPosMks"].text()) < 0:
            self.throwError(eId, "position marker size must be superior than 0.")
            return False

        return True

    def checkValiditySim(self):
        """Check example validity: simulation"""

        # Check simulation validity.
        eId = "simulation"
        if not self.checkValiditySimPrm():
            return False
        if not self.checkValiditySimCtl():
            return False
        if float(self.sim["prmVrb"].text()) < 0.:
            self.throwError(eId, "verbose level must be superior than 0.")
            return False

        return self.checkValiditySimVwr()

    def checkValiditySimPrm(self):
        """Check example validity: simulation parameters"""

        # Check simulation parameters validity.
        eId = "simulation"
        if float(self.sim["prmM"].text()) <= 0.:
            self.throwError(eId, "mass must be superior than 0.")
            return False
        if float(self.sim["prmC"].text()) < 0.:
            self.throwError(eId, "damping coef must be superior than 0.")
            return False
        if float(self.sim["prmDt"].text()) <= 0.:
            self.throwError(eId, "<em>&Delta;t</em> must be superior than 0.")
            return False
        if float(self.sim["prmExpOrd"].text()) <= 0.:
            self.throwError(eId, "exp. taylor expansion order must be superior than 0.")
            return False
        if float(self.sim["prmProNseSig"].text()) <= 0.:
            self.throwError(eId, "process noise std deviation must be superior than 0.")
            return False

        return True

    def checkValiditySimCtl(self):
        """Check example validity: simulation control law"""

        # Check simulation control law validity.
        eId = "simulation"
        ctlRolMax = float(self.sim["ctlRolMax"].text())
        if ctlRolMax < 0. or ctlRolMax > 90.:
            self.throwError(eId, "max roll must stay between 0° and 90°.")
            return False
        ctlPtcMax = float(self.sim["ctlPtcMax"].text())
        if ctlPtcMax < 0. or ctlPtcMax > 90.:
            self.throwError(eId, "max pitch must stay between 0° and 90°.")
            return False
        ctlYawMax = float(self.sim["ctlYawMax"].text())
        if ctlYawMax < 0. or ctlYawMax > 90.:
            self.throwError(eId, "max yaw must stay between 0° and 90°.")
            return False

        return True

    def checkValiditySimVwr(self):
        """Check example validity: viewer options of simulation"""

        # Check viewer options of simulation.
        eId = "simulation"
        if float(self.sim["vwrLnWd"].text()) < 0:
            self.throwError(eId, "line width must be superior than 0.")
            return False
        if float(self.sim["vwrPosMks"].text()) < 0:
            self.throwError(eId, "position marker size must be superior than 0.")
            return False

        return True

    @staticmethod
    def getLTISystemSize():
        """Get size of the Linear Time Invariant system"""

        # Return system size.
        return 9 # x, v, a in 3D.

    def getLTISystem(self):
        """Get matrices of the Linear Time Invariant system"""

        # Constant acceleration moving body (damped mass).
        #
        # m*a = -c*v + F (https://www.kalmanfilter.net/modeling.html)
        #
        # F = throttle force (generated by plane motors)
        #
        # |.|   |             |   | |   |       |   |   |
        # |x|   |0    1     0 |   |x|   |0  0  0|   | 0 |
        # | |   |             |   | |   |       |   |   |
        # |.|   |             |   | |   |       |   |   |
        # |v| = |0  -c/m    0 | * |v| + |0  1  0| * |F/m|
        # | |   |             |   | |   |       |   |   |
        # |.|   |             |   | |   |       |   |.  |
        # |a|   |0    0   -c/m|   |a|   |0  0  1|   |F/m|
        # | |   |             |   | |   |       |   |   |
        prmM = float(self.sim["prmM"].text())
        prmC = float(self.sim["prmC"].text())
        prmN = self.getLTISystemSize()
        matA = np.zeros((prmN, prmN), dtype=float)
        matB = np.zeros((prmN, prmN), dtype=float)
        for idx in [0, 1, 2]: # X, Y, Z
            matA[3*idx+0, 3*idx+1] = 1.
            matA[3*idx+1, 3*idx+1] = -1.*prmC/prmM
            matA[3*idx+2, 3*idx+2] = -1.*prmC/prmM
            matB[3*idx+1, 3*idx+1] = 1.
            matB[3*idx+2, 3*idx+2] = 1.

        # Outputs.
        #
        # | |   |           |   | |       |   |
        # |x|   |1    0    0|   |x|       | 0 |
        # | |   |           |   | |       |   |
        # | |   |           |   | |       |   |
        # |v| = |0    1    0| * |v| + 0 * |F/m|
        # | |   |           |   | |       |   |
        # | |   |           |   | |       |.  |
        # |a|   |0    0    1|   |a|       |F/m|
        # | |   |           |   | |       |   |
        matC = np.zeros((prmN, prmN), dtype=float)
        matD = None
        for idx in [0, 1, 2]: # X, Y, Z
            matC[3*idx+0, 3*idx+0] = 1.
            matC[3*idx+1, 3*idx+1] = 1.
            matC[3*idx+2, 3*idx+2] = 1.

        return matA, matB, matC, matD

    def initStates(self, sim):
        """Initialize states"""

        # Initialize states.
        prmN = self.getLTISystemSize()
        states = np.zeros((prmN, 1), dtype=float)
        states[0, 0] = sim["cdiX0"]
        states[1, 0] = sim["cdiVX0"]
        states[2, 0] = sim["cdiAX0"]
        states[3, 0] = sim["cdiY0"]
        states[4, 0] = sim["cdiVY0"]
        states[5, 0] = sim["cdiAY0"]
        states[6, 0] = sim["cdiZ0"]
        states[7, 0] = sim["cdiVZ0"]
        states[8, 0] = sim["cdiAZ0"]

        return states

    def initStateCovariance(self, sim):
        """Initialize state covariance"""

        # Initialize state covariance.
        prmN = self.getLTISystemSize()
        matP = np.zeros((prmN, prmN), dtype=float)
        matP[0, 0] = sim["cdiSigX0"]
        matP[1, 1] = sim["cdiSigVX0"]
        matP[2, 2] = sim["cdiSigAX0"]
        matP[3, 3] = sim["cdiSigY0"]
        matP[4, 4] = sim["cdiSigVY0"]
        matP[5, 5] = sim["cdiSigAY0"]
        matP[6, 6] = sim["cdiSigZ0"]
        matP[7, 7] = sim["cdiSigVZ0"]
        matP[8, 8] = sim["cdiSigAZ0"]

        return matP

    def computeControlLaw(self, states, sim, save=True):
        """Compute control law"""

        # Compute control law: get roll, pitch, yaw corrections.
        deltaAccRolY, deltaAccRolZ = self.computeRoll(states, sim, save)
        deltaAccPtcX, deltaAccPtcZ = self.computePitch(states, sim, save)
        deltaAccYawX, deltaAccYawY = self.computeYaw(states, sim, save)

        # Compute control law.
        fomX = deltaAccPtcX+deltaAccYawX
        fomY = deltaAccRolY+deltaAccYawY
        fomZ = deltaAccRolZ+deltaAccPtcZ
        matU = self.computeControl((fomX, fomY, fomZ), sim, save)

        # Save F/m to compute d(F/m)/dt next time.
        if save:
            sim["ctlOldFoMX"] = fomX
            sim["ctlOldFoMY"] = fomY
            sim["ctlOldFoMZ"] = fomZ

        return matU

    def computeRoll(self, states, sim, save):
        """Compute control law: roll"""

        # Compute roll around X axis.
        velNow = np.array([0., states[4, 0], states[7, 0]]) # Velocity in YZ plane.
        accNow = np.array([0., states[5, 0], states[8, 0]]) # Acceleration in YZ plane.
        prmDt = float(self.sim["prmDt"].text())
        velNxt = velNow+accNow*prmDt # New velocity in YZ plane.
        roll = np.arccos(np.dot(velNow, velNxt)/(npl.norm(velNow)*npl.norm(velNxt)))
        roll = roll*(180./np.pi) # Roll angle in degrees.

        # Save control law hidden variables.
        if save:
            sim["ctlHV"]["roll"].append(roll)

        # Control roll.
        accTgt = accNow # Target acceleration.
        ctlRolMax = float(self.sim["ctlRolMax"].text())
        while np.abs(roll) > ctlRolMax:
            accTgt = accTgt*0.95 # Decrease acceleration by 5%.
            velNxt = velNow+accTgt*prmDt # New velocity in YZ plane.
            roll = np.arccos(np.dot(velNow, velNxt)/(npl.norm(velNow)*npl.norm(velNxt)))
            roll = roll*(180./np.pi) # Roll angle in degrees.
        deltaAcc = accTgt-accNow

        return deltaAcc[1], deltaAcc[2]

    def computePitch(self, states, sim, save):
        """Compute control law: pitch"""

        # Compute pitch around Y axis.
        velNow = np.array([states[1, 0], 0., states[7, 0]]) # Velocity in XZ plane.
        accNow = np.array([states[2, 0], 0., states[8, 0]]) # Acceleration in XZ plane.
        prmDt = float(self.sim["prmDt"].text())
        velNxt = velNow+accNow*prmDt # New velocity in XZ plane.
        pitch = np.arccos(np.dot(velNow, velNxt)/(npl.norm(velNow)*npl.norm(velNxt)))
        pitch = pitch*(180./np.pi) # Pitch angle in degrees.

        # Save control law hidden variables.
        if save:
            sim["ctlHV"]["pitch"].append(pitch)

        # Control pitch.
        accTgt = accNow # Target acceleration.
        ctlPtcMax = float(self.sim["ctlPtcMax"].text())
        while np.abs(pitch) > ctlPtcMax:
            accTgt = accTgt*0.95 # Decrease acceleration by 5%.
            velNxt = velNow+accTgt*prmDt # New velocity in XZ plane.
            pitch = np.arccos(np.dot(velNow, velNxt)/(npl.norm(velNow)*npl.norm(velNxt)))
            pitch = pitch*(180./np.pi) # Pitch angle in degrees.
        deltaAcc = accTgt-accNow

        return deltaAcc[0], deltaAcc[2]

    def computeYaw(self, states, sim, save):
        """Compute control law: yaw"""

        # Compute yaw around Z axis.
        velNow = np.array([states[1, 0], states[4, 0], 0.]) # Velocity in XY plane.
        accNow = np.array([states[2, 0], states[5, 0], 0.]) # Acceleration in XY plane.
        prmDt = float(self.sim["prmDt"].text())
        velNxt = velNow+accNow*prmDt # New velocity in XY plane.
        yaw = np.arccos(np.dot(velNow, velNxt)/(npl.norm(velNow)*npl.norm(velNxt)))
        yaw = yaw*(180./np.pi) # Yaw angle in degrees.

        # Save control law hidden variables.
        if save:
            sim["ctlHV"]["yaw"].append(yaw)

        # Control yaw.
        accTgt = accNow # Target acceleration.
        ctlYawMax = float(self.sim["ctlYawMax"].text())
        while np.abs(yaw) > ctlYawMax:
            accTgt = accTgt*0.95 # Decrease acceleration by 5%.
            velNxt = velNow+accTgt*prmDt # New velocity in XY plane.
            yaw = np.arccos(np.dot(velNow, velNxt)/(npl.norm(velNow)*npl.norm(velNxt)))
            yaw = yaw*(180./np.pi) # Yaw angle in degrees.
        deltaAcc = accTgt-accNow

        return deltaAcc[0], deltaAcc[1]

    def computeControl(self, fom, sim, save):
        """Compute control"""

        # Compute control law: modify plane throttle (F/m == acceleration).
        prmN = self.getLTISystemSize()
        matU = np.zeros((prmN, 1), dtype=float)
        matU[1, 0] = fom[0]
        matU[4, 0] = fom[1]
        matU[7, 0] = fom[2]

        # Compute control law: modify plane acceleration (d(F/m)/dt).
        oldFoMX = self.sim["ctlOldFoMX"] if "ctlOldFoMX" in self.sim else 0.
        oldFoMY = self.sim["ctlOldFoMY"] if "ctlOldFoMY" in self.sim else 0.
        oldFoMZ = self.sim["ctlOldFoMZ"] if "ctlOldFoMZ" in self.sim else 0.
        prmDt = float(self.sim["prmDt"].text())
        matU[2, 0] = (fom[0]-oldFoMX)/prmDt
        matU[5, 0] = (fom[1]-oldFoMY)/prmDt
        matU[8, 0] = (fom[2]-oldFoMZ)/prmDt

        # Save control law hidden variables.
        if save:
            sim["ctlHV"]["FoM"]["X"].append(matU[1, 0])
            sim["ctlHV"]["FoM"]["Y"].append(matU[4, 0])
            sim["ctlHV"]["FoM"]["Z"].append(matU[7, 0])
            sim["ctlHV"]["d(FoM)/dt"]["X"].append(matU[2, 0])
            sim["ctlHV"]["d(FoM)/dt"]["Y"].append(matU[5, 0])
            sim["ctlHV"]["d(FoM)/dt"]["Z"].append(matU[8, 0])

        return matU

    @staticmethod
    def getStateKeys():
        """Get states keys"""

        # Get states keys.
        return ["X", "VX", "AX", "Y", "VY", "AY", "Z", "VZ", "AZ"]

    def getOutputKeys(self):
        """Get outputs keys"""

        # Get outputs keys.
        return self.getStateKeys()

class controllerGUI(QMainWindow):
    """Kalman filter controller"""

    def __init__(self):
        """Initialize"""

        # Initialize members.
        super().__init__()
        self.setWindowTitle("Kalman filter controller")
        self.examples = []
        self.examples.append(planeTrackingExample(self))
        self.comboEx, self.comboGUI = self.addExampleCombo()
        self.updateBtn = self.addUpdateVwrBtn()

        # Set window as non modal.
        self.setWindowModality(Qt.NonModal)

        # Show controls GUI.
        self.show()

        # Initialize example and viewer.
        self.onExampleChanged(self.examples[0].getName())

    def addExampleCombo(self):
        """Add combo to select example"""

        # Create a label.
        lblEx = QLabel(self)
        lblEx.setText("Example:")

        # Create a combo box for each example.
        comboEx = QComboBox(self)
        for example in self.examples:
            comboEx.addItem(example.getName())
        comboEx.activated[str].connect(self.onExampleChanged)

        # Create a layout.
        layEx = QHBoxLayout()
        layEx.addWidget(lblEx)
        layEx.addWidget(comboEx)

        # Create widget.
        comboGUI = QWidget(self)
        comboGUI.setLayout(layEx)

        return comboEx, comboGUI

    def onExampleChanged(self, txt):
        """Callback on example combo change"""

        # Customize controller GUI according to example.
        layCtr = QVBoxLayout()
        layCtr.addWidget(self.comboGUI)
        gdlVwr, gdlVwrRow, gdlVwrSpan = QGridLayout(), 0, 1
        for example in self.examples:
            if example.getName() == txt:
                sltGUI = example.createSltGUI()
                layCtr.addWidget(sltGUI)
                msrGUI = example.createMsrGUI()
                layCtr.addWidget(msrGUI)
                simGUI = example.createSimGUI()
                layCtr.addWidget(simGUI)
                gdlVwrRow, gdlVwrSpan = example.createVwrGUI(gdlVwr)
                break

        # Add viewer controls.
        gdlVwr.addWidget(self.updateBtn, gdlVwrRow, 0, 1, gdlVwrSpan)
        gpbVwr = QGroupBox()
        gpbVwr.setTitle("Viewer")
        gpbVwr.setAlignment(Qt.AlignHCenter)
        gpbVwr.setLayout(gdlVwr)
        layCtr.addWidget(gpbVwr)

        # Set up controls GUI.
        guiCtr = QWidget(self)
        guiCtr.setLayout(layCtr)
        self.setCentralWidget(guiCtr)

        # Update the view.
        self.onUpdateVwrBtnClick()

    def addUpdateVwrBtn(self):
        """Add button to update the viewer"""

        # Add button to update the viewer.
        updateBtn = QPushButton("Update viewer", self)
        updateBtn.clicked.connect(self.onUpdateVwrBtnClick)

        return updateBtn

    def onUpdateVwrBtnClick(self):
        """Callback on update viewer button click"""

        # Update the view.
        print("********** Update viewer **********")
        for example in self.examples:
            if example.getName() == self.comboEx.currentText():
                # Create viewer GUI.
                example.createViewer()

                # Check validity.
                validEx = example.checkValidity()
                if not validEx:
                    return

                # Update the view.
                example.updateViewer()
                break
        print("End of update")

# Main program.
if __name__ == "__main__":
    # Check for python3.
    assert sys.version_info.major == 3, "This script is a python3 script."

    # Create application and controls GUI.
    app = QApplication(sys.argv)
    ctrWin = controllerGUI()

    # End main program.
    sys.exit(app.exec_())
