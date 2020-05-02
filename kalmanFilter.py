#!/usr/bin/env python3

"""Kalman filter MVC (Model-View-Controller)"""

import sys
import math
import numpy as np
import numpy.linalg as npl
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
from PyQt5.QtWidgets import QListWidget, QCheckBox, QRadioButton
from PyQt5.QtWidgets import QScrollArea
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator

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
        self.rangeMin = QLineEdit("N.A.", self)
        self.rangeMax = QLineEdit("N.A.", self)
        self.rangeMin.setValidator(QDoubleValidator())
        self.rangeMax.setValidator(QDoubleValidator())

        # Set window as non modal.
        self.setWindowModality(Qt.NonModal)

        # Set the layout
        subLayout1 = QVBoxLayout()
        subLayout1.addWidget(self.toolbar)
        subLayout1.addWidget(self.mcvs)
        subLayout2 = QHBoxLayout()
        subLayout2.addWidget(QLabel("time min:", self))
        subLayout2.addWidget(self.rangeMin)
        subLayout2.addWidget(QLabel("time max:", self))
        subLayout2.addWidget(self.rangeMax)
        pltTRGBtn = QPushButton("Set range", self)
        pltTRGBtn.clicked.connect(self.onPltTRGBtnClick)
        subLayout2.addWidget(pltTRGBtn)
        rootLayout = QVBoxLayout()
        rootLayout.addLayout(subLayout1)
        rootLayout.addLayout(subLayout2)

        # Build the GUI.
        vwrGUI = QWidget(self)
        vwrGUI.setLayout(rootLayout)
        self.setCentralWidget(vwrGUI)

    def onPltTRGBtnClick(self):
        """Callback on changing plot time range"""

        # Check validity.
        rangeMin = float(self.rangeMin.text())
        rangeMax = float(self.rangeMax.text())
        if rangeMin >= rangeMax:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Error - plot: min >= max")
            msg.exec_()
            return

        # Set range for all axis.
        for row in range(self.nrows):
            for col in range(self.ncols):
                axis = self.getAxis(col+row*self.ncols)
                axis.set_xlim(rangeMin, rangeMax)

        # Draw scene.
        self.mcvs.draw()

    def setUp(self, rangeMax, rangeMin="0.", nrows=1, ncols=1):
        """Set up"""

        # Set up.
        self.nrows = nrows
        self.ncols = ncols
        self.mcvs.setUp(nrows, ncols)
        self.rangeMin.setText(rangeMin)
        self.rangeMax.setText(rangeMax)

    def getAxis(self, idx=0):
        """Get viewer axis"""

        # Return viewer axis.
        if idx < 0 or idx >= self.nrows*self.ncols:
            return None
        return self.mcvs.axes[idx]

    def draw(self):
        """Force draw of the scene"""

        # Set range and draw scene.
        self.onPltTRGBtnClick()

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
        self.sim = {"simCLV": {}, "simDgP": {}, "simDgK": {}}
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
        self.sim["simCLV"]["FoM"] = {}
        self.sim["simCLV"]["FoM"]["X"] = []
        self.sim["simCLV"]["FoM"]["Y"] = []
        self.sim["simCLV"]["FoM"]["Z"] = []
        self.sim["simCLV"]["d(FoM)/dt"] = {}
        self.sim["simCLV"]["d(FoM)/dt"]["X"] = []
        self.sim["simCLV"]["d(FoM)/dt"]["Y"] = []
        self.sim["simCLV"]["d(FoM)/dt"]["Z"] = []
        self.sim["simCLV"]["roll"] = []
        self.sim["simCLV"]["pitch"] = []
        self.sim["simCLV"]["yaw"] = []
        self.sim["simTEM"] = np.array([])

        # Clear previous covariance and Kalman gain variables.
        for key in ["simDgP", "simDgK"]:
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
        if self.sim["prmVrb"] >= 1:
            print("  "*2+"Initialisation:")
        time = 0.
        states = self.example.initStates(self.sim)
        matU = self.example.computeControlLaw(states, time, self.sim)
        outputs = self.computeOutputs(states, matU)
        matP = self.example.initStateCovariance(self.sim)
        if self.sim["prmVrb"] >= 2:
            self.printMat("X", np.transpose(states))
            self.printMat("Y", np.transpose(outputs))
            self.printMat("P", matP)
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

        # Compute Kalman gain K_{n}.
        matR, matK = self.computeKalmanGain(msrLst, matP, matH)
        self.saveK(newTime, matK)

        # Update estimate with measurement: x_{n,n} = x_{n,n-1} + K_{n}*(z_{n} - H*x_{n,n-1}).
        #
        # The Kalman gain tells how much you need to change your estimate by given a measurement:
        #   x_{n,n} = x_{n,n-1} + K_{n}*(z_{n} - H*x_{n,n-1})
        #   x_{n,n} = (I - K_{n}*H)*x_{n,n-1} + (K_{n}*H)*x_{n,n-1} + (K_{n}*H)*v_{n}
        #   x_{n,n} = (I -  alpha )*x_{n,n-1} +   alpha  *x_{n,n-1} +     constant
        matI = matZ-np.dot(matH, states) # Innovation.
        newStates = states+np.dot(matK, matI) # States correction = K_{n}*Innovation.
        if self.sim["prmVrb"] >= 2:
            self.printMat("X", np.transpose(newStates))

        # Update covariance.
        newMatP = self.updateCovariance(matK, matH, matP, matR)

        return newTime, newStates, newMatP

    def getMeasurement(self, msrLst):
        """Get measurement"""

        # Get measurement: z_{n} = H*x_{n} + v_{n}.
        prmN = self.example.getLTISystemSize()
        matZ = np.zeros((prmN, 1), dtype=float)
        matH = np.zeros((prmN, prmN), dtype=float)
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

        # Verbose on demand.
        if self.sim["prmVrb"] >= 2:
            self.printMat("Z", np.transpose(matZ))
        if self.sim["prmVrb"] >= 3:
            self.printMat("H", matH)

        return matZ, matH

    def computeKalmanGain(self, msrLst, matP, matH):
        """Compute Kalman gain"""

        # Compute measurement covariance.
        matR = self.computeMeasurementCovariance(msrLst)

        # Compute Kalman gain: K_{n} = P_{n,n-1}*Ht*(H*P_{n,n-1}*Ht + R_{n})^-1.
        matK = np.dot(matH, np.dot(matP, np.transpose(matH)))+matR
        if self.sim["prmVrb"] >= 4:
            self.printMat("H*P*Ht+R", matK)
        matK = np.dot(matP, np.dot(np.transpose(matH), npl.inv(matK)))

        # Verbose on demand.
        if self.sim["prmVrb"] >= 3:
            self.printMat("K", matK)

        return matR, matK # https://www.kalmanfilter.net/kalmanGain.html.

    def computeMeasurementCovariance(self, msrLst):
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
            print("  "*2+"Predictor: time %.3f" % newTime)
        newStates, matF, matG = self.predictStates(timeDt, newTime, states)

        # Outputs equation: y_{n+1,n+1} = C*x_{n+1,n+1} + D*u_{n+1,n+1}.
        newMatU = self.example.computeControlLaw(newStates, newTime, self.sim)
        newOutputs = self.computeOutputs(newStates, newMatU)
        if self.sim["prmVrb"] >= 2:
            self.printMat("Y", np.transpose(newOutputs))

        # Save simulation results.
        self.saveXY(newTime, newStates, newOutputs)
        self.saveP(newTime, matP)

        # Extrapolate uncertainty.
        newMatP = self.predictCovariance(matP, matF, matG)

        return newStates, newMatP

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
        self.sim["simTEM"] = np.append(self.sim["simTEM"], taylorExpLTM)

        # Compute G_{n,n}.
        matG = None
        if self.mat["B"] is not None:
            matG = np.dot(timeDt*matF, self.mat["B"])
            if self.sim["prmVrb"] >= 3:
                self.printMat("G", matG)

        # Compute process noise w_{n,n}.
        matW = self.getProcessNoise(states)

        # Compute control law u_{n,n}.
        matU = self.example.computeControlLaw(states, newTime, self.sim, save=False)
        if self.sim["prmVrb"] >= 2:
            self.printMat("U", np.transpose(matU))

        # Predictor equation: x_{n+1,n} = F*x_{n,n} + G*u_{n,n} + w_{n,n}.
        newStates = np.dot(matF, states)
        if matG is not None:
            newStates = newStates+np.dot(matG, matU)
        newStates = newStates+matW
        if self.sim["prmVrb"] >= 2:
            self.printMat("X", np.transpose(newStates))

        return newStates, matF, matG

    def predictCovariance(self, matP, matF, matG):
        """Predict covariance"""

        # Compute process noise matrix: Q_{n,n} = G_{n,n}*sigma^2*G_{n,n}t.
        varQ = self.sim["prmProNseSig"]*self.sim["prmProNseSig"]
        matQ = matG*varQ*np.transpose(matG) # https://www.kalmanfilter.net/covextrap.html.
        if self.sim["prmVrb"] >= 3:
            self.printMat("Q", matQ)

        # Covariance equation: P_{n+1,n} = F_{n,n}*P_{n,n}*F_{n,n}t + Q_{n,n}.
        newMatP = np.dot(matF, np.dot(matP, np.transpose(matF)))+matQ
        if self.sim["prmVrb"] >= 3:
            self.printMat("P", newMatP)

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
        """Save diagonal terms of covariance"""

        # Save diagonal terms of covariance.
        self.sim["simDgP"]["T"].append(time)
        keys = self.example.getStateKeys()
        for idx, key in enumerate(keys):
            self.sim["simDgP"][key].append(matP[idx, idx])

    def saveK(self, time, matK):
        """Save diagonal terms of Kalman gain"""

        # Save diagonal terms of Kalman gain.
        self.sim["simDgK"]["T"].append(time)
        keys = self.example.getStateKeys()
        for idx, key in enumerate(keys):
            self.sim["simDgK"][key].append(matK[idx, idx])

    def getProcessNoise(self, states):
        """Get process noise"""

        # Get random noise.
        prmMu, prmSigma = states, self.sim["prmProNseSig"]
        noisyStates = np.random.normal(prmMu, prmSigma)
        matW = noisyStates-states

        # Verbose on demand.
        if self.sim["prmVrb"] >= 2:
            self.printMat("W", np.transpose(matW))

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
        self.vwr["2D"]["fpeTZP"] = None
        self.vwr["2D"]["msrDat"] = None
        self.vwr["2D"]["simCLV"] = None
        self.vwr["2D"]["simOVr"] = None
        self.vwr["2D"]["simTSV"] = None
        self.vwr["2D"]["simDgP"] = None
        self.vwr["2D"]["simDgK"] = None
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
        if vwrId in ("all", "fpeTZP"):
            if self.vwr["2D"]["fpeTZP"]:
                axis = self.vwr["2D"]["fpeTZP"].getAxis()
                axis.cla()
                axis.set_xlabel("t")
                axis.set_ylabel("z")
                self.vwr["2D"]["fpeTZP"].draw()
        for key in ["msrDat", "simOVr", "simCLV", "simDgP", "simDgK"]:
            if vwrId in ("all", key):
                if self.vwr["2D"][key]:
                    for idx in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
                        axis = self.vwr["2D"][key].getAxis(idx)
                        axis.set_xlabel("t")
                        axis.cla()
                    self.vwr["2D"][key].draw()
        if vwrId in ("all", "simTSV"):
            if self.vwr["2D"]["simTSV"]:
                for idx in [0, 1]:
                    axis = self.vwr["2D"]["simTSV"].getAxis(idx)
                    axis.set_xlabel("t")
                    axis.cla()
                self.vwr["2D"]["simTSV"].draw()
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

        # Track solution features.
        self.slt["sltId"] = self.getSltId()

        # Plot solution.
        print("  "*1+"View analytic solution")
        self.updateViewerSltX()
        self.updateViewerSltV()
        self.updateViewerSltA()
        if self.vwr["2D"]["fpeTZP"] and not self.vwr["2D"]["fpeTZP"].closed:
            self.onPltTZPBtnClick()

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

        # Update V0/A0 indicators.
        self.slt["indVX0"].setText("%.6f" % eqnVX[0])
        self.slt["indVY0"].setText("%.6f" % eqnVY[0])
        self.slt["indVZ0"].setText("%.6f" % eqnVZ[0])
        self.slt["indAX0"].setText("%.6f" % eqnAX[0])
        self.slt["indAY0"].setText("%.6f" % eqnAY[0])
        self.slt["indAZ0"].setText("%.6f" % eqnAZ[0])

        # Update min/max indicators.
        eqnInd = np.sqrt(eqnX*eqnX+eqnY*eqnY+eqnZ*eqnZ)
        self.slt["indXMin"].setText("%.6f" % np.min(eqnInd))
        self.slt["indXMax"].setText("%.6f" % np.max(eqnInd))
        eqnInd = np.sqrt(eqnVX*eqnVX+eqnVY*eqnVY+eqnVZ*eqnVZ)
        self.slt["indVMin"].setText("%.6f" % np.min(eqnInd))
        self.slt["indVMax"].setText("%.6f" % np.max(eqnInd))
        eqnInd = np.sqrt(eqnAX*eqnAX+eqnAY*eqnAY+eqnAZ*eqnAZ)
        self.slt["indAMin"].setText("%.6f" % np.min(eqnInd))
        self.slt["indAMax"].setText("%.6f" % np.max(eqnInd))

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
        vwrVelALR = float(self.slt["vwrVelALR"].text())
        axis = self.vwr["3D"].getAxis()
        axis.quiver3D(eqnX, eqnY, eqnZ, eqnVX, eqnVY, eqnVZ, color=clr,
                      length=vwrVelLgh, normalize=vwrVelNrm, arrow_length_ratio=vwrVelALR,
                      label="flight path: v")

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
        vwrAccALR = float(self.slt["vwrAccALR"].text())
        axis = self.vwr["3D"].getAxis()
        axis.quiver3D(eqnX, eqnY, eqnZ, eqnAX, eqnAY, eqnAZ, colors=clr,
                      length=vwrAccLgh, normalize=vwrAccNrm, arrow_length_ratio=vwrAccALR,
                      label="flight path: a")

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
                self.msr["msrDat"].clear()
            self.computeMsr()
        self.msr["sltId"] = self.slt["sltId"]

        # Track measurement features.
        self.msr["msrId"] = self.getMsrId()

        # Plot measurements.
        print("  "*1+"View measurements")
        for txt in self.msr["msrDat"]:
            msrData = self.msr["msrDat"][txt]
            self.updateViewerMsrData(msrData)
        if self.vwr["2D"]["msrDat"] and not self.vwr["2D"]["msrDat"].closed:
            self.onPltMsrBtnClick()

    def computeMsr(self):
        """Compute measurements"""

        # Compute measurements.
        print("  "*1+"Compute measurements from analytic solution")
        for idx in range(self.msr["msrLst"].count()):
            # Skip unused items.
            txt = self.msr["msrLst"].item(idx).text()
            if txt == "":
                continue

            # Create measure data if needed.
            if txt not in self.msr["msrDat"]:
                print("  "*2+"Measurement: "+txt)
                self.msr["msrDat"][txt] = self.getMsrData(txt)

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
        eqnNoise = self.addNoise(prmSigma, eqnX, eqnY, eqnZ)
        msrData["X"] = eqnX+eqnNoise[:, 0]
        msrData["Y"] = eqnY+eqnNoise[:, 1]
        msrData["Z"] = eqnZ+eqnNoise[:, 2]

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
        msrData["X"], msrData["Y"], msrData["Z"] = self.getDisplEquations(eqnT)
        eqnVX, eqnVY, eqnVZ = self.getVelocEquations(eqnT)
        prmSigma = float(txt.split(";")[4].split()[1])
        eqnNoise = self.addNoise(prmSigma, eqnVX, eqnVY, eqnVZ)
        msrData["VX"] = eqnVX+eqnNoise[:, 0]
        msrData["VY"] = eqnVY+eqnNoise[:, 1]
        msrData["VZ"] = eqnVZ+eqnNoise[:, 2]

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
        msrData["X"], msrData["Y"], msrData["Z"] = self.getDisplEquations(eqnT)
        eqnAX, eqnAY, eqnAZ = self.getAccelEquations(eqnT)
        prmSigma = float(txt.split(";")[4].split()[1])
        eqnNoise = self.addNoise(prmSigma, eqnAX, eqnAY, eqnAZ)
        msrData["AX"] = eqnAX+eqnNoise[:, 0]
        msrData["AY"] = eqnAY+eqnNoise[:, 1]
        msrData["AZ"] = eqnAZ+eqnNoise[:, 2]

    @staticmethod
    def addNoise(prmSigma, eqnX, eqnY, eqnZ):
        """Add (gaussian) noise"""

        # Add noise to data: v_{n} such that z_{n} = H*x_{n} + v_{n}.
        eqnNorm = np.sqrt(eqnX*eqnX+eqnY*eqnY+eqnZ*eqnZ)
        eqnNoise = np.zeros((len(eqnNorm), 3), dtype=float)
        for idx, norm in enumerate(eqnNorm):
            # Skip values too close from zero.
            if np.abs(norm) < 1.e-03:
                continue

            # Generate noise.
            normNoise = np.random.normal(norm, prmSigma)
            addedNoise = normNoise-norm
            if np.abs(addedNoise) > 0.25*norm: # Cut off if too much noise.
                if addedNoise > 0.:
                    normNoise = norm*1.25
                else:
                    normNoise = norm*0.75
                addedNoise = normNoise-norm

            # Spread noise accross axis.
            eqnXYZ = np.array([eqnX[idx], eqnY[idx], eqnZ[idx]])
            noise = (eqnXYZ/norm)*addedNoise
            for axis, coord in zip([0, 1, 2], [eqnX, eqnY, eqnZ]): # X, Y, Z
                # Spread noise.
                spreadCoef = 0.
                if coord[idx] > 0.:
                    spreadCoef = np.random.uniform(0., coord[idx]/norm)
                else:
                    spreadCoef = np.random.uniform(coord[idx]/norm, 0.)
                noise[axis] = noise[axis]*(1.+spreadCoef)

                # Normalize noise.
                normNoise = np.sqrt(noise[0]*noise[0]+noise[1]*noise[1]+noise[2]*noise[2])
                noise = (noise/normNoise)*addedNoise

            # Orient noise.
            if np.dot(eqnXYZ, noise) < 0. < addedNoise:
                noise = -1.*noise
            elif addedNoise < 0. < np.dot(eqnXYZ, noise):
                noise = -1.*noise

            # Save noise.
            eqnNoise[idx, :] = noise

        return eqnNoise

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
        prmPhi = float(self.slt["fpePhix"].text())*(np.pi/180.) # In radians.
        prmT = float(self.slt["fpeTx"].text())
        prmB = prmV0-prmA*np.cos(prmPhi)
        omega = 2.*math.pi/prmT
        eqnX = prmA*np.cos(omega*eqnT+prmPhi)+prmB

        return eqnX

    def getVXEquation(self, eqnT):
        """Get X equation: velocity"""

        # Get X equation: velocity.
        prmA = float(self.slt["fpeAx"].text())
        prmPhi = float(self.slt["fpePhix"].text())*(np.pi/180.) # In radians.
        prmT = float(self.slt["fpeTx"].text())
        omega = 2.*math.pi/prmT
        eqnVX = -1.*prmA*omega*np.sin(omega*eqnT+prmPhi)

        return eqnVX

    def getAXEquation(self, eqnT):
        """Get X equation: acceleration"""

        # Get X equation: acceleration.
        prmA = float(self.slt["fpeAx"].text())
        prmPhi = float(self.slt["fpePhix"].text())*(np.pi/180.) # In radians.
        prmT = float(self.slt["fpeTx"].text())
        omega = 2.*math.pi/prmT
        eqnAX = -1.*prmA*omega*omega*np.cos(omega*eqnT+prmPhi)

        return eqnAX

    def getYEquation(self, eqnT):
        """Get Y equation: displacement"""

        # Get Y equation: displacement.
        prmV0 = float(self.slt["cdiY0"].text())
        prmA = float(self.slt["fpeAy"].text())
        prmPhi = float(self.slt["fpePhiy"].text())*(np.pi/180.) # In radians.
        prmT = float(self.slt["fpeTy"].text())
        prmB = prmV0-prmA*np.sin(prmPhi)
        omega = 2.*math.pi/prmT
        eqnY = prmA*np.sin(omega*eqnT+prmPhi)+prmB

        return eqnY

    def getVYEquation(self, eqnT):
        """Get Y equation: velocity"""

        # Get Y equation: velocity.
        prmA = float(self.slt["fpeAy"].text())
        prmPhi = float(self.slt["fpePhiy"].text())*(np.pi/180.) # In radians.
        prmT = float(self.slt["fpeTy"].text())
        omega = 2.*math.pi/prmT
        eqnVY = prmA*omega*np.cos(omega*eqnT+prmPhi)

        return eqnVY

    def getAYEquation(self, eqnT):
        """Get Y equation: acceleration"""

        # Get Y equation: acceleration.
        prmA = float(self.slt["fpeAy"].text())
        prmPhi = float(self.slt["fpePhiy"].text())*(np.pi/180.) # In radians.
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

    def updateViewerMsrData(self, msrData):
        """Update viewer: measurement data"""

        # View data measurements.
        if msrData["msrType"] == "x":
            self.updateViewerMsrDataX(msrData)
        if msrData["msrType"] == "v":
            self.updateViewerMsrDataV(msrData)
        if msrData["msrType"] == "a":
            self.updateViewerMsrDataA(msrData)

    def updateViewerMsrDataX(self, msrData):
        """Update viewer: displacement measurement data"""

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

    def updateViewerMsrDataV(self, msrData):
        """Update viewer: velocity measurement data"""

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
        vwrVelALR = float(self.msr["vwrVelALR"].text())
        axis = self.vwr["3D"].getAxis()
        axis.quiver3D(posX, posY, posZ, eqnVX, eqnVY, eqnVZ, colors=clr,
                      length=vwrVelLgh, normalize=vwrVelNrm, arrow_length_ratio=vwrVelALR,
                      label="measure: v")

    def updateViewerMsrDataA(self, msrData):
        """Update viewer: acceleration measurement data"""

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
        vwrAccALR = float(self.msr["vwrAccALR"].text())
        axis = self.vwr["3D"].getAxis()
        axis.quiver3D(posX, posY, posZ, eqnAX, eqnAY, eqnAZ, colors=clr,
                      length=vwrAccLgh, normalize=vwrAccNrm, arrow_length_ratio=vwrAccALR,
                      label="measure: a")

    def getMsrId(self):
        """Get measurements identity (track measurement features)"""

        # Get measurements identity.
        msrId = ""
        for idx in range(self.msr["msrLst"].count()):
            txt = self.msr["msrLst"].item(idx).text()
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

        # Track simulation features.
        self.sim["simId"] = self.getSimId()

        # Plot solver results.
        print("  "*1+"View simulation results")
        self.updateViewerSimX()
        self.updateViewerSimV()
        self.updateViewerSimA()
        if self.vwr["2D"]["simOVr"] and not self.vwr["2D"]["simOVr"].closed:
            self.onPltSOVBtnClick()
        if self.vwr["2D"]["simCLV"] and not self.vwr["2D"]["simCLV"].closed:
            self.onPltSCLBtnClick()
        if self.vwr["2D"]["simTSV"] and not self.vwr["2D"]["simTSV"].closed:
            self.onPltSTSBtnClick()
        if self.vwr["2D"]["simDgP"] and not self.vwr["2D"]["simDgP"].closed:
            self.onPltSCvBtnClick()
        if self.vwr["2D"]["simDgK"] and not self.vwr["2D"]["simDgK"].closed:
            self.onPltSKGBtnClick()

    def computeSim(self):
        """Compute simulation"""

        # Solve based on Kalman filter.
        print("  "*1+"Run simulation based on Kalman filter")
        self.kfm.clear()
        self.kfm.setUpSimPrm(self.sim, self.slt["cdfTf"].text())
        self.kfm.setUpMsrPrm(self.msr["msrDat"])
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
        vwrVelALR = float(self.sim["vwrVelALR"].text())
        axis = self.vwr["3D"].getAxis()
        axis.quiver3D(eqnX, eqnY, eqnZ, eqnVX, eqnVY, eqnVZ, color=clr,
                      length=vwrVelLgh, normalize=vwrVelNrm, arrow_length_ratio=vwrVelALR,
                      label="simulation: v")

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
        vwrAccALR = float(self.sim["vwrAccALR"].text())
        axis = self.vwr["3D"].getAxis()
        axis.quiver3D(eqnX, eqnY, eqnZ, eqnAX, eqnAY, eqnAZ, colors=clr,
                      length=vwrAccLgh, normalize=vwrAccNrm, arrow_length_ratio=vwrAccALR,
                      label="simulation: a")

    def throwError(self, eId, txt):
        """Throw an error message"""

        # Create error message box.
        msg = QMessageBox(self.ctrGUI)
        msg.setIcon(QMessageBox.Critical)
        msg.setText("Error"+" - "+eId+": "+txt)
        msg.exec_()

    def createPrbGUI(self):
        """Create preambule GUI"""

        # Create preambule GUI: specify units.

        lbl = QLabel("Units: distance in m, time in s, mass in kg, angle in °", self.ctrGUI)
        lbl.setAlignment(Qt.AlignHCenter)
        return lbl

    def createSltGUI(self):
        """Create solution GUI"""

        # Create group box.
        sltGUI = QGroupBox(self.ctrGUI)
        sltGUI.setTitle("Analytic solution: targeting real flight path")
        sltGUI.setAlignment(Qt.AlignHCenter)

        # Create analytic parameters.
        self.createSltGUIPrm()

        # Set default GUI.
        self.slt["indVX0"].setEnabled(False)
        self.slt["indVY0"].setEnabled(False)
        self.slt["indVZ0"].setEnabled(False)
        self.slt["indAX0"].setEnabled(False)
        self.slt["indAY0"].setEnabled(False)
        self.slt["indAZ0"].setEnabled(False)
        self.slt["indXMin"].setEnabled(False)
        self.slt["indXMax"].setEnabled(False)
        self.slt["indVMin"].setEnabled(False)
        self.slt["indVMax"].setEnabled(False)
        self.slt["indAMin"].setEnabled(False)
        self.slt["indAMax"].setEnabled(False)
        self.slt["vwrVelNrm"].setChecked(False)
        self.slt["vwrAccNrm"].setChecked(False)

        # Fill solution GUI.
        self.fillSltGUI(sltGUI)

        return sltGUI

    def createSltGUIPrm(self):
        """Create solution GUI: parameters"""

        # Create analytic parameters.
        self.slt["fpeAx"] = QLineEdit("N.A.", self.ctrGUI)
        self.slt["fpeAy"] = QLineEdit("N.A.", self.ctrGUI)
        self.slt["fpeTx"] = QLineEdit("N.A.", self.ctrGUI)
        self.slt["fpeTy"] = QLineEdit("N.A.", self.ctrGUI)
        self.slt["fpePhix"] = QLineEdit("N.A.", self.ctrGUI)
        self.slt["fpePhiy"] = QLineEdit("N.A.", self.ctrGUI)
        self.slt["fpeTiZi"] = QLineEdit("N.A.", self.ctrGUI)
        self.slt["cdiX0"] = QLineEdit("N.A.", self.ctrGUI)
        self.slt["cdiY0"] = QLineEdit("N.A.", self.ctrGUI)
        self.slt["cdiZ0"] = QLineEdit("N.A.", self.ctrGUI)
        self.slt["indVX0"] = QLineEdit("N.A.", self.ctrGUI)
        self.slt["indVY0"] = QLineEdit("N.A.", self.ctrGUI)
        self.slt["indVZ0"] = QLineEdit("N.A.", self.ctrGUI)
        self.slt["indAX0"] = QLineEdit("N.A.", self.ctrGUI)
        self.slt["indAY0"] = QLineEdit("N.A.", self.ctrGUI)
        self.slt["indAZ0"] = QLineEdit("N.A.", self.ctrGUI)
        self.slt["cdfTf"] = QLineEdit("N.A.", self.ctrGUI)
        self.slt["indXMin"] = QLineEdit("N.A.", self.ctrGUI)
        self.slt["indXMax"] = QLineEdit("N.A.", self.ctrGUI)
        self.slt["indVMin"] = QLineEdit("N.A.", self.ctrGUI)
        self.slt["indVMax"] = QLineEdit("N.A.", self.ctrGUI)
        self.slt["indAMin"] = QLineEdit("N.A.", self.ctrGUI)
        self.slt["indAMax"] = QLineEdit("N.A.", self.ctrGUI)
        self.slt["vwrNbPt"] = QLineEdit("50", self.ctrGUI)
        self.slt["vwrLnWd"] = QLineEdit("1.", self.ctrGUI)
        self.slt["vwrPosMks"] = QLineEdit("5", self.ctrGUI)
        self.slt["vwrVelLgh"] = QLineEdit("1.", self.ctrGUI)
        self.slt["vwrVelALR"] = QLineEdit("0.1", self.ctrGUI)
        self.slt["vwrVelNrm"] = QCheckBox("Normalize", self.ctrGUI)
        self.slt["vwrAccLgh"] = QLineEdit("1.", self.ctrGUI)
        self.slt["vwrAccALR"] = QLineEdit("0.1", self.ctrGUI)
        self.slt["vwrAccNrm"] = QCheckBox("Normalize", self.ctrGUI)

        # Allow only double in QLineEdit.
        for key in self.slt:
            if isinstance(self.slt[key], QLineEdit):
                self.slt[key].setValidator(QDoubleValidator())

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

        # Check validity.
        if not self.checkValiditySlt():
            return

        # Create or retrieve viewer.
        print("Plot Lagrange T-Z polynomial")
        if not self.vwr["2D"]["fpeTZP"] or self.vwr["2D"]["fpeTZP"].closed:
            self.vwr["2D"]["fpeTZP"] = viewer2DGUI(self.ctrGUI)
            self.vwr["2D"]["fpeTZP"].setUp(self.slt["cdfTf"].text())
            self.vwr["2D"]["fpeTZP"].setWindowTitle("Flight path equation: Lagrange T-Z polynomial")
            self.vwr["2D"]["fpeTZP"].show()

        # Clear the viewer.
        self.clearViewer(vwrId="fpeTZP")

        # Time.
        prmTf = float(self.slt["cdfTf"].text())
        vwrNbPt = float(self.slt["vwrNbPt"].text())
        eqnT = np.linspace(0., prmTf, vwrNbPt)

        # Compute lagrange Z polynomial.
        poly = self.getZPoly()
        eqnZ = poly(eqnT)

        # Plot lagrange Z polynomial.
        axis = self.vwr["2D"]["fpeTZP"].getAxis()
        vwrLnWd = float(self.slt["vwrLnWd"].text())
        if vwrLnWd > 0.:
            vwrPosMks = float(self.slt["vwrPosMks"].text())
            clr = (0., 0., 1.) # Blue.
            axis.plot(eqnT, eqnZ, color=clr, label="z", marker="o", lw=vwrLnWd, ms=vwrPosMks)
        prmTi, prmZi = self.getZPolyPts()
        axis.scatter(prmTi, prmZi, c="r", marker="X", label="interpolation point")
        axis.legend()

        # Draw scene.
        self.vwr["2D"]["fpeTZP"].draw()

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
        gdlTf.addWidget(QLabel("t<sub>f</sub>", sltGUI), 0, 1)
        gdlTf.addWidget(self.slt["cdfTf"], 0, 2)
        gdlTf.addWidget(QLabel("Position:", sltGUI), 1, 0)
        gdlTf.addWidget(QLabel("min", sltGUI), 1, 1)
        gdlTf.addWidget(self.slt["indXMin"], 1, 2)
        gdlTf.addWidget(QLabel("max", sltGUI), 1, 3)
        gdlTf.addWidget(self.slt["indXMax"], 1, 4)
        gdlTf.addWidget(QLabel("Velocity:", sltGUI), 2, 0)
        gdlTf.addWidget(QLabel("min", sltGUI), 2, 1)
        gdlTf.addWidget(self.slt["indVMin"], 2, 2)
        gdlTf.addWidget(QLabel("max", sltGUI), 2, 3)
        gdlTf.addWidget(self.slt["indVMax"], 2, 4)
        gdlTf.addWidget(QLabel("Acceleration:", sltGUI), 3, 0)
        gdlTf.addWidget(QLabel("min", sltGUI), 3, 1)
        gdlTf.addWidget(self.slt["indAMin"], 3, 2)
        gdlTf.addWidget(QLabel("max", sltGUI), 3, 3)
        gdlTf.addWidget(self.slt["indAMax"], 3, 4)

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
        gdlVwr.addWidget(QLabel("arrow/length ratio", sltGUI), 1, 3)
        gdlVwr.addWidget(self.slt["vwrVelALR"], 1, 4)
        gdlVwr.addWidget(self.slt["vwrVelNrm"], 1, 5)
        gdlVwr.addWidget(QLabel("Acceleration:", sltGUI), 2, 0)
        gdlVwr.addWidget(QLabel("length", sltGUI), 2, 1)
        gdlVwr.addWidget(self.slt["vwrAccLgh"], 2, 2)
        gdlVwr.addWidget(QLabel("arrow/length ratio", sltGUI), 2, 3)
        gdlVwr.addWidget(self.slt["vwrAccALR"], 2, 4)
        gdlVwr.addWidget(self.slt["vwrAccNrm"], 2, 5)

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

        # Create measurement parameters.
        self.createMsrGUIPrm()

        # Set default GUI.
        self.msr["vwrVelNrm"].setChecked(False)
        self.msr["vwrAccNrm"].setChecked(False)

        # Fill measurement GUI.
        self.fillMsrGUI(msrGUI)

        return msrGUI

    def createMsrGUIPrm(self):
        """Create measurement GUI: parameters"""

        # Create measurement parameters.
        self.msr["addType"] = QComboBox(self.ctrGUI)
        for msr in ["x", "v", "a"]:
            self.msr["addType"].addItem(msr)
        self.msr["addT0"] = QLineEdit("N.A.", self.ctrGUI)
        self.msr["addTf"] = QLineEdit("N.A.", self.ctrGUI)
        self.msr["addDt"] = QLineEdit("N.A.", self.ctrGUI)
        self.msr["addSigma"] = QLineEdit("N.A.", self.ctrGUI)
        self.msr["msrLst"] = QListWidget(self.ctrGUI)
        self.msr["msrDat"] = {}
        self.msr["vwrPosMks"] = QLineEdit("15", self.ctrGUI)
        self.msr["vwrVelLgh"] = QLineEdit("1.", self.ctrGUI)
        self.msr["vwrVelALR"] = QLineEdit("0.1", self.ctrGUI)
        self.msr["vwrVelNrm"] = QCheckBox("Normalize", self.ctrGUI)
        self.msr["vwrAccLgh"] = QLineEdit("1.", self.ctrGUI)
        self.msr["vwrAccALR"] = QLineEdit("0.1", self.ctrGUI)
        self.msr["vwrAccNrm"] = QCheckBox("Normalize", self.ctrGUI)

        # Allow only double in QLineEdit.
        for key in self.msr:
            if isinstance(self.msr[key], QLineEdit):
                self.msr[key].setValidator(QDoubleValidator())

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
        pltMsrBtn = QPushButton("plot measurements", msrGUI)
        pltMsrBtn.clicked.connect(self.onPltMsrBtnClick)
        gdlAdd.addWidget(pltMsrBtn, 2, 0, 1, 6)

        # Set group box layout.
        gpbAdd = QGroupBox(msrGUI)
        gpbAdd.setTitle("Add measurements")
        gpbAdd.setAlignment(Qt.AlignHCenter)
        gpbAdd.setLayout(gdlAdd)

        return gpbAdd

    def onPltMsrBtnClick(self):
        """Callback on plotting measurement data"""

        # Check validity.
        if not self.checkValidityMsr():
            return

        # Create or retrieve viewer.
        print("Plot measurement data")
        if not self.vwr["2D"]["msrDat"] or self.vwr["2D"]["msrDat"].closed:
            self.vwr["2D"]["msrDat"] = viewer2DGUI(self.ctrGUI)
            self.vwr["2D"]["msrDat"].setUp(self.slt["cdfTf"].text(), nrows=3, ncols=3)
            self.vwr["2D"]["msrDat"].setWindowTitle("Measurements: data")
            self.vwr["2D"]["msrDat"].show()

        # Clear the viewer.
        self.clearViewer(vwrId="msrDat")

        # Plot simulation output variables.
        self.plotMeasurementData()

        # Draw scene.
        self.vwr["2D"]["msrDat"].draw()

    def plotMeasurementData(self):
        """Plot measurement data"""

        # Compute measurements if needed.
        okSlt = 1 if self.msr["sltId"] == self.getSltId() else 0
        okMsr = 1 if self.msr["msrId"] == self.getMsrId() else 0
        if not okSlt:
            self.computeSlt()
        if not okMsr:
            self.computeMsr()

        # Plot measurement data.
        self.plotSltMsrSimVariablesX("msrDat", pltMsr=True)
        self.plotSltMsrSimVariablesV("msrDat", pltMsr=True)
        self.plotSltMsrSimVariablesA("msrDat", pltMsr=True)

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
        for idx in range(self.msr["msrLst"].count()):
            if self.msr["msrLst"].item(idx).text() == "": # Unused item.
                self.msr["msrLst"].item(idx).setText(item)
                added = True
                break
        if not added:
            self.msr["msrLst"].addItem(item)

    def fillMsrGUILstMsr(self, msrGUI):
        """Fill measurement GUI: list measurements"""

        # Create measurement parameters GUI: list measurements.
        gdlLst = QGridLayout(msrGUI)
        gdlLst.addWidget(self.msr["msrLst"], 0, 0)
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
        items = self.msr["msrLst"].selectedItems()
        if len(items) == 0:
            eId = "measurement list"
            self.throwError(eId, "select a measure to remove from the list")
            return
        for item in items:
            for idx in range(self.msr["msrLst"].count()):
                if item == self.msr["msrLst"].item(idx):
                    txt = self.msr["msrLst"].item(idx).text()
                    if txt in self.msr["msrDat"]:
                        del self.msr["msrDat"][txt]
                    self.msr["msrLst"].item(idx).setText("")
                    self.msr["msrLst"].sortItems(Qt.DescendingOrder)
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
        gdlVwr.addWidget(QLabel("arrow/length ratio", msrGUI), 1, 3)
        gdlVwr.addWidget(self.msr["vwrVelALR"], 1, 4)
        gdlVwr.addWidget(self.msr["vwrVelNrm"], 1, 5)
        gdlVwr.addWidget(QLabel("Acceleration:", msrGUI), 2, 0)
        gdlVwr.addWidget(QLabel("length", msrGUI), 2, 1)
        gdlVwr.addWidget(self.msr["vwrAccLgh"], 2, 2)
        gdlVwr.addWidget(QLabel("arrow/length ratio", msrGUI), 2, 3)
        gdlVwr.addWidget(self.msr["vwrAccALR"], 2, 4)
        gdlVwr.addWidget(self.msr["vwrAccNrm"], 2, 5)

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

        # Create simulation parameters.
        self.createSimGUIPrm()

        # Set default GUI.
        self.sim["vwrVelNrm"].setChecked(False)
        self.sim["vwrAccNrm"].setChecked(False)

        # Fill simulation GUI.
        self.fillSimGUI(simGUI)

        return simGUI

    def createSimGUIPrm(self):
        """Create simulation GUI: parameters"""

        # Create simulation parameters.
        self.sim["prmM"] = QLineEdit("N.A.", self.ctrGUI)
        self.sim["prmC"] = QLineEdit("N.A.", self.ctrGUI)
        self.sim["prmDt"] = QLineEdit("N.A.", self.ctrGUI)
        self.sim["prmExpOrd"] = QLineEdit("N.A.", self.ctrGUI)
        self.sim["prmProNseSig"] = QLineEdit("N.A.", self.ctrGUI)
        self.sim["prmVrb"] = QLineEdit("1", self.ctrGUI)
        self.sim["cdiX0"] = QLineEdit("N.A.", self.ctrGUI)
        self.sim["cdiY0"] = QLineEdit("N.A.", self.ctrGUI)
        self.sim["cdiZ0"] = QLineEdit("N.A.", self.ctrGUI)
        self.sim["cdiSigX0"] = QLineEdit("N.A.", self.ctrGUI)
        self.sim["cdiSigY0"] = QLineEdit("N.A.", self.ctrGUI)
        self.sim["cdiSigZ0"] = QLineEdit("N.A.", self.ctrGUI)
        self.sim["cdiVX0"] = QLineEdit("N.A.", self.ctrGUI)
        self.sim["cdiVY0"] = QLineEdit("N.A.", self.ctrGUI)
        self.sim["cdiVZ0"] = QLineEdit("N.A.", self.ctrGUI)
        self.sim["cdiSigVX0"] = QLineEdit("N.A.", self.ctrGUI)
        self.sim["cdiSigVY0"] = QLineEdit("N.A.", self.ctrGUI)
        self.sim["cdiSigVZ0"] = QLineEdit("N.A.", self.ctrGUI)
        self.sim["cdiAX0"] = QLineEdit("N.A.", self.ctrGUI)
        self.sim["cdiAY0"] = QLineEdit("N.A.", self.ctrGUI)
        self.sim["cdiAZ0"] = QLineEdit("N.A.", self.ctrGUI)
        self.sim["cdiSigAX0"] = QLineEdit("N.A.", self.ctrGUI)
        self.sim["cdiSigAY0"] = QLineEdit("N.A.", self.ctrGUI)
        self.sim["cdiSigAZ0"] = QLineEdit("N.A.", self.ctrGUI)
        self.sim["ctlRolMax"] = QLineEdit("N.A.", self.ctrGUI)
        self.sim["ctlPtcMax"] = QLineEdit("N.A.", self.ctrGUI)
        self.sim["ctlYawMax"] = QLineEdit("N.A.", self.ctrGUI)
        self.sim["ctlThfTkoK"] = QLineEdit("N.A.", self.ctrGUI)
        self.sim["ctlThfTkoDt"] = QLineEdit("N.A.", self.ctrGUI)
        self.sim["ctlThfFlgK"] = QLineEdit("N.A.", self.ctrGUI)
        self.sim["ctlThfLdgK"] = QLineEdit("N.A.", self.ctrGUI)
        self.sim["ctlThfLdgDt"] = QLineEdit("N.A.", self.ctrGUI)
        self.sim["vwrLnWd"] = QLineEdit("1.", self.ctrGUI)
        self.sim["vwrPosMks"] = QLineEdit("5", self.ctrGUI)
        self.sim["vwrVelLgh"] = QLineEdit("1.", self.ctrGUI)
        self.sim["vwrVelALR"] = QLineEdit("0.1", self.ctrGUI)
        self.sim["vwrVelNrm"] = QCheckBox("Normalize", self.ctrGUI)
        self.sim["vwrAccLgh"] = QLineEdit("1.", self.ctrGUI)
        self.sim["vwrAccALR"] = QLineEdit("0.1", self.ctrGUI)
        self.sim["vwrAccNrm"] = QCheckBox("Normalize", self.ctrGUI)

        # Allow only double in QLineEdit.
        for key in self.sim:
            if isinstance(self.sim[key], QLineEdit):
                self.sim[key].setValidator(QDoubleValidator())

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
        simSubLay2 = QHBoxLayout()
        simSubLay2.addWidget(gpbFCL)
        simSubLay2.addWidget(gpbPpg)
        simSubLay2.addWidget(gpbVwr)
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
        gdlFCL.addWidget(QLabel("Throttle force: F = (k/m)*V", simGUI), 0, 2, 1, 4)
        gdlFCL.addWidget(QLabel("Take-off k:", simGUI), 1, 2)
        gdlFCL.addWidget(self.sim["ctlThfTkoK"], 1, 3)
        gdlFCL.addWidget(QLabel("<em>&Delta;t</em>:", simGUI), 1, 4)
        gdlFCL.addWidget(self.sim["ctlThfTkoDt"], 1, 5)
        gdlFCL.addWidget(QLabel("Flight k:", simGUI), 2, 2)
        gdlFCL.addWidget(self.sim["ctlThfFlgK"], 2, 3)
        gdlFCL.addWidget(QLabel("Landing k:", simGUI), 3, 2)
        gdlFCL.addWidget(self.sim["ctlThfLdgK"], 3, 3)
        gdlFCL.addWidget(QLabel("<em>&Delta;t</em>:", simGUI), 3, 4)
        gdlFCL.addWidget(self.sim["ctlThfLdgDt"], 3, 5)

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
        gdlVwr.addWidget(QLabel("arrow/length ratio", simGUI), 1, 3)
        gdlVwr.addWidget(self.sim["vwrVelALR"], 1, 4)
        gdlVwr.addWidget(self.sim["vwrVelNrm"], 1, 5)
        gdlVwr.addWidget(QLabel("Acceleration:", simGUI), 2, 0)
        gdlVwr.addWidget(QLabel("length", simGUI), 2, 1)
        gdlVwr.addWidget(self.sim["vwrAccLgh"], 2, 2)
        gdlVwr.addWidget(QLabel("arrow/length ratio", simGUI), 2, 3)
        gdlVwr.addWidget(self.sim["vwrAccALR"], 2, 4)
        gdlVwr.addWidget(self.sim["vwrAccNrm"], 2, 5)

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
        pltSCLBtn = QPushButton("Control law variables", simGUI)
        pltSCLBtn.clicked.connect(self.onPltSCLBtnClick)
        pltSTSBtn = QPushButton("Time scheme", simGUI)
        pltSTSBtn.clicked.connect(self.onPltSTSBtnClick)
        pltSCvBtn = QPushButton("Covariance", simGUI)
        pltSCvBtn.clicked.connect(self.onPltSCvBtnClick)
        pltSKGBtn = QPushButton("Kalman gain", simGUI)
        pltSKGBtn.clicked.connect(self.onPltSKGBtnClick)

        # Create simulation GUI: simulation post processing.
        gdlPpg = QGridLayout(simGUI)
        gdlPpg.addWidget(pltSOVBtn, 0, 0)
        gdlPpg.addWidget(pltSTSBtn, 0, 1)
        gdlPpg.addWidget(pltSCLBtn, 1, 0, 1, 2)
        gdlPpg.addWidget(pltSCvBtn, 2, 0)
        gdlPpg.addWidget(pltSKGBtn, 2, 1)

        # Set group box layout.
        gpbPpg = QGroupBox(simGUI)
        gpbPpg.setTitle("Post processing options")
        gpbPpg.setAlignment(Qt.AlignHCenter)
        gpbPpg.setLayout(gdlPpg)

        return gpbPpg

    def onPltSOVBtnClick(self):
        """Callback on plotting simulation output variables"""

        # Create or retrieve viewer.
        print("Plot simulation output variables")
        if not self.vwr["2D"]["simOVr"] or self.vwr["2D"]["simOVr"].closed:
            self.vwr["2D"]["simOVr"] = viewer2DGUI(self.ctrGUI)
            self.vwr["2D"]["simOVr"].setUp(self.slt["cdfTf"].text(), nrows=3, ncols=3)
            self.vwr["2D"]["simOVr"].setWindowTitle("Simulation: outputs")
            self.vwr["2D"]["simOVr"].show()

        # Clear the viewer.
        self.clearViewer(vwrId="simOVr")

        # Plot simulation output variables.
        self.plotSimulationOutputVariables()

        # Draw scene.
        self.vwr["2D"]["simOVr"].draw()

    def plotSimulationOutputVariables(self):
        """Plot simulation output variables"""

        # Don't plot if there's nothing to plot.
        if not self.kfm.isSolved():
            return

        # Plot simulation output variables.
        self.plotSltMsrSimVariablesX("simOVr", pltSim=True)
        self.plotSltMsrSimVariablesV("simOVr", pltSim=True)
        self.plotSltMsrSimVariablesA("simOVr", pltSim=True)

    def plotSltMsrSimVariablesX(self, key, pltMsr=False, pltSim=False):
        """Plot variables: X"""

        # Plot variables.
        opts = {"pltMsr": pltMsr, "pltSim": pltSim}
        self.plotSltMsrSimVariables(key, 0, "X", opts)
        self.plotSltMsrSimVariables(key, 1, "Y", opts)
        self.plotSltMsrSimVariables(key, 2, "Z", opts)

    def plotSltMsrSimVariablesV(self, key, pltMsr=False, pltSim=False):
        """Plot variables: V"""

        # Plot variables.
        opts = {"pltMsr": pltMsr, "pltSim": pltSim}
        self.plotSltMsrSimVariables(key, 3, "VX", opts)
        self.plotSltMsrSimVariables(key, 4, "VY", opts)
        self.plotSltMsrSimVariables(key, 5, "VZ", opts)

    def plotSltMsrSimVariablesA(self, key, pltMsr=False, pltSim=False):
        """Plot variables: A"""

        # Plot variables.
        opts = {"pltMsr": pltMsr, "pltSim": pltSim}
        self.plotSltMsrSimVariables(key, 6, "AX", opts)
        self.plotSltMsrSimVariables(key, 7, "AY", opts)
        self.plotSltMsrSimVariables(key, 8, "AZ", opts)

    def plotSltMsrSimVariables(self, key, axisId, var, opts):
        """Plot variables"""

        # Plot variables.
        axis = self.vwr["2D"][key].getAxis(axisId)
        if opts["pltSim"]:
            time = self.kfm.time
            axis.plot(time, self.kfm.outputs[var], label="sim: "+var, marker="o", ms=3, c="g")
        if opts["pltMsr"] or self.vwr["ckbMsr"].isChecked():
            vwrPosMks = float(self.msr["vwrPosMks"].text())
            if vwrPosMks > 0:
                eqnT, posX = np.array([]), np.array([])
                for txt in self.msr["msrDat"]:
                    msrData = self.msr["msrDat"][txt]
                    if var in msrData:
                        eqnT = np.append(eqnT, msrData["T"])
                        posX = np.append(posX, msrData[var])
                axis.scatter(eqnT, posX, c="r", marker="^", alpha=1, s=vwrPosMks, label="msr: "+var)
        if self.vwr["ckbSlt"].isChecked():
            axis.plot(self.slt["T"], self.slt[var], label="slt: "+var, marker="o", ms=3, c="b")
        axis.set_xlabel("t")
        axis.set_ylabel(var)
        axis.legend()

    def onPltSCLBtnClick(self):
        """Callback on plotting simulation control law variables"""

        # Create or retrieve viewer.
        print("Plot simulation control law variables")
        if not self.vwr["2D"]["simCLV"] or self.vwr["2D"]["simCLV"].closed:
            self.vwr["2D"]["simCLV"] = viewer2DGUI(self.ctrGUI)
            self.vwr["2D"]["simCLV"].setUp(self.slt["cdfTf"].text(), nrows=3, ncols=3)
            self.vwr["2D"]["simCLV"].setWindowTitle("Simulation: control law")
            self.vwr["2D"]["simCLV"].show()

        # Clear the viewer.
        self.clearViewer(vwrId="simCLV")

        # Plot simulation control law variables.
        self.plotSimulationControlLawVariables()

        # Draw scene.
        self.vwr["2D"]["simCLV"].draw()

    def plotSimulationControlLawVariables(self):
        """Plot simulation control law variables"""

        # Don't plot if there's nothing to plot.
        if not self.kfm.isSolved():
            return

        # Plot simulation control law variables.
        key = "simCLV"
        time = self.kfm.time
        for subKey, lbl, idxBase in zip(["FoM", "d(FoM)/dt"], ["F/m", "d(F/m)/dt"], [0, 3]):
            for idx, var in enumerate(["X", "Y", "Z"]):
                axis = self.vwr["2D"][key].getAxis(idxBase+idx)
                axis.plot(time, self.kfm.sim[key][subKey][var],
                          label=lbl+" - "+var, marker="o", ms=3, c="g")
                axis.set_xlabel("t")
                axis.set_ylabel(lbl+" - "+var)
                axis.legend()
        self.plotSimVariables(key, 6, "roll", "roll")
        self.plotSimVariables(key, 7, "pitch", "pitch")
        self.plotSimVariables(key, 8, "yaw", "yaw")

    def onPltSTSBtnClick(self):
        """Callback on plotting simulation time scheme variables"""

        # Create or retrieve viewer.
        print("Plot simulation time scheme variables")
        if not self.vwr["2D"]["simTSV"] or self.vwr["2D"]["simTSV"].closed:
            self.vwr["2D"]["simTSV"] = viewer2DGUI(self.ctrGUI)
            self.vwr["2D"]["simTSV"].setUp(self.slt["cdfTf"].text(), nrows=1, ncols=2)
            self.vwr["2D"]["simTSV"].setWindowTitle("Simulation: time scheme")
            self.vwr["2D"]["simTSV"].show()

        # Clear the viewer.
        self.clearViewer(vwrId="simTSV")

        # Plot simulation time scheme variables.
        self.plotSimulationTimeSchemeVariables()

        # Draw scene.
        self.vwr["2D"]["simTSV"].draw()

    def plotSimulationTimeSchemeVariables(self):
        """Plot simulation time scheme variables"""

        # Don't plot if there's nothing to plot.
        if not self.kfm.isSolved():
            return

        # Plot simulation time scheme variables.
        key = "simTSV"
        axis = self.vwr["2D"][key].getAxis(0)
        cfl = np.array([], dtype=float)
        for idx in range(1, len(self.kfm.time)):
            deltaT = self.kfm.time[idx]-self.kfm.time[idx-1]
            deltaX = self.kfm.outputs["X"][idx]-self.kfm.outputs["X"][idx-1]
            deltaY = self.kfm.outputs["Y"][idx]-self.kfm.outputs["Y"][idx-1]
            deltaZ = self.kfm.outputs["Z"][idx]-self.kfm.outputs["Z"][idx-1]
            deltaVX = self.kfm.outputs["VX"][idx]-self.kfm.outputs["VX"][idx-1]
            deltaVY = self.kfm.outputs["VY"][idx]-self.kfm.outputs["VY"][idx-1]
            deltaVZ = self.kfm.outputs["VZ"][idx]-self.kfm.outputs["VZ"][idx-1]
            deltaDist = math.sqrt(deltaX*deltaX+deltaY*deltaY+deltaZ*deltaZ)
            deltaVel = math.sqrt(deltaVX*deltaVX+deltaVY*deltaVY+deltaVZ*deltaVZ)
            cfl = np.append(cfl, deltaVel*deltaT/deltaDist)
        axis.plot(self.kfm.time[1:], cfl, label="CFL", marker="o", ms=3, c="g")
        axis.set_xlabel("t")
        axis.set_ylabel("CFL")
        axis.legend()
        axis = self.vwr["2D"][key].getAxis(1)
        title = "Taylor expansion (exponential): last term magnitude"
        axis.plot(self.kfm.time[1:], self.kfm.sim["simTEM"], label=title, marker="o", ms=3, c="g")
        axis.set_xlabel("t")
        axis.set_ylabel("magnitude")
        axis.legend()

    def onPltSCvBtnClick(self):
        """Callback on plotting simulation covariance diagonal terms"""

        # Create or retrieve viewer.
        print("Plot simulation covariance diagonal terms")
        if not self.vwr["2D"]["simDgP"] or self.vwr["2D"]["simDgP"].closed:
            self.vwr["2D"]["simDgP"] = viewer2DGUI(self.ctrGUI)
            self.vwr["2D"]["simDgP"].setUp(self.slt["cdfTf"].text(), nrows=3, ncols=3)
            self.vwr["2D"]["simDgP"].setWindowTitle("Simulation: covariance")
            self.vwr["2D"]["simDgP"].show()

        # Clear the viewer.
        self.clearViewer(vwrId="simDgP")

        # Plot simulation covariance variables.
        self.plotSimulationCovarianceVariables()

        # Draw scene.
        self.vwr["2D"]["simDgP"].draw()

    def plotSimulationCovarianceVariables(self):
        """Plot covariance diagonal terms"""

        # Don't plot if there's nothing to plot.
        if not self.kfm.isSolved():
            return

        # Plot simulation covariance variables.
        self.plotSimVariables("simDgP", 0, "X", "$P_{xx}$")
        self.plotSimVariables("simDgP", 1, "Y", "$P_{yy}$")
        self.plotSimVariables("simDgP", 2, "Z", "$P_{zz}$")
        self.plotSimVariables("simDgP", 3, "VX", "$P_{vxvx}$")
        self.plotSimVariables("simDgP", 4, "VY", "$P_{vyvy}$")
        self.plotSimVariables("simDgP", 5, "VZ", "$P_{vzvz}$")
        self.plotSimVariables("simDgP", 6, "AX", "$P_{axax}$")
        self.plotSimVariables("simDgP", 7, "AY", "$P_{ayay}$")
        self.plotSimVariables("simDgP", 8, "AZ", "$P_{azaz}$")

    def onPltSKGBtnClick(self):
        """Callback on plotting simulation Kalman gain diagonal terms"""

        # Create or retrieve viewer.
        print("Plot simulation Kalman gain diagonal terms")
        if not self.vwr["2D"]["simDgK"] or self.vwr["2D"]["simDgK"].closed:
            self.vwr["2D"]["simDgK"] = viewer2DGUI(self.ctrGUI)
            self.vwr["2D"]["simDgK"].setUp(self.slt["cdfTf"].text(), nrows=3, ncols=3)
            self.vwr["2D"]["simDgK"].setWindowTitle("Simulation: Kalman gain")
            self.vwr["2D"]["simDgK"].show()

        # Clear the viewer.
        self.clearViewer(vwrId="simDgK")

        # Plot Kalman gain variables.
        self.plotSimulationKalmanGainVariables()

        # Draw scene.
        self.vwr["2D"]["simDgK"].draw()

    def plotSimulationKalmanGainVariables(self):
        """Plot Kalman gain diagonal terms"""

        # Don't plot if there's nothing to plot.
        if not self.kfm.isSolved():
            return

        # Plot simulation Kalman gain variables.
        self.plotSimVariables("simDgK", 0, "X", "$K_{xx}$")
        self.plotSimVariables("simDgK", 1, "Y", "$K_{yy}$")
        self.plotSimVariables("simDgK", 2, "Z", "$K_{zz}$")
        self.plotSimVariables("simDgK", 3, "VX", "$K_{vxvx}$")
        self.plotSimVariables("simDgK", 4, "VY", "$K_{vyvy}$")
        self.plotSimVariables("simDgK", 5, "VZ", "$K_{vzvz}$")
        self.plotSimVariables("simDgK", 6, "AX", "$K_{axax}$")
        self.plotSimVariables("simDgK", 7, "AY", "$K_{ayay}$")
        self.plotSimVariables("simDgK", 8, "AZ", "$K_{azaz}$")

    def plotSimVariables(self, key, axisId, var, lbl):
        """Plot simulation variables"""

        # Plot simulation variables.
        axis = self.vwr["2D"][key].getAxis(axisId)
        time = self.kfm.sim[key]["T"] if "T" in self.kfm.sim[key] else self.kfm.time
        axis.plot(time, self.kfm.sim[key][var], label=lbl, marker="o", ms=3, c="g")
        axis.set_xlabel("t")
        axis.set_ylabel(lbl)
        axis.legend()

    def createExpGUI(self):
        """Create examples GUI"""

        # Create group box.
        expGUI = QGroupBox(self.ctrGUI)
        expGUI.setTitle("Examples")
        expGUI.setAlignment(Qt.AlignHCenter)

        # Set radio button.
        qrbSL = QRadioButton("Straight line", self.ctrGUI)
        qrbUD = QRadioButton("Up-down", self.ctrGUI)
        qrbZZ = QRadioButton("Zig-zag", self.ctrGUI)
        qrbRT = QRadioButton("Round trip", self.ctrGUI)
        qrbLP = QRadioButton("Looping", self.ctrGUI)
        qrbSL.toggled.connect(self.onExampleClicked)
        qrbUD.toggled.connect(self.onExampleClicked)
        qrbZZ.toggled.connect(self.onExampleClicked)
        qrbRT.toggled.connect(self.onExampleClicked)
        qrbLP.toggled.connect(self.onExampleClicked)
        qrbSL.setChecked(True)

        # Set group box layout.
        expLay = QHBoxLayout()
        expLay.addWidget(qrbSL)
        expLay.addWidget(qrbUD)
        expLay.addWidget(qrbZZ)
        expLay.addWidget(qrbRT)
        expLay.addWidget(qrbLP)
        expGUI.setLayout(expLay)

        return expGUI

    def onExampleClicked(self):
        """Callback on click: example radio button"""

        # Reset indicators.
        for key in ["indVX0", "indVY0", "indVZ0", "indAX0", "indAY0", "indAZ0"]:
            self.slt[key].setText("N.A.")
        for key in ["indXMin", "indXMax", "indVMin", "indVMax", "indAMin", "indAMax"]:
            self.slt[key].setText("N.A.")

        # Set parameters according to example.
        sigPosGPS = 2. # GPS: sigma x (5 m but often more precise).
        sigVelGPS = 0.5 # GPS: sigma v (0.5 m/s).
        sigAccSensor = 0.0001 # IMU accelerometers: sigma a (from 10 mg to 1µg).
        qrb = self.ctrGUI.sender()
        if qrb.isChecked():
            self.sim["prmM"].setText("1000.")
            self.sim["prmC"].setText("50.")
            self.sim["ctlThfTkoK"].setText("60")
            self.sim["ctlThfTkoDt"].setText("300.")
            self.sim["ctlThfFlgK"].setText("55")
            self.sim["ctlThfLdgDt"].setText("300.")
            self.sim["ctlThfLdgK"].setText("40")
            if qrb.text() == "Straight line":
                self.onStraightLineExampleClicked(sigPosGPS, sigVelGPS, sigAccSensor)
            if qrb.text() == "Up-down":
                self.onUpDownExampleClicked(sigPosGPS, sigVelGPS, sigAccSensor)
            if qrb.text() == "Zig-zag":
                self.onZigZagExampleClicked(sigPosGPS, sigVelGPS, sigAccSensor)
            if qrb.text() == "Round trip":
                self.onRoundTripExampleClicked(sigPosGPS, sigVelGPS, sigAccSensor)
            if qrb.text() == "Looping":
                self.onLoopingExampleClicked(sigPosGPS, sigVelGPS, sigAccSensor)
            self.sim["ctlRolMax"].setText("45.")
            self.sim["ctlPtcMax"].setText("30.")
            self.sim["ctlYawMax"].setText("45.")

        # Reset all previous measurements.
        for idx in range(self.msr["msrLst"].count()):
            self.msr["msrLst"].item(idx).setText("")

        # Initialize the measurement list with GPS measurements (x, v).
        self.msr["addType"].setCurrentIndex(0) # Set combo to "x".
        self.msr["addT0"].setText("0.")
        self.msr["addTf"].setText("3600.")
        self.msr["addDt"].setText("1.") # GPS frequency: 1 s.
        self.msr["addSigma"].setText("%.6f" % sigPosGPS)
        self.onAddMsrBtnClick() # Adding "x" measurement.
        self.msr["addType"].setCurrentIndex(1) # Set combo to "v".
        self.msr["addT0"].setText("0.")
        self.msr["addTf"].setText("3600.")
        self.msr["addDt"].setText("1.") # GPS frequency: 1 s.
        self.msr["addSigma"].setText("%.6f" % sigVelGPS)
        self.onAddMsrBtnClick() # Adding "v" measurement.

        # Initialize the measurement list with accelerometer measurements (a).
        self.msr["addType"].setCurrentIndex(2) # Set combo to "a".
        self.msr["addT0"].setText("0.")
        self.msr["addTf"].setText("3600.")
        self.msr["addDt"].setText("0.1") # IMU sensors provide more data than GPS.
        self.msr["addSigma"].setText("%.6f" % sigAccSensor)
        self.onAddMsrBtnClick() # Adding "a" measurement.

    def onStraightLineExampleClicked(self, sigPosGPS, sigVelGPS, sigAccSensor):
        """Callback on click: straight line example radio button"""

        # Flight path equation: parameters.
        self.slt["fpeAx"].setText("10000.")
        self.slt["fpeAy"].setText("10000.")
        self.slt["fpeTx"].setText("360000.")
        self.slt["fpeTy"].setText("360000.")
        self.slt["fpePhix"].setText("270.")
        self.slt["fpePhiy"].setText("0.")
        self.slt["fpeTiZi"].setText("3600 10000.")
        self.slt["cdiX0"].setText("0.")
        self.slt["cdiY0"].setText("0.")
        self.slt["cdiZ0"].setText("0.")
        self.slt["cdfTf"].setText("3600.")

        # Evaluate sigma: simulation sigma (less trusted) > GPS sigma (more trusted).
        sigPosSim = 2.*sigPosGPS
        sigVelSim = 2.*sigVelGPS
        sigAccSim = 2.*sigAccSensor

        # Simulation: parameters.
        self.sim["prmDt"].setText("5.")
        self.sim["prmExpOrd"].setText("3")
        self.sim["prmProNseSig"].setText("0.1")
        self.sim["cdiX0"].setText("0.5")
        self.sim["cdiY0"].setText("0.5")
        self.sim["cdiZ0"].setText("0.")
        self.sim["cdiSigX0"].setText("%.6f" % sigPosSim)
        self.sim["cdiSigY0"].setText("%.6f" % sigPosSim)
        self.sim["cdiSigZ0"].setText("%.6f" % sigPosSim)
        self.sim["cdiVX0"].setText("0.")
        self.sim["cdiVY0"].setText("0.")
        self.sim["cdiVZ0"].setText("3.")
        self.sim["cdiSigVX0"].setText("%.6f" % sigVelSim)
        self.sim["cdiSigVY0"].setText("%.6f" % sigVelSim)
        self.sim["cdiSigVZ0"].setText("%.6f" % sigVelSim)
        self.sim["cdiAX0"].setText("0.")
        self.sim["cdiAY0"].setText("0.")
        self.sim["cdiAZ0"].setText("0.")
        self.sim["cdiSigAX0"].setText("%.6f" % sigAccSim)
        self.sim["cdiSigAY0"].setText("%.6f" % sigAccSim)
        self.sim["cdiSigAZ0"].setText("%.6f" % sigAccSim)

        # Viewer options.
        keys = [("vwrVelLgh", "0"), ("vwrVelALR", "0.3"),
                ("vwrAccLgh", "0"), ("vwrAccALR", "0.3")]
        for key in keys:
            self.slt[key[0]].setText(key[1])
            self.msr[key[0]].setText(key[1])
            self.sim[key[0]].setText(key[1])

    def onUpDownExampleClicked(self, sigPosGPS, sigVelGPS, sigAccSensor):
        """Callback on click: up-down example radio button"""

        # Flight path equation: parameters.
        self.slt["fpeAx"].setText("10000.")
        self.slt["fpeAy"].setText("10000.")
        self.slt["fpeTx"].setText("36000.")
        self.slt["fpeTy"].setText("36000.")
        self.slt["fpePhix"].setText("270.")
        self.slt["fpePhiy"].setText("0.")
        self.slt["fpeTiZi"].setText("100 10., 3500 10., 3600 0.")
        self.slt["cdiX0"].setText("0.")
        self.slt["cdiY0"].setText("0.")
        self.slt["cdiZ0"].setText("0.")
        self.slt["cdfTf"].setText("3600.")

        # Evaluate sigma: simulation sigma (less trusted) > GPS sigma (more trusted).
        sigPosSim = 2.*sigPosGPS
        sigVelSim = 2.*sigVelGPS
        sigAccSim = 2.*sigAccSensor

        # Simulation: parameters.
        self.sim["prmDt"].setText("5.")
        self.sim["prmExpOrd"].setText("3")
        self.sim["prmProNseSig"].setText("0.1")
        self.sim["cdiX0"].setText("0.5")
        self.sim["cdiY0"].setText("0.5")
        self.sim["cdiZ0"].setText("0.")
        self.sim["cdiSigX0"].setText("%.6f" % sigPosSim)
        self.sim["cdiSigY0"].setText("%.6f" % sigPosSim)
        self.sim["cdiSigZ0"].setText("%.6f" % sigPosSim)
        self.sim["cdiVX0"].setText("2.")
        self.sim["cdiVY0"].setText("2.")
        self.sim["cdiVZ0"].setText("0.5")
        self.sim["cdiSigVX0"].setText("%.6f" % sigVelSim)
        self.sim["cdiSigVY0"].setText("%.6f" % sigVelSim)
        self.sim["cdiSigVZ0"].setText("%.6f" % sigVelSim)
        self.sim["cdiAX0"].setText("0.")
        self.sim["cdiAY0"].setText("0.")
        self.sim["cdiAZ0"].setText("0.")
        self.sim["cdiSigAX0"].setText("%.6f" % sigAccSim)
        self.sim["cdiSigAY0"].setText("%.6f" % sigAccSim)
        self.sim["cdiSigAZ0"].setText("%.6f" % sigAccSim)

        # Viewer options.
        keys = [("vwrVelLgh", "0"), ("vwrVelALR", "0.3"),
                ("vwrAccLgh", "0"), ("vwrAccALR", "0.3")]
        for key in keys:
            self.slt[key[0]].setText(key[1])
            self.msr[key[0]].setText(key[1])
            self.sim[key[0]].setText(key[1])

    def onZigZagExampleClicked(self, sigPosGPS, sigVelGPS, sigAccSensor):
        """Callback on click: zig-zag example radio button"""

        # Flight path equation: parameters.
        self.slt["fpeAx"].setText("10000.")
        self.slt["fpeAy"].setText("10000.")
        self.slt["fpeTx"].setText("36000.")
        self.slt["fpeTy"].setText("1800.")
        self.slt["fpePhix"].setText("270.")
        self.slt["fpePhiy"].setText("0.")
        self.slt["fpeTiZi"].setText("3600 10000.")
        self.slt["cdiX0"].setText("0.")
        self.slt["cdiY0"].setText("0.")
        self.slt["cdiZ0"].setText("0.")
        self.slt["cdfTf"].setText("3600.")

        # Evaluate sigma: simulation sigma (less trusted) > GPS sigma (more trusted).
        sigPosSim = 2.*sigPosGPS
        sigVelSim = 2.*sigVelGPS
        sigAccSim = 2.*sigAccSensor

        # Simulation: parameters.
        self.sim["prmDt"].setText("5.")
        self.sim["prmExpOrd"].setText("3")
        self.sim["prmProNseSig"].setText("0.1")
        self.sim["cdiX0"].setText("0.5")
        self.sim["cdiY0"].setText("0.5")
        self.sim["cdiZ0"].setText("0.")
        self.sim["cdiSigX0"].setText("%.6f" % sigPosSim)
        self.sim["cdiSigY0"].setText("%.6f" % sigPosSim)
        self.sim["cdiSigZ0"].setText("%.6f" % sigPosSim)
        self.sim["cdiVX0"].setText("2.")
        self.sim["cdiVY0"].setText("35.")
        self.sim["cdiVZ0"].setText("2.")
        self.sim["cdiSigVX0"].setText("%.6f" % sigVelSim)
        self.sim["cdiSigVY0"].setText("%.6f" % sigVelSim)
        self.sim["cdiSigVZ0"].setText("%.6f" % sigVelSim)
        self.sim["cdiAX0"].setText("0.")
        self.sim["cdiAY0"].setText("0.")
        self.sim["cdiAZ0"].setText("0.")
        self.sim["cdiSigAX0"].setText("%.6f" % sigAccSim)
        self.sim["cdiSigAY0"].setText("%.6f" % sigAccSim)
        self.sim["cdiSigAZ0"].setText("%.6f" % sigAccSim)

        # Viewer options.
        keys = [("vwrVelLgh", "0"), ("vwrVelALR", "0.3"),
                ("vwrAccLgh", "0"), ("vwrAccALR", "0.3")]
        for key in keys:
            self.slt[key[0]].setText(key[1])
            self.msr[key[0]].setText(key[1])
            self.sim[key[0]].setText(key[1])

    def onRoundTripExampleClicked(self, sigPosGPS, sigVelGPS, sigAccSensor):
        """Callback on click: round trip example radio button"""

        # Flight path equation: parameters.
        self.slt["fpeAx"].setText("10000.")
        self.slt["fpeAy"].setText("20000.")
        self.slt["fpeTx"].setText("3600.")
        self.slt["fpeTy"].setText("3600.")
        self.slt["fpePhix"].setText("0.")
        self.slt["fpePhiy"].setText("0.")
        self.slt["fpeTiZi"].setText("100 10, 3500 10, 3600 0")
        self.slt["cdiX0"].setText("0.")
        self.slt["cdiY0"].setText("0.")
        self.slt["cdiZ0"].setText("0.")
        self.slt["cdfTf"].setText("3600.")

        # Evaluate sigma: simulation sigma (less trusted) > GPS sigma (more trusted).
        sigPosSim = 2.*sigPosGPS
        sigVelSim = 2.*sigVelGPS
        sigAccSim = 2.*sigAccSensor

        # Simulation: parameters.
        self.sim["prmDt"].setText("5.")
        self.sim["prmExpOrd"].setText("3")
        self.sim["prmProNseSig"].setText("0.1")
        self.sim["cdiX0"].setText("0.5")
        self.sim["cdiY0"].setText("0.5")
        self.sim["cdiZ0"].setText("0.")
        self.sim["cdiSigX0"].setText("%.6f" % sigPosSim)
        self.sim["cdiSigY0"].setText("%.6f" % sigPosSim)
        self.sim["cdiSigZ0"].setText("%.6f" % sigPosSim)
        self.sim["cdiVX0"].setText("0.5")
        self.sim["cdiVY0"].setText("35.")
        self.sim["cdiVZ0"].setText("0.5")
        self.sim["cdiSigVX0"].setText("%.6f" % sigVelSim)
        self.sim["cdiSigVY0"].setText("%.6f" % sigVelSim)
        self.sim["cdiSigVZ0"].setText("%.6f" % sigVelSim)
        self.sim["cdiAX0"].setText("0.")
        self.sim["cdiAY0"].setText("0.")
        self.sim["cdiAZ0"].setText("0.")
        self.sim["cdiSigAX0"].setText("%.6f" % sigAccSim)
        self.sim["cdiSigAY0"].setText("%.6f" % sigAccSim)
        self.sim["cdiSigAZ0"].setText("%.6f" % sigAccSim)

        # Viewer options.
        keys = [("vwrVelLgh", "0"), ("vwrVelALR", "0.1"),
                ("vwrAccLgh", "0"), ("vwrAccALR", "0.1")]
        for key in keys:
            self.slt[key[0]].setText(key[1])
            self.msr[key[0]].setText(key[1])
            self.sim[key[0]].setText(key[1])

    def onLoopingExampleClicked(self, sigPosGPS, sigVelGPS, sigAccSensor):
        """Callback on click: looping example radio button"""

        # Flight path equation: parameters.
        self.slt["fpeAx"].setText("100.")
        self.slt["fpeAy"].setText("100.")
        self.slt["fpeTx"].setText("3650.")
        self.slt["fpeTy"].setText("3550.")
        self.slt["fpePhix"].setText("270.")
        self.slt["fpePhiy"].setText("0.")
        self.slt["fpeTiZi"].setText("300 5010., 2500 5150, 3400 5010., 3600 5000.")
        self.slt["cdiX0"].setText("0.")
        self.slt["cdiY0"].setText("0.")
        self.slt["cdiZ0"].setText("5000.")
        self.slt["cdfTf"].setText("3600.")

        # Evaluate sigma: simulation sigma (less trusted) > GPS sigma (more trusted).
        sigPosSim = 2.*sigPosGPS
        sigVelSim = 2.*sigVelGPS
        sigAccSim = 2.*sigAccSensor

        # Simulation: parameters.
        self.sim["prmDt"].setText("5.")
        self.sim["prmExpOrd"].setText("3")
        self.sim["prmProNseSig"].setText("0.1")
        self.sim["cdiX0"].setText("0.5")
        self.sim["cdiY0"].setText("0.5")
        self.sim["cdiZ0"].setText("5000.")
        self.sim["cdiSigX0"].setText("%.6f" % sigPosSim)
        self.sim["cdiSigY0"].setText("%.6f" % sigPosSim)
        self.sim["cdiSigZ0"].setText("%.6f" % sigPosSim)
        self.sim["cdiVX0"].setText("0.")
        self.sim["cdiVY0"].setText("0.")
        self.sim["cdiVZ0"].setText("0.")
        self.sim["cdiSigVX0"].setText("%.6f" % sigVelSim)
        self.sim["cdiSigVY0"].setText("%.6f" % sigVelSim)
        self.sim["cdiSigVZ0"].setText("%.6f" % sigVelSim)
        self.sim["cdiAX0"].setText("0.")
        self.sim["cdiAY0"].setText("0.")
        self.sim["cdiAZ0"].setText("0.")
        self.sim["cdiSigAX0"].setText("%.6f" % sigAccSim)
        self.sim["cdiSigAY0"].setText("%.6f" % sigAccSim)
        self.sim["cdiSigAZ0"].setText("%.6f" % sigAccSim)

        # Viewer options.
        keys = [("vwrVelLgh", "0"), ("vwrVelALR", "0.3"),
                ("vwrAccLgh", "0"), ("vwrAccALR", "0.3")]
        for key in keys:
            self.slt[key[0]].setText(key[1])
            self.msr[key[0]].setText(key[1])
            self.sim[key[0]].setText(key[1])

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
        for idx in range(self.msr["msrLst"].count()):
            txt = self.msr["msrLst"].item(idx).text()
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
        if float(self.sim["prmProNseSig"].text()) < 0.:
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
        ctlThfTkoK = float(self.sim["ctlThfTkoK"].text())
        ctlThfFlgK = float(self.sim["ctlThfFlgK"].text())
        ctlThfLdgK = float(self.sim["ctlThfLdgK"].text())
        if ctlThfTkoK < 0 or ctlThfFlgK < 0 or ctlThfLdgK < 0:
            self.throwError(eId, "throttle coefficients must be superior than 0.")
            return False
        cdfTf = float(self.slt["cdfTf"].text())
        ctlThfTkoDt = float(self.sim["ctlThfTkoDt"].text())
        ctlThfLdgDt = float(self.sim["ctlThfLdgDt"].text())
        if not 0. < ctlThfTkoDt < cdfTf or not 0. < ctlThfLdgDt < cdfTf:
            self.throwError(eId, "throttle time slot must belong to [0., t<sub>f</sub>].")
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
        matP[0, 0] = np.power(sim["cdiSigX0"], 2)
        matP[1, 1] = np.power(sim["cdiSigVX0"], 2)
        matP[2, 2] = np.power(sim["cdiSigAX0"], 2)
        matP[3, 3] = np.power(sim["cdiSigY0"], 2)
        matP[4, 4] = np.power(sim["cdiSigVY0"], 2)
        matP[5, 5] = np.power(sim["cdiSigAY0"], 2)
        matP[6, 6] = np.power(sim["cdiSigZ0"], 2)
        matP[7, 7] = np.power(sim["cdiSigVZ0"], 2)
        matP[8, 8] = np.power(sim["cdiSigAZ0"], 2)

        return matP

    def computeControlLaw(self, states, time, sim, save=True):
        """Compute control law"""

        # Compute throttle force.
        velNow = np.array([states[1, 0], states[4, 0], states[7, 0]]) # Velocity.
        thfX, thfY, thfZ = self.computeThrottleForce(velNow, time)

        # Compute control law: get roll, pitch, yaw corrections.
        accNow = np.array([states[2, 0], states[5, 0], states[8, 0]]) # Acceleration.
        accNxt = self.computeRoll(velNow, accNow, sim, save)
        accNxt = self.computePitch(velNow, accNxt, sim, save)
        accNxt = self.computeYaw(velNow, accNxt, sim, save)

        # Compute control law.
        fomX = accNxt[0]-accNow[0]
        fomY = accNxt[1]-accNow[1]
        fomZ = accNxt[2]-accNow[2]
        matU = self.computeControl((thfX+fomX, thfY+fomY, thfZ+fomZ), sim, save)

        # Save F/m to compute d(F/m)/dt next time.
        if save:
            sim["ctlOldFoMX"] = fomX
            sim["ctlOldFoMY"] = fomY
            sim["ctlOldFoMZ"] = fomZ

        return matU

    def computeThrottleForce(self, velNow, time):
        """Compute throttle force"""

        # Get throttle parameters.
        ctlThfK = 0.
        cdfTf = float(self.slt["cdfTf"].text())
        ctlThfTkoDt = float(self.sim["ctlThfTkoDt"].text())
        ctlThfLdgDt = float(self.sim["ctlThfLdgDt"].text())
        if 0. < time <= ctlThfTkoDt:
            ctlThfK = float(self.sim["ctlThfTkoK"].text())
        elif cdfTf-ctlThfLdgDt < time <= cdfTf:
            ctlThfK = float(self.sim["ctlThfLdgK"].text())
        else:
            ctlThfK = float(self.sim["ctlThfFlgK"].text())

        # Compute throttle force F = (k/m)*V.
        prmM = float(self.sim["prmM"].text())
        thrForce = ctlThfK/prmM*velNow

        return thrForce[0], thrForce[1], thrForce[2]

    def computeRoll(self, velNow, accNow, sim, save):
        """Compute control law: roll"""

        # Compute roll around X axis.
        prmDt = float(self.sim["prmDt"].text())
        velNxt = velNow+accNow*prmDt # New velocity.
        proj = np.array([0., 1., 1.]) # Projection in YZ plane.
        roll = self.getAngle(velNow, velNxt, proj)

        # Save control law hidden variables.
        if save:
            sim["simCLV"]["roll"].append(roll)

        # Control roll.
        ctlRolMax = float(self.sim["ctlRolMax"].text())
        accNxt, rollTgt = accNow, roll
        while np.abs(rollTgt) > ctlRolMax:
            accNxt = accNxt*0.95 # Decrease acceleration by 5%.
            velNxt = velNow+accNxt*prmDt # New velocity.
            rollTgt = self.getAngle(velNow, velNxt, proj)

        return accNxt

    def computePitch(self, velNow, accNow, sim, save):
        """Compute control law: pitch"""

        # Compute pitch around Y axis.
        prmDt = float(self.sim["prmDt"].text())
        velNxt = velNow+accNow*prmDt # New velocity.
        proj = np.array([1., 0., 1.]) # Projection in XZ plane.
        pitch = self.getAngle(velNow, velNxt, proj)

        # Save control law hidden variables.
        if save:
            sim["simCLV"]["pitch"].append(pitch)

        # Control pitch.
        ctlPtcMax = float(self.sim["ctlPtcMax"].text())
        accNxt, pitchTgt = accNow, pitch
        while np.abs(pitchTgt) > ctlPtcMax:
            accNxt = accNxt*0.95 # Decrease acceleration by 5%.
            velNxt = velNow+accNxt*prmDt # New velocity.
            pitchTgt = self.getAngle(velNow, velNxt, proj)

        return accNxt

    def computeYaw(self, velNow, accNow, sim, save):
        """Compute control law: yaw"""

        # Compute yaw around Z axis.
        prmDt = float(self.sim["prmDt"].text())
        velNxt = velNow+accNow*prmDt # New velocity.
        proj = np.array([1., 1., 0.]) # Projection in XY plane.
        yaw = self.getAngle(velNow, velNxt, proj)

        # Save control law hidden variables.
        if save:
            sim["simCLV"]["yaw"].append(yaw)

        # Control yaw.
        ctlYawMax = float(self.sim["ctlYawMax"].text())
        accNxt, yawTgt = accNow, yaw
        while np.abs(yawTgt) > ctlYawMax:
            accNxt = accNxt*0.95 # Decrease acceleration by 5%.
            velNxt = velNow+accNxt*prmDt # New velocity.
            yawTgt = self.getAngle(velNow, velNxt, proj)

        return accNxt

    @staticmethod
    def getAngle(velNow, velNxt, proj):
        """Get angle between 2 vectors"""

        theta = 0.
        velNowProj, velNxtProj = velNow*proj, velNxt*proj
        normCoef = npl.norm(velNowProj)*npl.norm(velNxtProj)
        if np.abs(normCoef) > 1.e-6:
            theta = np.arccos(np.dot(velNowProj, velNxtProj)/normCoef)
            theta = theta*(180./np.pi) # Yaw angle in degrees.

        return theta

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
            sim["simCLV"]["FoM"]["X"].append(matU[1, 0])
            sim["simCLV"]["FoM"]["Y"].append(matU[4, 0])
            sim["simCLV"]["FoM"]["Z"].append(matU[7, 0])
            sim["simCLV"]["d(FoM)/dt"]["X"].append(matU[2, 0])
            sim["simCLV"]["d(FoM)/dt"]["Y"].append(matU[5, 0])
            sim["simCLV"]["d(FoM)/dt"]["Z"].append(matU[8, 0])

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
                prbGUI = example.createPrbGUI()
                layCtr.addWidget(prbGUI)
                sltGUI = example.createSltGUI()
                layCtr.addWidget(sltGUI)
                msrGUI = example.createMsrGUI()
                layCtr.addWidget(msrGUI)
                simGUI = example.createSimGUI()
                layCtr.addWidget(simGUI)
                expGUI = example.createExpGUI()
                layCtr.addWidget(expGUI)
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

        # Add vertical and horizontal scroll bars to GUI.
        scrollWidget = QScrollArea()
        scrollWidget.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scrollWidget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scrollWidget.setWidget(guiCtr)
        scrollWidget.setWidgetResizable(True)
        self.setCentralWidget(scrollWidget)

    def addUpdateVwrBtn(self):
        """Add button to update the viewer"""

        # Add button to update the viewer.
        updateBtn = QPushButton("Update", self)
        updateBtn.clicked.connect(self.onUpdateVwrBtnClick)

        return updateBtn

    def onUpdateVwrBtnClick(self):
        """Callback on update viewer button click"""

        # Update the view.
        print("********** Update : begin **********")
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
        print("********** Update : end **********")

# Main program.
if __name__ == "__main__":
    # Check for python3.
    assert sys.version_info.major == 3, "This script is a python3 script."

    # Create application and controls GUI.
    app = QApplication(sys.argv)
    ctrWin = controllerGUI()
    ctrWin.showMaximized()

    # End main program.
    sys.exit(app.exec_())
