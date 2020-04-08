#!/usr/bin/env python3

"""Kalman filter MVC (Model-View-Controller)"""

import sys
import math
import random
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
        self.sim = {"ctlHV": {}, "time": []}
        self.msr = {}
        self.example = example
        self.solved = False
        self.states = {}
        self.outputs = {}
        self.mat = {}
        self.clear()

    def clear(self):
        """Clear previous results"""

        # Clear previous time.
        self.sim["time"] = []

        # Clear previous results.
        self.solved = False
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

    def setUpSimPrm(self, sim, cdfTf):
        """Setup solver: simulation parameters"""

        # Set up solver parameters.
        for key in sim:
            if key.find("prm") == 0 or key.find("cdi") == 0:
                self.sim[key] = float(sim[key].text())
        self.sim["cdfTf"] = float(cdfTf)

    def setUpMsrPrm(self, msr):
        """Setup solver: measurement parameters"""

        # Set up solver measurements.
        self.msr = msr

    def setLTI(self, matA, matB, matC, matD):
        """Set Linear Time Invariant matrices"""

        # Set matirces.
        self.mat["A"] = matA
        self.mat["B"] = matB
        self.mat["C"] = matC
        self.mat["D"] = matD
        if self.sim["prmVrb"] >= 2:
            print("Linear Time Invariant system:")
            self.printMat("A", self.mat["A"])
            if self.mat["B"] is not None:
                self.printMat("B", self.mat["B"])
            self.printMat("C", self.mat["C"])
            if self.mat["D"] is not None:
                self.printMat("D", self.mat["D"])

    def solve(self):
        """Solve based on Kalman filter"""

        # Don't solve if we have already a solution.
        if self.solved:
            return

        # Initialize states.
        time = 0.
        states = self.example.initStates(self.sim)
        matU = self.example.computeControlLaw(states, self.sim)
        outputs = self.computeOutputs(states, matU)
        if self.sim["prmVrb"] >= 1:
            print("Initialisation:")
            self.printMat("Predictor - X", np.transpose(states))
            self.printMat("Predictor - Y", np.transpose(outputs))

        # Save initial states and outputs.
        self.sim["time"].append(time)
        self.example.saveStatesOutputs(states, self.states, outputs, self.outputs)

        # Solve.
        prmDt, prmTf = self.sim["prmDt"], self.sim["cdfTf"]
        while time < prmTf:
            # Increase time.
            time = time+prmDt
            if time > prmTf:
                time = prmTf
            self.sim["time"].append(time)
            if self.sim["prmVrb"] >= 1:
                print("Iteration: time %.3f" % time)

            # Solve with Kalman filter.
            self.predictor()

        # Mark solver as solved.
        self.solved = True

    def predictor(self):
        """Solve predictor equation"""

        # Compute F.
        prmN = self.example.getLTISystemSize()
        matF = np.identity(prmN, dtype=float)
        taylorExpLTM = 0.
        for idx in range(1, int(self.sim["prmExpOrd"])+1):
            fac = np.math.factorial(idx)
            taylorExp = npl.matrix_power(self.mat["A"]*self.sim["prmDt"], idx)/fac
            taylorExpLTM = np.amax(np.abs(taylorExp))
            matF = matF + taylorExp
        if self.sim["prmVrb"] >= 2:
            msg = "Predictor - F (last term magnitude of taylor expansion %.6f)" % taylorExpLTM
            self.printMat(msg, matF)

        # Compute G.
        matG = None
        if self.mat["B"] is not None:
            matG = np.dot(self.sim["prmDt"]*matF, self.mat["B"])
            if self.sim["prmVrb"] >= 2:
                self.printMat("Predictor - G", matG)

        # Compute process noise.
        states = self.getLastStates()
        matW = self.getProcessNoise(states)
        if self.sim["prmVrb"] >= 1:
            self.printMat("Predictor - W", np.transpose(matW))

        # Compute control law.
        matU = self.example.computeControlLaw(states, self.sim)
        if self.sim["prmVrb"] >= 1:
            self.printMat("Predictor - U", np.transpose(matU))

        # Predictor equation: x_{n+1} = F*x_{n} + G*u_{n} + w_{n}.
        states = np.dot(matF, states)
        if matG is not None:
            states = states + np.dot(matG, matU)
        states = states + matW
        assert states.shape == (prmN, 1), "states: bad dimension"
        if self.sim["prmVrb"] >= 1:
            self.printMat("Predictor - X", np.transpose(states))

        # Outputs equation: y_{n+1} = C*x_{n} + D*u_{n}.
        outputs = self.computeOutputs(states, matU)
        assert outputs.shape == (prmN, 1), "outputs: bad dimension"
        if self.sim["prmVrb"] >= 1:
            self.printMat("Predictor - Y", np.transpose(outputs))

        # Save states and outputs.
        self.example.saveStatesOutputs(states, self.states, outputs, self.outputs)

    def getLastStates(self):
        """Get last states"""

        # Get last states.
        prmN = self.example.getLTISystemSize()
        states = np.zeros((prmN, 1), dtype=float)
        keys = self.example.getStateKeys()
        for idx, key in enumerate(keys):
            lastIdx = len(self.states[key])-1
            states[idx, 0] = self.states[key][lastIdx]

        return states

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
            outputs = outputs + np.dot(self.mat["D"], matU)

        return outputs

    @staticmethod
    def printMat(msg, mat, indent=1, fmt=".6f"):
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
        self.slt = {"sltId": "", "time": [], "eqn": {}}
        self.msr = {"sltId": "", "msrId": ""}
        self.sim = {"sltId": "", "msrId": "", "simId": ""}
        self.vwr = {"2D": {"tzp": None, "ctlHV": None, "simOV": None}, "3D": None}
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
        if vwrId in ("all", "simOV"):
            if self.vwr["2D"]["simOV"]:
                for idx in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
                    axis = self.vwr["2D"]["simOV"].getAxis(idx)
                    axis.set_xlabel("t")
                    axis.cla()
                self.vwr["2D"]["simOV"].draw()
        if vwrId in ("all", "ctlHV"):
            if self.vwr["2D"]["ctlHV"]:
                for idx in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
                    axis = self.vwr["2D"]["ctlHV"].getAxis(idx)
                    axis.set_xlabel("t")
                    axis.cla()
                self.vwr["2D"]["ctlHV"].draw()
        if vwrId in ("all", "3D"):
            axis = self.vwr["3D"].getAxis()
            axis.cla()
            axis.set_xlabel("x")
            axis.set_ylabel("y")
            axis.set_zlabel("z")
            self.vwr["3D"].draw()

    def updateViewerSlt(self):
        """Update viewer: solution"""

        # Update V0/A0 indicators.
        self.slt["cdiVX0"].setText("N.A.")
        self.slt["cdiVY0"].setText("N.A.")
        self.slt["cdiVZ0"].setText("N.A.")
        self.slt["cdiAX0"].setText("N.A.")
        self.slt["cdiAY0"].setText("N.A.")
        self.slt["cdiAZ0"].setText("N.A.")

        # Plot only if checked.
        if not self.vwr["ckbSlt"].isChecked():
            return

        # Plot Z.
        if self.vwr["2D"]["tzp"] and not self.vwr["2D"]["tzp"].closed:
            self.onPltTZPBtnClick()

        # Time.
        prmTf = float(self.slt["cdfTf"].text())
        vwrNbPt = float(self.slt["vwrNbPt"].text())
        eqnT = np.linspace(0., prmTf, vwrNbPt)
        self.slt["time"] = eqnT # Save time.

        # Plot solution.
        eqnX, eqnY, eqnZ = self.updateViewerSltX(eqnT)
        self.updateViewerSltV(eqnT, eqnX, eqnY, eqnZ)
        self.updateViewerSltA(eqnT, eqnX, eqnY, eqnZ)

        # Track solution features.
        self.slt["sltId"] = self.getSltId()

    def updateViewerSltX(self, eqnT):
        """Update viewer: plot displacement of the solution"""

        # Plot solution: displacement.
        eqnX, eqnY, eqnZ = self.getDisplEquations(eqnT)
        vwrLnWd = float(self.slt["vwrLnWd"].text())
        if vwrLnWd == 0.:
            return eqnX, eqnY, eqnZ
        vwrPosMks = float(self.slt["vwrPosMks"].text())
        clr = (0., 0., 1.) # Blue.
        axis = self.vwr["3D"].getAxis()
        axis.plot3D(eqnX, eqnY, eqnZ, lw=vwrLnWd, color=clr,
                    label="flight path: x", marker="o", ms=vwrPosMks)

        # Save equations.
        self.slt["eqn"]["X"] = eqnX
        self.slt["eqn"]["Y"] = eqnY
        self.slt["eqn"]["Z"] = eqnZ

        return eqnX, eqnY, eqnZ

    def updateViewerSltV(self, eqnT, eqnX, eqnY, eqnZ):
        """Update viewer: plot velocity of the solution"""

        # Update V0 indicators.
        eqnVX, eqnVY, eqnVZ = self.getVelocEquations(eqnT)
        self.slt["cdiVX0"].setText("%.3f" % eqnVX[0])
        self.slt["cdiVY0"].setText("%.3f" % eqnVY[0])
        self.slt["cdiVZ0"].setText("%.3f" % eqnVZ[0])

        # Plot solution: velocity.
        clr = (0., 0.75, 1.) # Skyblue.
        vwrVelLgh = float(self.slt["vwrVelLgh"].text())
        if vwrVelLgh == 0.:
            return
        vwrVelNrm = self.slt["vwrVelNrm"].isChecked()
        axis = self.vwr["3D"].getAxis()
        axis.quiver3D(eqnX, eqnY, eqnZ, eqnVX, eqnVY, eqnVZ, color=clr,
                      length=vwrVelLgh, normalize=vwrVelNrm, label="flight path: v")

        # Save equations.
        self.slt["eqn"]["VX"] = eqnVX
        self.slt["eqn"]["VY"] = eqnVY
        self.slt["eqn"]["VZ"] = eqnVZ

    def updateViewerSltA(self, eqnT, eqnX, eqnY, eqnZ):
        """Update viewer: plot acceleration of the solution"""

        # Update A0 indicators.
        eqnAX, eqnAY, eqnAZ = self.getAccelEquations(eqnT)
        self.slt["cdiAX0"].setText("%.3f" % eqnAX[0])
        self.slt["cdiAY0"].setText("%.3f" % eqnAY[0])
        self.slt["cdiAZ0"].setText("%.3f" % eqnAZ[0])

        # Plot solution: acceleration.
        clr = (0.25, 0., 0.5) # Indigo.
        vwrAccLgh = float(self.slt["vwrAccLgh"].text())
        if vwrAccLgh == 0.:
            return
        vwrAccNrm = self.slt["vwrAccNrm"].isChecked()
        axis = self.vwr["3D"].getAxis()
        axis.quiver3D(eqnX, eqnY, eqnZ, eqnAX, eqnAY, eqnAZ, colors=clr,
                      length=vwrAccLgh, normalize=vwrAccNrm, label="flight path: a")

        # Save equations.
        self.slt["eqn"]["AX"] = eqnAX
        self.slt["eqn"]["AY"] = eqnAY
        self.slt["eqn"]["AZ"] = eqnAZ

    def getSltId(self):
        """Get solution identity (track solution features)"""

        # Get solution identity.
        sltId = ""
        for key in self.slt:
            if key.find("fpe") == 0 or key.find("cd") == 0:
                sltId += ":"+self.slt[key].text()

        return sltId

    def updateViewerMsr(self):
        """Update viewer: measurements"""

        # Clean all measurements if analytic solution has changed.
        if self.msr["sltId"] != self.slt["sltId"]:
            self.msr["sltId"] = self.slt["sltId"]
            self.msr["datMsr"].clear()

        # Plot only if checked.
        if not self.vwr["ckbMsr"].isChecked():
            return

        # Clean removed measurement data.
        delKeys = []
        for key in self.msr["datMsr"]:
            delKey = True
            for idx in range(self.msr["lstMsr"].count()):
                txt = self.msr["lstMsr"].item(idx).text()
                if txt == key:
                    delKey = False
            if delKey:
                delKeys.append(key)
        for key in delKeys:
            del self.msr["datMsr"][key]

        # Update viewer.
        for idx in range(self.msr["lstMsr"].count()):
            # Skip unused items.
            txt = self.msr["lstMsr"].item(idx).text()
            if txt == "":
                continue

            # Create or retrieve measure data.
            msrData = None
            if txt not in self.msr["datMsr"]:
                self.msr["datMsr"][txt] = self.getMsrData(txt)
            msrData = self.msr["datMsr"][txt]

            # View measurement.
            self.viewMsrData(msrData)

        # Track measurement features.
        self.msr["msrId"] = self.getMsrId()

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

        # Data.
        eqnX, eqnY, eqnZ = self.getDisplEquations(eqnT)
        prmSigma = float(txt.split(";")[4].split()[1])
        msrData["posX"] = self.addUncertainty(eqnX, prmSigma)
        msrData["posY"] = self.addUncertainty(eqnY, prmSigma)
        msrData["posZ"] = self.addUncertainty(eqnZ, prmSigma)

    def getMsrDataV(self, txt, msrData):
        """Get measure data: velocity"""

        # Time.
        prmT0 = float(txt.split(";")[1].split()[1])
        prmTf = float(txt.split(";")[2].split()[1])
        prmDt = float(txt.split(";")[3].split()[1])
        prmNbPt = (prmTf-prmT0)/prmDt
        eqnT = np.linspace(prmT0, prmTf, prmNbPt)

        # Data.
        eqnX, eqnY, eqnZ = self.getDisplEquations(eqnT)
        msrData["posX"] = eqnX
        msrData["posY"] = eqnY
        msrData["posZ"] = eqnZ
        eqnVX, eqnVY, eqnVZ = self.getVelocEquations(eqnT)
        prmSigma = float(txt.split(";")[4].split()[1])
        msrData["eqnVX"] = self.addUncertainty(eqnVX, prmSigma)
        msrData["eqnVY"] = self.addUncertainty(eqnVY, prmSigma)
        msrData["eqnVZ"] = self.addUncertainty(eqnVZ, prmSigma)

    def getMsrDataA(self, txt, msrData):
        """Get measure data: acceleration"""

        # Time.
        prmT0 = float(txt.split(";")[1].split()[1])
        prmTf = float(txt.split(";")[2].split()[1])
        prmDt = float(txt.split(";")[3].split()[1])
        prmNbPt = (prmTf-prmT0)/prmDt
        eqnT = np.linspace(prmT0, prmTf, prmNbPt)

        # Data.
        eqnX, eqnY, eqnZ = self.getDisplEquations(eqnT)
        msrData["posX"] = eqnX
        msrData["posY"] = eqnY
        msrData["posZ"] = eqnZ
        eqnAX, eqnAY, eqnAZ = self.getAccelEquations(eqnT)
        prmSigma = float(txt.split(";")[4].split()[1])
        msrData["eqnAX"] = self.addUncertainty(eqnAX, prmSigma)
        msrData["eqnAY"] = self.addUncertainty(eqnAY, prmSigma)
        msrData["eqnAZ"] = self.addUncertainty(eqnAZ, prmSigma)

    @staticmethod
    def addUncertainty(eqn, prmSigma):
        """Add uncertainty"""

        # Add uncertainty to data.
        eqnSig = eqn
        for idx, val in enumerate(eqn):
            alpha = random.uniform(-1., 1.)
            eqnSig[idx] = val + alpha*prmSigma

        return eqnSig

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
        posX = msrData["posX"]
        posY = msrData["posY"]
        posZ = msrData["posZ"]
        vwrPosMks = float(self.msr["vwrPosMks"].text())
        if vwrPosMks == 0.:
            return
        axis = self.vwr["3D"].getAxis()
        axis.scatter3D(posX, posY, posZ, c="r", marker="^", alpha=1, s=vwrPosMks,
                       label="measure: x")

    def viewMsrDataV(self, msrData):
        """View measure data: velocity"""

        # View measure data: velocity.
        posX = msrData["posX"]
        posY = msrData["posY"]
        posZ = msrData["posZ"]
        eqnVX = msrData["eqnVX"]
        eqnVY = msrData["eqnVY"]
        eqnVZ = msrData["eqnVZ"]
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
        posX = msrData["posX"]
        posY = msrData["posY"]
        posZ = msrData["posZ"]
        eqnAX = msrData["eqnAX"]
        eqnAY = msrData["eqnAY"]
        eqnAZ = msrData["eqnAZ"]
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

        # Clean solver results if analytic solution or measurements have changed.
        if self.sim["sltId"] != self.slt["sltId"]:
            self.sim["sltId"] = self.slt["sltId"]
            self.kfm.clear()
        if self.sim["msrId"] != self.msr["msrId"]:
            self.sim["msrId"] = self.msr["msrId"]
            self.kfm.clear()
        if self.sim["simId"] != self.getSimId():
            self.sim["simId"] = self.getSimId()
            self.kfm.clear()

        # Plot only if checked.
        if not self.vwr["ckbSim"].isChecked():
            return

        # Solve based on Kalman filter.
        self.kfm.setUpSimPrm(self.sim, self.slt["cdfTf"].text())
        self.kfm.setUpMsrPrm(self.msr["datMsr"])
        matA, matB, matC, matD = self.getLTISystem()
        self.kfm.setLTI(matA, matB, matC, matD)
        self.kfm.solve()

        # Plot solver results.
        self.updateViewerSimX()
        self.updateViewerSimV()
        self.updateViewerSimA()

        # Plot 2D viewers.
        if self.vwr["2D"]["ctlHV"] and not self.vwr["2D"]["ctlHV"].closed:
            self.onPltCHVBtnClick()
        if self.vwr["2D"]["simOV"] and not self.vwr["2D"]["simOV"].closed:
            self.onPltSOVBtnClick()

    def getSimId(self):
        """Get simulation identity (track simulation features)"""

        # Get simulation identity.
        simId = ""
        for key in self.sim:
            if key.find("prm") == 0 or key.find("cdi") == 0:
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
        self.slt["cdiVX0"] = QLineEdit("N.A.", self.ctrGUI)
        self.slt["cdiVY0"] = QLineEdit("N.A.", self.ctrGUI)
        self.slt["cdiVZ0"] = QLineEdit("N.A.", self.ctrGUI)
        self.slt["cdiAX0"] = QLineEdit("N.A.", self.ctrGUI)
        self.slt["cdiAY0"] = QLineEdit("N.A.", self.ctrGUI)
        self.slt["cdiAZ0"] = QLineEdit("N.A.", self.ctrGUI)
        self.slt["cdfTf"] = QLineEdit("2.", self.ctrGUI)
        self.slt["vwrNbPt"] = QLineEdit("50", self.ctrGUI)
        self.slt["vwrLnWd"] = QLineEdit("1.", self.ctrGUI)
        self.slt["vwrPosMks"] = QLineEdit("5", self.ctrGUI)
        self.slt["vwrVelLgh"] = QLineEdit("0.01", self.ctrGUI)
        self.slt["vwrVelNrm"] = QCheckBox("Normalize", self.ctrGUI)
        self.slt["vwrAccLgh"] = QLineEdit("0.001", self.ctrGUI)
        self.slt["vwrAccNrm"] = QCheckBox("Normalize", self.ctrGUI)

        self.slt["cdiVX0"].setEnabled(False)
        self.slt["cdiVY0"].setEnabled(False)
        self.slt["cdiVZ0"].setEnabled(False)
        self.slt["cdiAX0"].setEnabled(False)
        self.slt["cdiAY0"].setEnabled(False)
        self.slt["cdiAZ0"].setEnabled(False)

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
        anlLay = QHBoxLayout(sltGUI)
        anlLay.addWidget(gpbXi)
        anlLay.addWidget(gpbX0)
        anlLay.addWidget(gpbTf)
        anlLay.addWidget(gpbVwr)
        sltGUI.setLayout(anlLay)

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
        gdlX0.addWidget(self.slt["cdiVX0"], 1, 4)
        title = "V<sub>y</sub>(t = 0) = V<sub>y0</sub>"
        gdlX0.addWidget(QLabel(title, sltGUI), 2, 3, 1, 2)
        gdlX0.addWidget(QLabel("V<sub>y0</sub>", sltGUI), 3, 3)
        gdlX0.addWidget(self.slt["cdiVY0"], 3, 4)
        title = "V<sub>z</sub>(t = 0) = V<sub>z0</sub>"
        gdlX0.addWidget(QLabel(title, sltGUI), 4, 3, 1, 2)
        gdlX0.addWidget(QLabel("V<sub>z0</sub>", sltGUI), 5, 3)
        gdlX0.addWidget(self.slt["cdiVZ0"], 5, 4)
        title = "A<sub>x</sub>(t = 0) = A<sub>x0</sub>"
        gdlX0.addWidget(QLabel(title, sltGUI), 0, 6, 1, 2)
        gdlX0.addWidget(QLabel("A<sub>x0</sub>", sltGUI), 1, 6)
        gdlX0.addWidget(self.slt["cdiAX0"], 1, 7)
        title = "A<sub>y</sub>(t = 0) = A<sub>y0</sub>"
        gdlX0.addWidget(QLabel(title, sltGUI), 2, 6, 1, 2)
        gdlX0.addWidget(QLabel("A<sub>y0</sub>", sltGUI), 3, 6)
        gdlX0.addWidget(self.slt["cdiAY0"], 3, 7)
        title = "A<sub>z</sub>(t = 0) = A<sub>z0</sub>"
        gdlX0.addWidget(QLabel(title, sltGUI), 4, 6, 1, 2)
        gdlX0.addWidget(QLabel("A<sub>z0</sub>", sltGUI), 5, 6)
        gdlX0.addWidget(self.slt["cdiAZ0"], 5, 7)

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
        self.msr["add"] = QComboBox(self.ctrGUI)
        for msr in ["x", "v", "a"]:
            self.msr["add"].addItem(msr)
        self.msr["addT0"] = QLineEdit("0.1", self.ctrGUI)
        finalTime = self.slt["cdfTf"].text()
        self.msr["addTf"] = QLineEdit(str(float(finalTime)*0.9), self.ctrGUI)
        self.msr["addDt"] = QLineEdit("0.1", self.ctrGUI)
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

        # Initialize the measurement list.
        self.onAddMsrBtnClick() # Adding "x" measurement.
        self.msr["add"].setCurrentIndex(1) # Set combo to "v" after adding "x" measurement.
        self.msr["addDt"].setText("0.15")
        self.msr["addSigma"].setText("0.2")
        self.onAddMsrBtnClick() # Adding "v" measurement.
        self.msr["add"].setCurrentIndex(2) # Set combo to "a" after adding "v" measurement.
        self.msr["addDt"].setText("0.2")
        self.msr["addSigma"].setText("0.2")
        self.onAddMsrBtnClick() # Adding "a" measurement.

        # Reset measurement list options.
        self.msr["add"].setCurrentIndex(0)
        self.msr["addDt"].setText("0.1")
        self.msr["addSigma"].setText("0.1")

        return msrGUI

    def fillMsrGUI(self, msrGUI):
        """Fill measurement GUI"""

        # Create group box.
        gpbAdd = self.fillMsrGUIAddMsr(msrGUI)
        gpbLst = self.fillMsrGUILstMsr(msrGUI)
        gpbVwr = self.fillMsrGUIVwrMsr(msrGUI)

        # Set group box layout.
        anlLay = QHBoxLayout(msrGUI)
        anlLay.addWidget(gpbAdd)
        anlLay.addWidget(gpbLst)
        anlLay.addWidget(gpbVwr)
        msrGUI.setLayout(anlLay)

    def fillMsrGUIAddMsr(self, msrGUI):
        """Fill measurement GUI: add measurements"""

        # Create measurement parameters GUI: add measurements.
        gdlAdd = QGridLayout(msrGUI)
        gdlAdd.addWidget(QLabel("Measure:", msrGUI), 0, 0)
        gdlAdd.addWidget(self.msr["add"], 0, 1)
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
        item = "measure "+self.msr["add"].currentText()
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
        self.sim["cdiX0"] = QLineEdit("0.2", self.ctrGUI)
        self.sim["cdiY0"] = QLineEdit("0.2", self.ctrGUI)
        self.sim["cdiZ0"] = QLineEdit("0.2", self.ctrGUI)
        self.sim["cdiSigX0"] = QLineEdit("0.5", self.ctrGUI)
        self.sim["cdiSigY0"] = QLineEdit("0.5", self.ctrGUI)
        self.sim["cdiSigZ0"] = QLineEdit("0.5", self.ctrGUI)
        self.sim["cdiVX0"] = QLineEdit("0.0", self.ctrGUI)
        self.sim["cdiVY0"] = QLineEdit("12.", self.ctrGUI)
        self.sim["cdiVZ0"] = QLineEdit("0.5", self.ctrGUI)
        self.sim["cdiSigVX0"] = QLineEdit("1.", self.ctrGUI)
        self.sim["cdiSigVY0"] = QLineEdit("1.", self.ctrGUI)
        self.sim["cdiSigVZ0"] = QLineEdit("1.", self.ctrGUI)
        self.sim["cdiAX0"] = QLineEdit("-39.", self.ctrGUI)
        self.sim["cdiAY0"] = QLineEdit("0.", self.ctrGUI)
        self.sim["cdiAZ0"] = QLineEdit("19.", self.ctrGUI)
        self.sim["cdiSigAX0"] = QLineEdit("1.", self.ctrGUI)
        self.sim["cdiSigAY0"] = QLineEdit("1.", self.ctrGUI)
        self.sim["cdiSigAZ0"] = QLineEdit("1.", self.ctrGUI)
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

        # Set group box layout.
        anlLay = QHBoxLayout(simGUI)
        anlLay.addWidget(gpbPrm)
        anlLay.addWidget(gpbX0)
        anlLay.addWidget(gpbFCL)
        anlLay.addWidget(gpbVwr)
        simGUI.setLayout(anlLay)

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
        pltSOVBtn = QPushButton("Output variables", simGUI)
        pltSOVBtn.clicked.connect(self.onPltSOVBtnClick)
        gdlPrm.addWidget(pltSOVBtn, 3, 3, 1, 2)

        # Set group box layout.
        gpbPrm = QGroupBox(simGUI)
        gpbPrm.setTitle("Simulation parameters")
        gpbPrm.setAlignment(Qt.AlignHCenter)
        gpbPrm.setLayout(gdlPrm)

        return gpbPrm

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
        if not self.kfm.solved:
            return

        # Plot simulation output variables.
        self.plotSimulationOutputVariablesX()
        self.plotSimulationOutputVariablesV()
        self.plotSimulationOutputVariablesA()

    def plotSimulationOutputVariablesX(self):
        """Plot simulation output variables: X"""

        # Plot simulation output variables.
        axis = self.vwr["2D"]["simOV"].getAxis(0)
        axis.plot(self.slt["time"], self.slt["eqn"]["X"], label="slt - X",
                  marker="o", ms=3, c="b")
        axis.plot(self.kfm.sim["time"], self.kfm.outputs["X"], label="sim - X",
                  marker="o", ms=3, c="g")
        axis.set_xlabel("t")
        axis.set_ylabel("X")
        axis.legend()
        axis = self.vwr["2D"]["simOV"].getAxis(1)
        axis.plot(self.slt["time"], self.slt["eqn"]["Y"], label="slt - Y",
                  marker="o", ms=3, c="b")
        axis.plot(self.kfm.sim["time"], self.kfm.outputs["Y"], label="sim - Y",
                  marker="o", ms=3, c="g")
        axis.set_xlabel("t")
        axis.set_ylabel("Y")
        axis.legend()
        axis = self.vwr["2D"]["simOV"].getAxis(2)
        axis.plot(self.slt["time"], self.slt["eqn"]["Z"], label="slt - Z",
                  marker="o", ms=3, c="b")
        axis.plot(self.kfm.sim["time"], self.kfm.outputs["Z"], label="sim - Z",
                  marker="o", ms=3, c="g")
        axis.set_xlabel("t")
        axis.set_ylabel("Z")
        axis.legend()

    def plotSimulationOutputVariablesV(self):
        """Plot simulation output variables: V"""

        # Plot simulation output variables.
        axis = self.vwr["2D"]["simOV"].getAxis(3)
        axis.plot(self.slt["time"], self.slt["eqn"]["VX"], label="slt - VX",
                  marker="o", ms=3, c="b")
        axis.plot(self.kfm.sim["time"], self.kfm.outputs["VX"], label="sim - VX",
                  marker="o", ms=3, c="g")
        axis.set_xlabel("t")
        axis.set_ylabel("VX")
        axis.legend()
        axis = self.vwr["2D"]["simOV"].getAxis(4)
        axis.plot(self.slt["time"], self.slt["eqn"]["VY"], label="slt - VY",
                  marker="o", ms=3, c="b")
        axis.plot(self.kfm.sim["time"], self.kfm.outputs["VY"], label="sim - VY",
                  marker="o", ms=3, c="g")
        axis.set_xlabel("t")
        axis.set_ylabel("VY")
        axis.legend()
        axis = self.vwr["2D"]["simOV"].getAxis(5)
        axis.plot(self.slt["time"], self.slt["eqn"]["VZ"], label="slt - VZ",
                  marker="o", ms=3, c="b")
        axis.plot(self.kfm.sim["time"], self.kfm.outputs["VZ"], label="sim - VZ",
                  marker="o", ms=3, c="g")
        axis.set_xlabel("t")
        axis.set_ylabel("VZ")
        axis.legend()

    def plotSimulationOutputVariablesA(self):
        """Plot simulation output variables: A"""

        # Plot simulation output variables.
        axis = self.vwr["2D"]["simOV"].getAxis(6)
        axis.plot(self.slt["time"], self.slt["eqn"]["AX"], label="slt - AX",
                  marker="o", ms=3, c="b")
        axis.plot(self.kfm.sim["time"], self.kfm.outputs["AX"], label="sim - AX",
                  marker="o", ms=3, c="g")
        axis.set_xlabel("t")
        axis.set_ylabel("AX")
        axis.legend()
        axis = self.vwr["2D"]["simOV"].getAxis(7)
        axis.plot(self.slt["time"], self.slt["eqn"]["AY"], label="slt - AY",
                  marker="o", ms=3, c="b")
        axis.plot(self.kfm.sim["time"], self.kfm.outputs["AY"], label="sim - AY",
                  marker="o", ms=3, c="g")
        axis.set_xlabel("t")
        axis.set_ylabel("AY")
        axis.legend()
        axis = self.vwr["2D"]["simOV"].getAxis(8)
        axis.plot(self.slt["time"], self.slt["eqn"]["AZ"], label="slt - AZ",
                  marker="o", ms=3, c="b")
        axis.plot(self.kfm.sim["time"], self.kfm.outputs["AZ"], label="sim - AZ",
                  marker="o", ms=3, c="g")
        axis.set_xlabel("t")
        axis.set_ylabel("AZ")
        axis.legend()

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
        pltCHVBtn = QPushButton("Hidden variables", simGUI)
        pltCHVBtn.clicked.connect(self.onPltCHVBtnClick)
        gdlFCL.addWidget(pltCHVBtn, 4, 0, 1, 2)

        # Set group box layout.
        gpbFCL = QGroupBox(simGUI)
        gpbFCL.setTitle("Control law")
        gpbFCL.setAlignment(Qt.AlignHCenter)
        gpbFCL.setLayout(gdlFCL)

        return gpbFCL

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
        if not self.kfm.solved:
            return

        # Plot control law hidden variables.
        time = self.kfm.sim["time"]
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
        if float(self.slt["cdfTf"].text()) < 0.:
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
            if prmDt < 0.:
                msg = "<em>&Delta;t</em> must be superior than 0."
                self.throwError(eId, "list item "+str(idx+1)+", "+msg)
                return False
            prmSigma = float(txt.split(";")[4].split()[1])
            if prmSigma < 0.:
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
        if float(self.sim["prmM"].text()) < 0.:
            self.throwError(eId, "mass must be superior than 0.")
            return False
        if float(self.sim["prmC"].text()) < 0.:
            self.throwError(eId, "damping coef must be superior than 0.")
            return False
        if float(self.sim["prmDt"].text()) < 0.:
            self.throwError(eId, "<em>&Delta;t</em> must be superior than 0.")
            return False
        if float(self.sim["prmExpOrd"].text()) < 0.:
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

    def computeControlLaw(self, states, sim):
        """Compute control law"""

        # Compute control law: get roll, pitch, yaw corrections.
        deltaAccRolY, deltaAccRolZ = self.computeRoll(states, sim)
        deltaAccPtcX, deltaAccPtcZ = self.computePitch(states, sim)
        deltaAccYawX, deltaAccYawY = self.computeYaw(states, sim)

        # Compute control law.
        fomX = deltaAccPtcX+deltaAccYawX
        fomY = deltaAccRolY+deltaAccYawY
        fomZ = deltaAccRolZ+deltaAccPtcZ
        matU = self.computeControl(fomX, fomY, fomZ, sim)

        # Save F/m to compute d(F/m)/dt next time.
        sim["ctlOldFoMX"] = fomX
        sim["ctlOldFoMY"] = fomY
        sim["ctlOldFoMZ"] = fomZ

        return matU

    def computeControl(self, fomX, fomY, fomZ, sim):
        """Compute control"""

        # Compute control law: modify plane throttle (F/m == acceleration).
        prmN = self.getLTISystemSize()
        matU = np.zeros((prmN, 1), dtype=float)
        matU[1, 0] = fomX
        matU[4, 0] = fomY
        matU[7, 0] = fomZ

        # Compute control law: modify plane acceleration (d(F/m)/dt).
        oldFoMX = self.sim["ctlOldFoMX"] if "ctlOldFoMX" in self.sim else 0.
        oldFoMY = self.sim["ctlOldFoMY"] if "ctlOldFoMY" in self.sim else 0.
        oldFoMZ = self.sim["ctlOldFoMZ"] if "ctlOldFoMZ" in self.sim else 0.
        prmDt = float(self.sim["prmDt"].text())
        matU[2, 0] = (fomX-oldFoMX)/prmDt
        matU[5, 0] = (fomY-oldFoMY)/prmDt
        matU[8, 0] = (fomZ-oldFoMZ)/prmDt

        # Save control law hidden variables.
        sim["ctlHV"]["FoM"]["X"].append(matU[1, 0])
        sim["ctlHV"]["FoM"]["Y"].append(matU[4, 0])
        sim["ctlHV"]["FoM"]["Z"].append(matU[7, 0])
        sim["ctlHV"]["d(FoM)/dt"]["X"].append(matU[2, 0])
        sim["ctlHV"]["d(FoM)/dt"]["Y"].append(matU[5, 0])
        sim["ctlHV"]["d(FoM)/dt"]["Z"].append(matU[8, 0])

        return matU

    def computeRoll(self, states, sim):
        """Compute control law: roll"""

        # Compute roll around X axis.
        velNow = np.array([0., states[4, 0], states[7, 0]]) # Velocity in YZ plane.
        accNow = np.array([0., states[5, 0], states[8, 0]]) # Acceleration in YZ plane.
        prmDt = float(self.sim["prmDt"].text())
        velNxt = velNow+accNow*prmDt # New velocity in YZ plane.
        roll = np.arccos(np.dot(velNow, velNxt)/(npl.norm(velNow)*npl.norm(velNxt)))
        roll = roll*(180./np.pi) # Roll angle in degrees.

        # Control roll.
        accTgt = accNow # Target acceleration.
        ctlRolMax = float(self.sim["ctlRolMax"].text())
        while np.abs(roll) > ctlRolMax:
            accTgt = accTgt*0.95 # Decrease acceleration by 5%.
            velNxt = velNow+accTgt*prmDt # New velocity in YZ plane.
            roll = np.arccos(np.dot(velNow, velNxt)/(npl.norm(velNow)*npl.norm(velNxt)))
            roll = roll*(180./np.pi) # Roll angle in degrees.
        deltaAcc = accTgt-accNow

        # Save control law hidden variables.
        sim["ctlHV"]["roll"].append(roll)

        return deltaAcc[1], deltaAcc[2]

    def computePitch(self, states, sim):
        """Compute control law: pitch"""

        # Compute pitch around Y axis.
        velNow = np.array([states[1, 0], 0., states[7, 0]]) # Velocity in XZ plane.
        accNow = np.array([states[2, 0], 0., states[8, 0]]) # Acceleration in XZ plane.
        prmDt = float(self.sim["prmDt"].text())
        velNxt = velNow+accNow*prmDt # New velocity in XZ plane.
        pitch = np.arccos(np.dot(velNow, velNxt)/(npl.norm(velNow)*npl.norm(velNxt)))
        pitch = pitch*(180./np.pi) # Pitch angle in degrees.

        # Control pitch.
        accTgt = accNow # Target acceleration.
        ctlPtcMax = float(self.sim["ctlPtcMax"].text())
        while np.abs(pitch) > ctlPtcMax:
            accTgt = accTgt*0.95 # Decrease acceleration by 5%.
            velNxt = velNow+accTgt*prmDt # New velocity in XZ plane.
            pitch = np.arccos(np.dot(velNow, velNxt)/(npl.norm(velNow)*npl.norm(velNxt)))
            pitch = pitch*(180./np.pi) # Pitch angle in degrees.
        deltaAcc = accTgt-accNow

        # Save control law hidden variables.
        sim["ctlHV"]["pitch"].append(pitch)

        return deltaAcc[0], deltaAcc[2]

    def computeYaw(self, states, sim):
        """Compute control law: yaw"""

        # Compute yaw around Z axis.
        velNow = np.array([states[1, 0], states[4, 0], 0.]) # Velocity in XY plane.
        accNow = np.array([states[2, 0], states[5, 0], 0.]) # Acceleration in XY plane.
        prmDt = float(self.sim["prmDt"].text())
        velNxt = velNow+accNow*prmDt # New velocity in XY plane.
        yaw = np.arccos(np.dot(velNow, velNxt)/(npl.norm(velNow)*npl.norm(velNxt)))
        yaw = yaw*(180./np.pi) # Yaw angle in degrees.

        # Control yaw.
        accTgt = accNow # Target acceleration.
        ctlYawMax = float(self.sim["ctlYawMax"].text())
        while np.abs(yaw) > ctlYawMax:
            accTgt = accTgt*0.95 # Decrease acceleration by 5%.
            velNxt = velNow+accTgt*prmDt # New velocity in XY plane.
            yaw = np.arccos(np.dot(velNow, velNxt)/(npl.norm(velNow)*npl.norm(velNxt)))
            yaw = yaw*(180./np.pi) # Yaw angle in degrees.
        deltaAcc = accTgt-accNow

        # Save control law hidden variables.
        sim["ctlHV"]["yaw"].append(yaw)

        return deltaAcc[0], deltaAcc[1]

    @staticmethod
    def getStateKeys():
        """Get states keys"""

        # Get states keys.
        return ["X", "VX", "AX", "Y", "VY", "AY", "Z", "VZ", "AZ"]

    def getOutputKeys(self):
        """Get outputs keys"""

        # Get outputs keys.
        return self.getStateKeys()

    def saveStatesOutputs(self, states, stateDic, outputs, outputDic):
        """Save states and outputs"""

        # Save states and outputs.
        keys = self.getStateKeys()
        for idx, key in enumerate(keys):
            stateDic[key].append(states[idx, 0])
        keys = self.getOutputKeys()
        for idx, key in enumerate(keys):
            outputDic[key].append(outputs[idx, 0])

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
        updateBtn.setToolTip("Update viewer")
        updateBtn.clicked.connect(self.onUpdateVwrBtnClick)

        return updateBtn

    def onUpdateVwrBtnClick(self):
        """Callback on update viewer button click"""

        # Update the view.
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

# Main program.
if __name__ == "__main__":
    # Check for python3.
    assert sys.version_info.major == 3, "This script is a python3 script."

    # Create application and controls GUI.
    app = QApplication(sys.argv)
    ctrWin = controllerGUI()

    # End main program.
    sys.exit(app.exec_())
