#!/usr/bin/env python3

"""Kalman filter MVC (Model-View-Controller)"""

import sys
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from scipy.interpolate import lagrange
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget
from PyQt5.QtWidgets import QLabel, QComboBox, QPushButton, QHBoxLayout, QVBoxLayout
from PyQt5.QtWidgets import QGroupBox, QGridLayout, QLineEdit
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QListWidget, QCheckBox
from PyQt5.QtCore import Qt

matplotlib.use('Qt5Agg')

class mpl2DCanvas(FigureCanvasQTAgg):
    """Matplotlib 2D canvas to be embedded in Qt widget"""

    def __init__(self, parent=None):
        """Initialize"""

        # Initialize.
        fig = plt.figure()
        super(mpl2DCanvas, self).__init__(fig)
        self.setParent(parent)
        self.axes = fig.add_subplot(111)

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
        self.setCentralWidget(self.mcvs)
        self.closed = False

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

class viewer3DGUI(QMainWindow):
    """Kalman filter 3D viewer"""

    def __init__(self, *args, **kwargs):
        """Initialize"""

        # Initialize.
        super(viewer3DGUI, self).__init__(*args, **kwargs)
        self.mcvs = mpl3DCanvas(self)
        self.setCentralWidget(self.mcvs)
        self.closed = False

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

    def __init__(self):
        """Initialize"""

        # Initialize members.
        self.slt = {}
        self.msr = {}
        self.sim = {}
        self.solved = False
        self.clear()

    def clear(self):
        """Clear previous results"""

        # Clear previous results.
        keys = ["X", "Y", "Z", "VX", "VY", "VZ", "AX", "AY", "AZ"]
        for k in keys:
            self.slt[k] = None
        self.solved = False

    def setUp(self, msr, sim):
        """Setup solver"""

        # Copy data.
        self.msr = msr.copy()
        self.sim = sim.copy()

    def solve(self):
        """Solve based on Kalman filter"""

        # Don't solve if we have already a solution.
        if self.solved:
            return 0

        # Solve.
        self.solved = True

        return 0

class planeTrackingExample:
    """Plane tracking example"""

    def __init__(self, ctrGUI):
        """Initialize"""

        # Initialize members.
        self.vwr2D = None
        self.vwr3D = None
        self.ctrGUI = ctrGUI
        self.slt = {"sltId": ""}
        self.msr = {"sltId": "", "msrId": ""}
        self.sim = {"sltId": "", "msrId": ""}
        self.vwr = {}
        self.kfm = kalmanFilterModel()

    @staticmethod
    def getName():
        """Return example name"""

        # Return name.
        return "plane tracking"

    def createViewer(self):
        """Create viewer"""

        # Create viewer.
        if not self.vwr3D or self.vwr3D.closed:
            self.vwr3D = viewer3DGUI(self.ctrGUI)
        self.vwr3D.setWindowTitle("Kalman filter viewer")
        self.vwr3D.show()

        return self.vwr3D

    def updateViewer(self):
        """Update viewer"""

        # Clear the viewer.
        if self.vwr2D:
            axis = self.vwr2D.getAxis()
            axis.cla()
            axis.set_xlabel('t')
            axis.set_ylabel('z')
        axis = self.vwr3D.getAxis()
        axis.cla()
        axis.set_xlabel('x')
        axis.set_ylabel('y')
        axis.set_zlabel('z')

        # Update viewer.
        self.updateViewerSlt()
        self.updateViewerMsr()
        self.updateViewerSim()

        # Order and show legend.
        handles, labels = axis.get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        axis.legend(handles, labels)

        # Force viewer redraw.
        self.vwr3D.draw()

    def updateViewerSlt(self):
        """Update viewer: solution"""

        # Plot only if checked.
        if not self.vwr["ckbSlt"].isChecked():
            return

        # Plot Z.
        if self.vwr2D and not self.vwr2D.closed:
            self.onPltTZPBtnClick()

        # Time.
        prmTf = float(self.slt["cdfTf"].text())
        vwrNbPt = float(self.slt["vwrNbPt"].text())
        eqnT = np.linspace(0., prmTf, vwrNbPt)

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
        axis = self.vwr3D.getAxis()
        axis.plot3D(eqnX, eqnY, eqnZ, lw=vwrLnWd, color=clr,
                    label="flight path: x", marker="o", ms=vwrPosMks)

        return eqnX, eqnY, eqnZ

    def updateViewerSltV(self, eqnT, eqnX, eqnY, eqnZ):
        """Update viewer: plot velocity of the solution"""

        # Plot solution: velocity.
        eqnVX, eqnVY, eqnVZ = self.getVelocEquations(eqnT)
        clr = (0., 0.75, 1.) # Skyblue.
        vwrVelLgh = float(self.slt["vwrVelLgh"].text())
        if vwrVelLgh == 0.:
            return
        vwrVelNrm = self.slt["vwrVelNrm"].isChecked()
        axis = self.vwr3D.getAxis()
        axis.quiver3D(eqnX, eqnY, eqnZ, eqnVX, eqnVY, eqnVZ, color=clr,
                      length=vwrVelLgh, normalize=vwrVelNrm, label="flight path: v")

    def updateViewerSltA(self, eqnT, eqnX, eqnY, eqnZ):
        """Update viewer: plot acceleration of the solution"""

        # Plot solution: acceleration.
        eqnAX, eqnAY, eqnAZ = self.getAccelEquations(eqnT)
        clr = (0.25, 0., 0.5) # Indigo.
        vwrAccLgh = float(self.slt["vwrAccLgh"].text())
        if vwrAccLgh == 0.:
            return
        vwrAccNrm = self.slt["vwrAccNrm"].isChecked()
        axis = self.vwr3D.getAxis()
        axis.quiver3D(eqnX, eqnY, eqnZ, eqnAX, eqnAY, eqnAZ, colors=clr,
                      length=vwrAccLgh, normalize=vwrAccNrm, label="flight path: a")

    def getSltId(self):
        """Get solution identity (track solution features)"""

        # Get solution identity.
        sltId = self.slt["fpeAx"].text()
        sltId += ":"+self.slt["fpeTx"].text()
        sltId += ":"+self.slt["fpePhix"].text()
        sltId += ":"+self.slt["fpeAy"].text()
        sltId += ":"+self.slt["fpeTy"].text()
        sltId += ":"+self.slt["fpePhiy"].text()
        sltId += ":"+self.slt["fpeTiZi"].text()
        sltId += ":"+self.slt["cdiX0"].text()
        sltId += ":"+self.slt["cdiY0"].text()
        sltId += ":"+self.slt["cdiZ0"].text()
        sltId += ":"+self.slt["cdfTf"].text()

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
        axis = self.vwr3D.getAxis()
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
        axis = self.vwr3D.getAxis()
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
        axis = self.vwr3D.getAxis()
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

        # Plot only if checked.
        if not self.vwr["ckbSim"].isChecked():
            return

        # Solve based on Kalman filter.
        self.sim["cdfTf"] = self.slt["cdfTf"].text()
        self.kfm.solve()

        # Temporary: to be killed.
        prmTf = float(self.slt["cdfTf"].text())
        vwrNbPt = float(self.slt["vwrNbPt"].text())
        eqnT = np.linspace(0., prmTf, vwrNbPt)
        eqnX, eqnY, eqnZ = self.getDisplEquations(eqnT)
        eqnVX, eqnVY, eqnVZ = self.getVelocEquations(eqnT)
        eqnAX, eqnAY, eqnAZ = self.getAccelEquations(eqnT)
        self.kfm.slt["X"] = eqnX+1
        self.kfm.slt["Y"] = eqnY+1
        self.kfm.slt["Z"] = eqnZ+1
        self.kfm.slt["VX"] = eqnVX
        self.kfm.slt["VY"] = eqnVY
        self.kfm.slt["VZ"] = eqnVZ
        self.kfm.slt["AX"] = eqnAX
        self.kfm.slt["AY"] = eqnAY
        self.kfm.slt["AZ"] = eqnAZ

        # Plot solver results.
        self.updateViewerSimX()
        self.updateViewerSimV()
        self.updateViewerSimA()

    def updateViewerSimX(self):
        """Update viewer: plot displacement of the simulation"""

        # Plot simulation: displacement.
        eqnX, eqnY, eqnZ = self.kfm.slt["X"], self.kfm.slt["Y"], self.kfm.slt["Z"]
        vwrLnWd = float(self.sim["vwrLnWd"].text())
        if vwrLnWd == 0.:
            return
        vwrPosMks = float(self.sim["vwrPosMks"].text())
        clr = (0., 0.5, 0.) # Green.
        axis = self.vwr3D.getAxis()
        axis.plot3D(eqnX, eqnY, eqnZ, lw=vwrLnWd, color=clr,
                    label="simulation: x", marker="o", ms=vwrPosMks)

    def updateViewerSimV(self):
        """Update viewer: plot velocity of the simulation"""

        # Plot simulation: velocity.
        eqnX, eqnY, eqnZ = self.kfm.slt["X"], self.kfm.slt["Y"], self.kfm.slt["Z"]
        eqnVX, eqnVY, eqnVZ = self.kfm.slt["VX"], self.kfm.slt["VY"], self.kfm.slt["VZ"]
        clr = (0., 1., 0.) # Lime green.
        vwrVelLgh = float(self.sim["vwrVelLgh"].text())
        if vwrVelLgh == 0.:
            return
        vwrVelNrm = self.sim["vwrVelNrm"].isChecked()
        axis = self.vwr3D.getAxis()
        axis.quiver3D(eqnX, eqnY, eqnZ, eqnVX, eqnVY, eqnVZ, color=clr,
                      length=vwrVelLgh, normalize=vwrVelNrm, label="simulation: v")

    def updateViewerSimA(self):
        """Update viewer: plot acceleration of the simulation"""

        # Plot simulation: acceleration.
        eqnX, eqnY, eqnZ = self.kfm.slt["X"], self.kfm.slt["Y"], self.kfm.slt["Z"]
        eqnAX, eqnAY, eqnAZ = self.kfm.slt["AX"], self.kfm.slt["AY"], self.kfm.slt["AZ"]
        clr = (0., 0.2, 0.) # Dark green.
        vwrAccLgh = float(self.sim["vwrAccLgh"].text())
        if vwrAccLgh == 0.:
            return
        vwrAccNrm = self.sim["vwrAccNrm"].isChecked()
        axis = self.vwr3D.getAxis()
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
        self.slt["cdfTf"] = QLineEdit("2.", self.ctrGUI)
        self.slt["vwrNbPt"] = QLineEdit("50", self.ctrGUI)
        self.slt["vwrLnWd"] = QLineEdit("1.", self.ctrGUI)
        self.slt["vwrPosMks"] = QLineEdit("5", self.ctrGUI)
        self.slt["vwrVelLgh"] = QLineEdit("0.01", self.ctrGUI)
        self.slt["vwrVelNrm"] = QCheckBox("Normalize", self.ctrGUI)
        self.slt["vwrAccLgh"] = QLineEdit("0.001", self.ctrGUI)
        self.slt["vwrAccNrm"] = QCheckBox("Normalize", self.ctrGUI)

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
        """Callback on plotting lagrange Z polynomial"""

        # Time.
        prmTf = float(self.slt["cdfTf"].text())
        vwrNbPt = float(self.slt["vwrNbPt"].text())
        eqnT = np.linspace(0., prmTf, vwrNbPt)

        # Compute lagrange Z polynomial.
        poly = self.getZPoly()
        eqnZ = poly(eqnT)

        # Create or retrieve viewer.
        if not self.vwr2D or self.vwr2D.closed:
            self.vwr2D = viewer2DGUI(self.ctrGUI)
            self.vwr2D.setWindowTitle("Lagrange Z polynomial")
            self.vwr2D.show()

        # Plot lagrange Z polynomial.
        axis = self.vwr2D.getAxis()
        vwrLnWd = float(self.slt["vwrLnWd"].text())
        if vwrLnWd > 0.:
            vwrPosMks = float(self.slt["vwrPosMks"].text())
            clr = (0., 0., 1.) # Blue.
            axis.plot(eqnT, eqnZ, color=clr, label="z", marker="o", lw=vwrLnWd, ms=vwrPosMks)
        prmTi, prmZi = self.getZPolyPts()
        axis.scatter(prmTi, prmZi, c="r", marker="X", label="interpolation point")
        axis.legend()

        # Draw scene.
        self.vwr2D.draw()

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
        self.sim["prmDt"] = QLineEdit("0.03", self.ctrGUI)
        self.sim["prmVrb"] = QLineEdit("1", self.ctrGUI)
        self.sim["cdiX0"] = QLineEdit("0.2", self.ctrGUI)
        self.sim["cdiY0"] = QLineEdit("0.2", self.ctrGUI)
        self.sim["cdiZ0"] = QLineEdit("0.2", self.ctrGUI)
        self.sim["cdiSigX0"] = QLineEdit("0.5", self.ctrGUI)
        self.sim["cdiSigY0"] = QLineEdit("0.5", self.ctrGUI)
        self.sim["cdiSigZ0"] = QLineEdit("0.5", self.ctrGUI)
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
        gpbVwr = self.fillSimGUIVwr(simGUI)

        # Set group box layout.
        anlLay = QHBoxLayout(simGUI)
        anlLay.addWidget(gpbPrm)
        anlLay.addWidget(gpbX0)
        anlLay.addWidget(gpbVwr)
        simGUI.setLayout(anlLay)

    def fillSimGUIPrm(self, simGUI):
        """Fill simulation GUI: parameters"""

        # Create simulation GUI: simulation parameters.
        gdlPrm = QGridLayout(simGUI)
        gdlPrm.addWidget(QLabel("<em>&Delta;t</em>:", simGUI), 0, 0)
        gdlPrm.addWidget(self.sim["prmDt"], 0, 1)
        gdlPrm.addWidget(QLabel("verbose level:", simGUI), 1, 0)
        gdlPrm.addWidget(self.sim["prmVrb"], 1, 1)

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
        title = "x(t = 0) = x<sub>0</sub> &plusmn; &sigma;<sub>x0</sub>"
        gdlX0.addWidget(QLabel(title, simGUI), 0, 0, 1, 2)
        gdlX0.addWidget(QLabel("x<sub>0</sub>", simGUI), 1, 0)
        gdlX0.addWidget(self.sim["cdiX0"], 1, 1)
        title = "y(t = 0) = y<sub>0</sub> &plusmn; &sigma;<sub>y0</sub>"
        gdlX0.addWidget(QLabel(title, simGUI), 2, 0, 1, 2)
        gdlX0.addWidget(QLabel("y<sub>0</sub>", simGUI), 3, 0)
        gdlX0.addWidget(self.sim["cdiY0"], 3, 1)
        title = "z(t = 0) = z<sub>0</sub> &plusmn; &sigma;<sub>z0</sub>"
        gdlX0.addWidget(QLabel(title, simGUI), 4, 0, 1, 2)
        gdlX0.addWidget(QLabel("z<sub>0</sub>", simGUI), 5, 0)
        gdlX0.addWidget(self.sim["cdiZ0"], 5, 1)
        gdlX0.addWidget(QLabel("&sigma;<sub>x0</sub>", simGUI), 1, 2)
        gdlX0.addWidget(self.sim["cdiSigX0"], 1, 3)
        gdlX0.addWidget(QLabel("&sigma;<sub>y0</sub>", simGUI), 3, 2)
        gdlX0.addWidget(self.sim["cdiSigY0"], 3, 3)
        gdlX0.addWidget(QLabel("&sigma;<sub>z0</sub>", simGUI), 5, 2)
        gdlX0.addWidget(self.sim["cdiSigZ0"], 5, 3)

        # Set group box layout.
        gpbX0 = QGroupBox(simGUI)
        gpbX0.setTitle("Initial conditions")
        gpbX0.setAlignment(Qt.AlignHCenter)
        gpbX0.setLayout(gdlX0)

        return gpbX0

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
        if float(self.sim["prmDt"].text()) < 0.:
            self.throwError(eId, "<em>&Delta;t</em> must be superior than 0.")
            return False
        if float(self.sim["prmVrb"].text()) < 0.:
            self.throwError(eId, "verbose level must be superior than 0.")
            return False

        return self.checkValiditySimVwr()

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
