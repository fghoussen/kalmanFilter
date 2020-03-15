#!/usr/bin/env python3

"""Kalman filter MVC (Model-View-Controller)"""

import sys
import math
import numpy as np
from scipy.interpolate import lagrange
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget
from PyQt5.QtWidgets import QLabel, QComboBox, QPushButton, QHBoxLayout, QVBoxLayout
from PyQt5.QtWidgets import QGroupBox, QGridLayout, QLineEdit
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QVector3D
import pyqtgraph.opengl as pgl

class planeTrackingExample:
    """Plane tracking example"""

    def __init__(self, ctrGUI):
        """Initialize"""

        # Initialize members.
        self.viewer = None
        self.ctrGUI = ctrGUI
        self.slt = {}
        self.vwr = {}

    @staticmethod
    def getName():
        """Return example name"""

        # Return name.
        return "plane tracking"

    def createViewer(self, sameOpts=True):
        """Return example viewer"""

        # Get viewer options if any.
        opts = None
        if self.viewer and sameOpts:
            opts = self.viewer.opts

        # Create viewer with options.
        self.viewer = pgl.GLViewWidget()
        if opts:
            self.viewer.opts = opts

        return self.viewer

    def updateViewer(self):
        """Update viewer"""

        # Set axis and grid.
        self.setAxisAndGrid()

        # Time.
        prmTf = float(self.slt["tf"].text())
        vwrSltNbPt = float(self.vwr["sltNbPt"].text())
        eqnT = np.linspace(0., prmTf, vwrSltNbPt)

        # 3D plot.
        eqnX, eqnY, eqnZ = self.getEquations(eqnT)
        pts = np.array([(xyz[0], xyz[1], xyz[2]) for xyz in zip(eqnX, eqnY, eqnZ)])
        vwrSltLineWdh = float(self.vwr["sltLineWdh"].text())
        viewer = pgl.GLLinePlotItem(pos=pts, width=vwrSltLineWdh, antialias=True)
        self.viewer.addItem(viewer)
        vwrSltPtSz = float(self.vwr["sltPtSz"].text())
        viewer = pgl.GLScatterPlotItem(pos=pts, size=vwrSltPtSz)
        self.viewer.addItem(viewer)

    def setAxisAndGrid(self):
        """Set axis and grid"""

        # Set axis and grid.
        vwrXGridSize = float(self.vwr["xGridSize"].text())
        vwrYGridSize = float(self.vwr["yGridSize"].text())
        vwrZGridSize = float(self.vwr["zGridSize"].text())
        axis = pgl.GLAxisItem(size=QVector3D(vwrXGridSize/2, vwrYGridSize/2, vwrZGridSize/2))
        self.viewer.addItem(axis)
        xGrid = pgl.GLGridItem(QVector3D(vwrXGridSize, vwrXGridSize, 1))
        xGrid.setSpacing(spacing=QVector3D(1, 1, 1))
        yGrid = pgl.GLGridItem(QVector3D(vwrYGridSize, vwrYGridSize, 1))
        yGrid.setSpacing(spacing=QVector3D(1, 1, 1))
        yGrid.rotate(90, 0, 1, 0)
        zGrid = pgl.GLGridItem(QVector3D(vwrZGridSize, vwrZGridSize, 1))
        zGrid.setSpacing(spacing=QVector3D(1, 1, 1))
        zGrid.rotate(90, 1, 0, 0)
        self.viewer.addItem(xGrid)
        self.viewer.addItem(yGrid)
        self.viewer.addItem(zGrid)

    def getEquations(self, eqnT):
        """Get equations"""

        # Get equations.
        eqnX = self.getXEquation(eqnT)
        eqnY = self.getYEquation(eqnT)
        eqnZ = self.getZEquation(eqnT)

        return eqnX, eqnY, eqnZ

    def getXEquation(self, eqnT):
        """Get X equation"""

        # Get X equation.
        prmV0 = float(self.slt["x0"].text())
        prmA = float(self.slt["ax"].text())
        prmPhi = float(self.slt["phix"].text())
        prmT = float(self.slt["Tx"].text())
        prmB = prmV0-prmA*np.cos(prmPhi)
        omega = 2.*math.pi/prmT
        eqnX = prmA*np.cos(omega*eqnT+prmPhi)+prmB

        return eqnX

    def getYEquation(self, eqnT):
        """Get Y equation"""

        # Get Y equation.
        prmV0 = float(self.slt["y0"].text())
        prmA = float(self.slt["ay"].text())
        prmPhi = float(self.slt["phiy"].text())
        prmT = float(self.slt["Ty"].text())
        prmB = prmV0-prmA*np.sin(prmPhi)
        omega = 2.*math.pi/prmT
        eqnY = prmA*np.sin(omega*eqnT+prmPhi)+prmB

        return eqnY

    def getZEquation(self, eqnT):
        """Get Z equation"""

        # Get Z equation.
        prmZ0 = float(self.slt["z0"].text())
        prmTiZi = self.slt["tizi"].text()
        prmTi = np.array([0.], dtype=float)
        prmZi = np.array([prmZ0], dtype=float)
        for tokTiZi in prmTiZi.split(","):
            if len(tokTiZi.split()) != 2:
                self.throwError("each t<sub>i</sub> must match a z<sub>i</sub>")
                continue
            tokTi, tokZi = tokTiZi.split()
            if np.abs(float(tokTi)) < 1.e-6:
                self.throwError("t<sub>i</sub> must be superior than 0.")
                continue
            prmTi = np.append(prmTi, float(tokTi))
            prmZi = np.append(prmZi, float(tokZi))
        poly = lagrange(prmTi, prmZi)
        eqnZ = poly(eqnT)

        return eqnZ

    def throwError(self, txt):
        """Throw an error message"""

        # Create error message box.
        msg = QMessageBox(self.ctrGUI)
        msg.setIcon(QMessageBox.Critical)
        msg.setText("Error")
        msg.setText("Error: "+txt)
        msg.exec_()

    def createSltGUI(self):
        """Create solution GUI"""

        # Create group box.
        sltGUI = QGroupBox(self.ctrGUI)
        sltGUI.setTitle("Analytic solution: targeting real flight path")
        sltGUI.setAlignment(Qt.AlignHCenter)

        # Store analytic parameters.
        self.slt["ax"] = QLineEdit("1.", self.ctrGUI)
        self.slt["ay"] = QLineEdit("2.", self.ctrGUI)
        self.slt["Tx"] = QLineEdit("1.", self.ctrGUI)
        self.slt["Ty"] = QLineEdit("1.", self.ctrGUI)
        self.slt["phix"] = QLineEdit("0.", self.ctrGUI)
        self.slt["phiy"] = QLineEdit("0.", self.ctrGUI)
        self.slt["tizi"] = QLineEdit("0.1 0.1, 0.5 1.2, 1.5 0.8, 1.9 0.1, 2. 0.", self.ctrGUI)
        self.slt["x0"] = QLineEdit("0.", self.ctrGUI)
        self.slt["y0"] = QLineEdit("0.", self.ctrGUI)
        self.slt["z0"] = QLineEdit("0.", self.ctrGUI)
        self.slt["tf"] = QLineEdit("2.", self.ctrGUI)
        self.vwr["sltNbPt"] = QLineEdit("50", self.ctrGUI)
        self.vwr["sltLineWdh"] = QLineEdit("1.", self.ctrGUI)
        self.vwr["sltPtSz"] = QLineEdit("5.", self.ctrGUI)
        self.vwr["xGridSize"] = QLineEdit("4", self.ctrGUI)
        self.vwr["yGridSize"] = QLineEdit("8", self.ctrGUI)
        self.vwr["zGridSize"] = QLineEdit("12", self.ctrGUI)

        # Fill solution GUI.
        self.fillSltGUI(sltGUI)

        return sltGUI

    def fillSltGUI(self, sltGUI):
        """Fill solution GUI"""

        # Create group box.
        gpbXi = self.fillSltGUIXi(sltGUI)
        gpbX0 = self.fillSltGUIX0(sltGUI)
        gpbTf = self.fillSltGUITf(sltGUI)
        gpbWvr = self.fillSltGUIVwr(sltGUI)

        # Set group box layout.
        anlLay = QHBoxLayout(sltGUI)
        anlLay.addWidget(gpbXi)
        anlLay.addWidget(gpbX0)
        anlLay.addWidget(gpbTf)
        anlLay.addWidget(gpbWvr)
        sltGUI.setLayout(anlLay)

    def fillSltGUIXi(self, sltGUI):
        """Fill solution GUI : flight path equation"""

        # Create analytic parameters GUI: flight path equation.
        gdlXi = QGridLayout(sltGUI)
        title = "x(t) = a<sub>x</sub>cos(2&pi;/T<sub>x</sub>*t+&phi;<sub>x</sub>)"
        gdlXi.addWidget(QLabel(title, sltGUI), 0, 0, 1, 6)
        gdlXi.addWidget(QLabel("a<sub>x</sub>", sltGUI), 1, 0)
        gdlXi.addWidget(self.slt["ax"], 1, 1)
        gdlXi.addWidget(QLabel("T<sub>x</sub>", sltGUI), 1, 2)
        gdlXi.addWidget(self.slt["Tx"], 1, 3)
        gdlXi.addWidget(QLabel("&phi;<sub>x</sub>", sltGUI), 1, 4)
        gdlXi.addWidget(self.slt["phix"], 1, 5)
        title = "y(t) = a<sub>y</sub>sin(2&pi;/T<sub>y</sub>*t+&phi;<sub>y</sub>)"
        gdlXi.addWidget(QLabel(title, sltGUI), 2, 0, 1, 6)
        gdlXi.addWidget(QLabel("a<sub>y</sub>", sltGUI), 3, 0)
        gdlXi.addWidget(self.slt["ay"], 3, 1)
        gdlXi.addWidget(QLabel("T<sub>y</sub>", sltGUI), 3, 2)
        gdlXi.addWidget(self.slt["Ty"], 3, 3)
        gdlXi.addWidget(QLabel("&phi;<sub>y</sub>", sltGUI), 3, 4)
        gdlXi.addWidget(self.slt["phiy"], 3, 5)
        title = "z<sub>1</sub> = z(t<sub>1</sub>), z<sub>2</sub> = z(t<sub>2</sub>), "
        title += "z<sub>3</sub> = z(t<sub>3</sub>), ... - "
        title += "z(t) = z<sub>1</sub>t+z<sub>2</sub>t<sup>2</sup>+z<sub>3</sub>t<sup>3</sup>+..."
        gdlXi.addWidget(QLabel(title, sltGUI), 4, 0, 1, 6)
        gdlXi.addWidget(QLabel("t<sub>i</sub> z<sub>i</sub>", sltGUI), 5, 0)
        gdlXi.addWidget(self.slt["tizi"], 5, 1, 1, 6)

        # Set group box layout.
        gpbXi = QGroupBox(sltGUI)
        gpbXi.setTitle("Flight path equation")
        gpbXi.setAlignment(Qt.AlignHCenter)
        gpbXi.setLayout(gdlXi)

        return gpbXi

    def fillSltGUIX0(self, sltGUI):
        """Fill solution GUI : initial conditions"""

        # Create analytic parameters GUI: initial conditions.
        gdlX0 = QGridLayout(sltGUI)
        title = "x(t = 0) = x<sub>0</sub>"
        gdlX0.addWidget(QLabel(title, sltGUI), 0, 0, 1, 2)
        gdlX0.addWidget(QLabel("x<sub>0</sub>", sltGUI), 1, 0)
        gdlX0.addWidget(self.slt["x0"], 1, 1)
        title = "y(t = 0) = y<sub>0</sub>"
        gdlX0.addWidget(QLabel(title, sltGUI), 2, 0, 1, 2)
        gdlX0.addWidget(QLabel("y<sub>0</sub>", sltGUI), 3, 0)
        gdlX0.addWidget(self.slt["y0"], 3, 1)
        title = "z(t = 0) = z<sub>0</sub>"
        gdlX0.addWidget(QLabel(title, sltGUI), 4, 0, 1, 2)
        gdlX0.addWidget(QLabel("z<sub>0</sub>", sltGUI), 5, 0)
        gdlX0.addWidget(self.slt["z0"], 5, 1)

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
        gdlTf.addWidget(self.slt["tf"], 0, 1)

        # Set group box layout.
        gpbTf = QGroupBox(sltGUI)
        gpbTf.setTitle("Final conditions")
        gpbTf.setAlignment(Qt.AlignHCenter)
        gpbTf.setLayout(gdlTf)

        return gpbTf

    def fillSltGUIVwr(self, sltGUI):
        """Fill solution GUI : viewer parameters"""

        # Create analytic parameters GUI: plot parameters.
        gdlWvr = QGridLayout(sltGUI)
        gdlWvr.addWidget(QLabel("Nb points", sltGUI), 0, 0)
        gdlWvr.addWidget(self.vwr["sltNbPt"], 0, 1)
        gdlWvr.addWidget(QLabel("Line width", sltGUI), 1, 0)
        gdlWvr.addWidget(self.vwr["sltLineWdh"], 1, 1)
        gdlWvr.addWidget(QLabel("Point size", sltGUI), 2, 0)
        gdlWvr.addWidget(self.vwr["sltPtSz"], 2, 1)
        gdlWvr.addWidget(QLabel("X-grid size", sltGUI), 3, 0)
        gdlWvr.addWidget(self.vwr["xGridSize"], 3, 1)
        gdlWvr.addWidget(QLabel("Y-grid size", sltGUI), 4, 0)
        gdlWvr.addWidget(self.vwr["yGridSize"], 4, 1)
        gdlWvr.addWidget(QLabel("Z-grid size", sltGUI), 5, 0)
        gdlWvr.addWidget(self.vwr["zGridSize"], 5, 1)

        # Set group box layout.
        gpbWvr = QGroupBox(sltGUI)
        gpbWvr.setTitle("3D viewer parameters")
        gpbWvr.setAlignment(Qt.AlignHCenter)
        gpbWvr.setLayout(gdlWvr)

        return gpbWvr

class controllerGUI(QMainWindow):
    """Kalman filter controller"""

    def __init__(self):
        """Initialize"""

        # Initialize members.
        super().__init__()
        self.setWindowTitle("Kalman filter controller")
        self.viewer = QMainWindow(self)
        self.viewer.setWindowTitle("Kalman filter viewer")
        self.examples = []
        self.examples.append(planeTrackingExample(self))
        self.comboEx, self.comboGUI = self.addExampleCombo()
        self.updateBtn = self.addUpdateButton()

        # Show controls GUI.
        self.show()

        # Show viewer GUI.
        self.viewer.show()

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

        # Create controls.
        layCtr = QVBoxLayout()
        layCtr.addWidget(self.comboGUI)
        for example in self.examples:
            if example.getName() == txt:
                sltGUI = example.createSltGUI()
                layCtr.addWidget(sltGUI)
                break
        layCtr.addWidget(self.updateBtn)
        guiCtr = QWidget(self)
        guiCtr.setLayout(layCtr)
        self.setCentralWidget(guiCtr)

        # Update the view.
        self.onUpdateBtnClick()

    def addUpdateButton(self):
        """Add button to update the viewer"""

        # Add button to update the viewer.
        updateBtn = QPushButton("Update viewer", self)
        updateBtn.setToolTip("Update viewer")
        updateBtn.clicked.connect(self.onUpdateBtnClick)

        return updateBtn

    def onUpdateBtnClick(self):
        """Callback on update button click"""

        # Update the view.
        for example in self.examples:
            if example.getName() == self.comboEx.currentText():
                # Recreate a viewer to reset the view.
                viewer = example.createViewer()
                self.viewer.setCentralWidget(viewer)

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
