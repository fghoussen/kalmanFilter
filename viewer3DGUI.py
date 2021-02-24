#!/usr/bin/env python3

"""3D viewer"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import QMainWindow, QWidget
from PyQt5.QtWidgets import QLabel, QPushButton, QHBoxLayout, QVBoxLayout
from PyQt5.QtWidgets import QLineEdit
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QCheckBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator

matplotlib.use("Qt5Agg")

class mpl3DCanvas(FigureCanvasQTAgg):
    """Matplotlib 3D canvas to be embedded in Qt widget"""

    def __init__(self, parent=None):
        """Initialize"""

        # Initialize.
        fig = plt.figure()
        super().__init__(fig)
        self.setParent(parent)
        self.axes = fig.add_subplot(111, projection=Axes3D.name)

class viewer3DGUI(QMainWindow):
    """3D viewer"""

    def __init__(self, *args, **kwargs):
        """Initialize"""

        # Initialize.
        super().__init__(*args, **kwargs)
        self.mcvs = mpl3DCanvas(self)
        self.closed = False
        self.rangeMin = QLineEdit("N.A.", self)
        self.rangeMax = QLineEdit("N.A.", self)
        self.rangeMin.setValidator(QDoubleValidator())
        self.rangeMax.setValidator(QDoubleValidator())
        self.plot3D = {}
        self.scatter3D = {}
        self.quiver3D = {}

        # Set window as non modal.
        self.setWindowModality(Qt.NonModal)

        # Set the layout
        subLayout1 = QVBoxLayout()
        subLayout1.addWidget(NavigationToolbar(self.mcvs, self))
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

        # Plot, scatter, quiver.
        axis = self.getAxis()
        axis.cla()
        for lbl in self.plot3D:
            self.viewPlot3D(lbl, self.plot3D[lbl])
        for lbl in self.scatter3D:
            self.viewScatter3D(lbl, self.scatter3D[lbl])
        for lbl in self.quiver3D:
            self.viewQuiver3D(lbl, self.quiver3D[lbl])

        # 3D viewer: order and show legend.
        handles, labels = axis.get_legend_handles_labels()
        if len(handles) > 0 and len(labels) > 0:
            labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
            axis.legend(handles, labels)

        # Draw scene.
        self.mcvs.draw()

    def setUp(self, rangeMax, rangeMin="0."):
        """Set up"""

        # Set up.
        self.rangeMin.setText(rangeMin)
        self.rangeMax.setText(rangeMax)

    def setRange(self, eqnT, eqnX, eqnY, eqnZ):
        """Set range to data"""

        # Set range to data.
        rangeMin = float(self.rangeMin.text())
        rangeMax = float(self.rangeMax.text())
        eqnMask = np.all([rangeMin <= eqnT, eqnT <= rangeMax], axis=0)

        return eqnX[eqnMask], eqnY[eqnMask], eqnZ[eqnMask]

    def clear(self):
        """Clear data to view"""

        # Clear data to view.
        self.plot3D.clear()
        self.scatter3D.clear()
        self.quiver3D.clear()

    def getAxis(self):
        """Get viewer axis"""

        # Return viewer axis.
        return self.mcvs.axes

    def draw(self):
        """Force draw of the scene"""

        # Set range and draw scene.
        self.onPltTRGBtnClick()

    def closeEvent(self, event):
        """Callback on closing window"""

        # Mark window as closed.
        self.closed = True
        event.accept()

    def addPlot(self, lbl, data, clr):
        """Add plot to data to view"""

        # Add plot to data to view.
        self.plot3D[lbl] = {}
        for var in ["T", "X", "Y", "Z"]:
            self.plot3D[lbl][var] = data[var]
        self.plot3D[lbl]["clr"] = clr
        for key in ["vwrLnWd", "vwrPosMks"]:
            if isinstance(data[key], QLineEdit):
                self.plot3D[lbl][key] = float(data[key].text())
            else:
                self.plot3D[lbl][key] = data[key]

    def addScatter(self, lbl, data, clr):
        """Add scatter to data to view"""

        # Add scatter to data to view.
        self.scatter3D[lbl] = {}
        for var in ["T", "X", "Y", "Z"]:
            self.scatter3D[lbl][var] = data[var]
        self.scatter3D[lbl]["clr"] = clr
        self.scatter3D[lbl]["vwrPosMks"] = data["vwrPosMks"]

    def addQuiver(self, lbl, data, uvw, opts):
        """Add quiver to data to view"""

        # Add quiver to data to view.
        self.quiver3D[lbl] = {}
        for var in ["T", "X", "Y", "Z"]:
            self.quiver3D[lbl][var] = data[var]
        for idx, var in enumerate(["U", "V", "W"]):
            self.quiver3D[lbl][var] = data[uvw[idx]]
        self.quiver3D[lbl]["clr"] = opts["clr"]
        lnr = opts["lnr"]
        for idx, key in enumerate(["vwrLgh", "vwrNrm", "vwrALR"]):
            if isinstance(data[lnr[idx]], QCheckBox):
                self.quiver3D[lbl][key] = data[lnr[idx]].isChecked()
            elif isinstance(data[lnr[idx]], QLineEdit):
                self.quiver3D[lbl][key] = float(data[lnr[idx]].text())
            else:
                self.quiver3D[lbl][key] = data[lnr[idx]]

    def viewPlot3D(self, lbl, data):
        """Update viewer: plot"""

        # Plot.
        eqnX, eqnY, eqnZ = data["X"], data["Y"], data["Z"]
        vwrLnWd, vwrPosMks, clr = data["vwrLnWd"], data["vwrPosMks"], data["clr"]
        if vwrLnWd == 0.:
            return
        axis = self.getAxis()
        eqnX, eqnY, eqnZ = self.setRange(data["T"], eqnX, eqnY, eqnZ)
        axis.plot3D(eqnX, eqnY, eqnZ, lw=vwrLnWd, color=clr, label=lbl, marker="o", ms=vwrPosMks)

    def viewScatter3D(self, lbl, data):
        """Update viewer: scatter"""

        # Plot.
        posX, posY, posZ = data["X"], data["Y"], data["Z"]
        vwrPosMks, clr = data["vwrPosMks"], data["clr"]
        if vwrPosMks == 0.:
            return
        axis = self.getAxis()
        posX, posY, posZ = self.setRange(data["T"], posX, posY, posZ)
        axis.scatter3D(posX, posY, posZ, c=clr, marker="^", alpha=1, s=vwrPosMks, label=lbl)

    def viewQuiver3D(self, lbl, data):
        """Update viewer: quiver"""

        # Plot solution: velocity.
        eqnX, eqnY, eqnZ = data["X"], data["Y"], data["Z"]
        eqnU, eqnV, eqnW = data["U"], data["V"], data["W"]
        vwrLgh, clr = data["vwrLgh"], data["clr"]
        if vwrLgh == 0.:
            return
        vwrNrm, vwrALR = data["vwrNrm"], data["vwrALR"]
        axis = self.getAxis()
        eqnX, eqnY, eqnZ = self.setRange(data["T"], eqnX, eqnY, eqnZ)
        eqnU, eqnV, eqnW = self.setRange(data["T"], eqnU, eqnV, eqnW)
        axis.quiver3D(eqnX, eqnY, eqnZ, eqnU, eqnV, eqnW, color=clr,
                      length=vwrLgh, normalize=vwrNrm, arrow_length_ratio=vwrALR, label=lbl)
