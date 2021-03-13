#!/usr/bin/env python3

"""2D viewer"""

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import QMainWindow, QWidget
from PyQt5.QtWidgets import QLabel, QPushButton, QHBoxLayout, QVBoxLayout
from PyQt5.QtWidgets import QLineEdit
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator

matplotlib.use("Qt5Agg")

class mpl2DCanvas(FigureCanvasQTAgg):
    """Matplotlib 2D canvas to be embedded in Qt widget"""

    def __init__(self, parent=None):
        """Initialize"""

        # Initialize.
        fig = plt.figure()
        super().__init__(fig)
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
        self.twinAxes = []
        for idx in range(nrows*ncols):
            axis = self.fig.add_subplot(nrows, ncols, idx+1)
            axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            self.axes.append(axis)
            self.twinAxes.append(None)
        self.fig.tight_layout()
        self.fig.subplots_adjust(hspace=0.3, wspace=0.3)

class kalmanFilter2DViewer(QMainWindow):
    """2D viewer"""

    def __init__(self, *args, **kwargs):
        """Initialize"""

        # Initialize.
        super().__init__(*args, **kwargs)
        self.mcvs = mpl2DCanvas(self)
        self.closed = False
        self.nrows = 0
        self.ncols = 0
        self.rangeMin = QLineEdit("N.A.", self)
        self.rangeMax = QLineEdit("N.A.", self)
        self.rangeMin.setValidator(QDoubleValidator())
        self.rangeMax.setValidator(QDoubleValidator())

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

        # Check axis.
        if idx < 0 or idx >= self.nrows*self.ncols:
            return None

        return self.mcvs.axes[idx]

    def getTwinAxis(self, idx=0, visible=True):
        """Get viewer twin axis"""

        # Check axis.
        if idx < 0 or idx >= self.nrows*self.ncols:
            return None

        # Return viewer twin axis.
        if self.mcvs.twinAxes[idx] is None:
            self.mcvs.twinAxes[idx] = self.mcvs.axes[idx].twinx()
        axis = self.mcvs.twinAxes[idx]
        axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        axis.set_visible(visible)

        return axis

    def draw(self):
        """Force draw of the scene"""

        # Set range and draw scene.
        self.onPltTRGBtnClick()

    def closeEvent(self, event):
        """Callback on closing window"""

        # Mark window as closed.
        self.closed = True
        event.accept()
