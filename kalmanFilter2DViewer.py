#!/usr/bin/env python3

"""2D viewer"""

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from PyQt5.QtWidgets import QLineEdit
from PyQt5.QtGui import QDoubleValidator

from kalmanFilterViewer import kalmanFilterViewer

matplotlib.use("Qt5Agg")

class mpl2DCanvas(FigureCanvasQTAgg):
    """Matplotlib 2D canvas to be embedded in widget"""

    def __init__(self, parent=None):
        """Initialize"""

        # Initialize.
        plt.close('all')
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

class kalmanFilter2DViewer(kalmanFilterViewer):
    """2D viewer"""

    def __init__(self, *args, **kwargs):
        """Initialize"""

        # Initialize.
        super().__init__(*args, **kwargs)
        self.mcvs = mpl2DCanvas(self)
        self.nrows = 0
        self.ncols = 0
        self.rangeMin = QLineEdit("N.A.", self)
        self.rangeMax = QLineEdit("N.A.", self)
        self.rangeMin.setValidator(QDoubleValidator())
        self.rangeMax.setValidator(QDoubleValidator())

        # Build the GUI.
        self.buildGUI(self.mcvs, \
                      self.rangeMin, self.rangeMax, \
                      self.onPltTRGBtnClick)

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

    def onPltTRGBtnClick(self):
        """Callback on changing plot time range"""

        # Check validity.
        rangeMin = float(self.rangeMin.text())
        rangeMax = float(self.rangeMax.text())
        if rangeMin >= rangeMax:
            self.throwErrorMsg("plot: min >= max")
            return

        # Set range for all axis.
        for row in range(self.nrows):
            for col in range(self.ncols):
                axis = self.getAxis(col+row*self.ncols)
                axis.set_xlim(rangeMin, rangeMax)

        # Draw scene.
        self.mcvs.draw()
