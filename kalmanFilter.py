#!/usr/bin/env python3

"""Kalman filter MVC (Model-View-Controller)"""

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget
from PyQt5.QtWidgets import QLabel, QComboBox, QPushButton, QHBoxLayout, QVBoxLayout
from PyQt5.QtGui import QVector3D
import pyqtgraph.opengl as pgl

class planeTrackingExample:
    """Plane tracking example"""

    def __init__(self):
        """Initialize"""

        # Initialize members.
        self.viewer = None
        self.parent = None
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

    def setAxisAndGrid(self):
        """Set axis and grid"""

        # Set axis and grid.
        vwrXGridSize = 4
        vwrYGridSize = 8
        vwrZGridSize = 12
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
        self.examples.append(planeTrackingExample())
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
    ctrGUI = controllerGUI()

    # End main program.
    sys.exit(app.exec_())
