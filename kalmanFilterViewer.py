#!/usr/bin/env python3

"""Viewer"""

from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QLabel, QPushButton, QHBoxLayout, QVBoxLayout
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt

class kalmanFilterViewer(QMainWindow):
    """2D viewer"""

    def buildGUI(self, mcvs, rangeMin, rangeMax, onPltTRGBtnClick):
        """Build GUI"""

        # Set the layout
        subLayout1 = QVBoxLayout()
        subLayout1.addWidget(NavigationToolbar(mcvs, self))
        subLayout1.addWidget(mcvs)
        subLayout2 = QHBoxLayout()
        subLayout2.addWidget(QLabel("time min:", self))
        subLayout2.addWidget(rangeMin)
        subLayout2.addWidget(QLabel("time max:", self))
        subLayout2.addWidget(rangeMax)
        pltTRGBtn = QPushButton("Set range", self)
        pltTRGBtn.clicked.connect(onPltTRGBtnClick)
        subLayout2.addWidget(pltTRGBtn)
        rootLayout = QVBoxLayout()
        rootLayout.addLayout(subLayout1)
        rootLayout.addLayout(subLayout2)

        # Set window as non modal.
        self.setWindowModality(Qt.NonModal)

        # Build the GUI.
        vwrGUI = QWidget(self)
        vwrGUI.setLayout(rootLayout)
        self.setCentralWidget(vwrGUI)
