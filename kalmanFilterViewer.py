#!/usr/bin/env python3

"""Viewer"""

from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QLabel, QPushButton, QHBoxLayout, QVBoxLayout
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMessageBox

class kalmanFilterViewer(QMainWindow):
    """Viewer"""

    def __init__(self, *args, **kwargs):
        """Initialize"""

        # Initialize.
        super().__init__(*args, **kwargs)
        self.closed = False

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

    def throwErrorMsg(self, msg):
        """Throw error message."""

        # Throw error message.
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Critical)
        msg.setText(msg)
        msg.exec_()

    def closeEvent(self, event):
        """Callback on closing window"""

        # Mark window as closed.
        self.closed = True
        event.accept()
