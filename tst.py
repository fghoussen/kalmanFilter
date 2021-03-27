#!/usr/bin/env python3

"""Kalman filter tests"""

import unittest

import sys
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest

from kalmanFilterController import kalmanFilterController

class testKalmanFilter(unittest.TestCase):
    """Kalman filter unittest class"""

    @staticmethod
    def testKalmanFilterPlaneExampleXYZLine():
        """Kalman filter test: plane example, XYZ line"""

        # Create application and controls GUI.
        app = QApplication(sys.argv)
        ctrWin = kalmanFilterController()
        ctrWin.comboEx.setCurrentText("plane tracking")
        for example in ctrWin.examples:
            if example.getName() == ctrWin.comboEx.currentText():
                QTest.mouseClick(example.expQRB["3L"], QtCore.Qt.LeftButton)
        QTest.mouseClick(ctrWin.updateBtn, QtCore.Qt.LeftButton)
        QtCore.QTimer.singleShot(5 * 1000, QtCore.QCoreApplication.quit)
        app.exec_()

# Main program.
if __name__ == '__main__':
    unittest.main(failfast=True)
