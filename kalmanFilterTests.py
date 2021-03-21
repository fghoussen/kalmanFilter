#!/usr/bin/env python3

"""Kalman filter tests"""

import unittest

import sys
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication

from kalmanFilterController import kalmanFilterController

class testKalmanFilter(unittest.TestCase):
    """Kalman filter unittest class"""

    @staticmethod
    def testKalmanFilterXYZLine():
        """Kalman filter test: XYZ line"""

        # Create application and controls GUI.
        app = QApplication(sys.argv)
        ctrWin = kalmanFilterController()
        ctrWin.showMaximized()
        QtCore.QTimer.singleShot(5 * 1000, QtCore.QCoreApplication.quit)
        app.exec_()

# Main program.
if __name__ == '__main__':
    unittest.main(failfast=True)
