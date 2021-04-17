#!/usr/bin/env python3

"""Kalman filter tests"""

import unittest

import functools

from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest

from kalmanFilterController import kalmanFilterController

app = QApplication([])

class testKalmanFilter(unittest.TestCase):
    """Kalman filter unittest class"""

    @staticmethod
    def testKalmanFilterPlaneExampleXYZLine():
        """Kalman filter test: plane example - XYZ line"""

        # Kalman filter test.
        testKalmanFilter.kalmanFilterPlaneExample("3L")

    @staticmethod
    def testKalmanFilterPlaneExampleXYLine():
        """Kalman filter test: plane example - XY line"""

        # Kalman filter test.
        testKalmanFilter.kalmanFilterPlaneExample("2L")

    @staticmethod
    def testKalmanFilterPlaneExampleUpDown():
        """Kalman filter test: plane example - Up-down"""

        # Kalman filter test.
        testKalmanFilter.kalmanFilterPlaneExample("UD")

    @staticmethod
    def testKalmanFilterPlaneExampleZigZag():
        """Kalman filter test: plane example - Zig-zag"""

        # Kalman filter test.
        testKalmanFilter.kalmanFilterPlaneExample("ZZ")

    @staticmethod
    def testKalmanFilterPlaneExampleCircle():
        """Kalman filter test: plane example - Circle"""

        # Kalman filter test.
        testKalmanFilter.kalmanFilterPlaneExample("Cc")

    @staticmethod
    def testKalmanFilterPlaneExampleRoundTrip():
        """Kalman filter test: plane example - Round trip"""

        # Kalman filter test.
        testKalmanFilter.kalmanFilterPlaneExample("RT")

    @staticmethod
    def testKalmanFilterPlaneExampleLooping():
        """Kalman filter test: plane example - Looping"""

        # Kalman filter test.
        testKalmanFilter.kalmanFilterPlaneExample("LP")

    @staticmethod
    def onTestEnd(ctrWin):
        """Callback on test end"""

        # Close controller.
        ctrWin.close()

    @staticmethod
    def kalmanFilterPlaneExample(expID, timeSec=5):
        """Kalman filter test: plane example"""

        # Create application and controls GUI.
        ctrWin = kalmanFilterController()
        ctrWin.comboEx.setCurrentText("plane tracking")
        for example in ctrWin.examples:
            if example.getName() == ctrWin.comboEx.currentText():
                for key in example.expGUI["QPB"]:
                    QTest.mouseClick(example.expGUI["QPB"][key], QtCore.Qt.LeftButton)
                QTest.mouseClick(example.expGUI["QRB"][expID], QtCore.Qt.LeftButton)
        QTest.mouseClick(ctrWin.updateBtn, QtCore.Qt.LeftButton)

        # Set timer to close controls GUI.
        timerCallback = functools.partial(testKalmanFilter.onTestEnd, ctrWin)
        timer = QtCore.QTimer()
        timer.timeout.connect(timerCallback)
        timer.start(timeSec*1000)
        app.exec()

# Main program.
if __name__ == '__main__':
    unittest.main(failfast=True)
