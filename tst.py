#!/usr/bin/env python3

"""Kalman filter tests"""

import unittest

import functools

import h5py
import numpy as np

from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest

from kalmanFilterController import kalmanFilterController

app = QApplication([])

class kalmanFilterTestCase(unittest.TestCase):
    """Kalman filter unittest class"""

    def testKalmanFilterPlaneExampleXYZLine(self):
        """Kalman filter test: plane example - XYZ line"""

        # Kalman filter test.
        self.kalmanFilterPlaneExample("3L")

    def testKalmanFilterPlaneExampleXYLine(self):
        """Kalman filter test: plane example - XY line"""

        # Kalman filter test.
        self.kalmanFilterPlaneExample("2L")

    def testKalmanFilterPlaneExampleUpDown(self):
        """Kalman filter test: plane example - Up-down"""

        # Kalman filter test.
        self.kalmanFilterPlaneExample("UD")

    def testKalmanFilterPlaneExampleZigZag(self):
        """Kalman filter test: plane example - Zig-zag"""

        # Kalman filter test.
        self.kalmanFilterPlaneExample("ZZ")

    def testKalmanFilterPlaneExampleCircle(self):
        """Kalman filter test: plane example - Circle"""

        # Kalman filter test.
        self.kalmanFilterPlaneExample("Cc")

    def testKalmanFilterPlaneExampleRoundTrip(self):
        """Kalman filter test: plane example - Round trip"""

        # Kalman filter test.
        self.kalmanFilterPlaneExample("RT")

    def testKalmanFilterPlaneExampleLooping(self):
        """Kalman filter test: plane example - Looping"""

        # Kalman filter test.
        self.kalmanFilterPlaneExample("LP")

    @staticmethod
    def onTestEnd(ctrWin):
        """Callback on test end"""

        # Close controller.
        ctrWin.close()

    def kalmanFilterPlaneExample(self, expID, timeSec=5):
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
        timerCallback = functools.partial(kalmanFilterTestCase.onTestEnd, ctrWin)
        timer = QtCore.QTimer()
        timer.timeout.connect(timerCallback)
        timer.start(timeSec*1000)
        app.exec()

        # Verify test.
        self.verifyTest(expID)

    def verifyTest(self, expID):
        """Verify test"""
        # Check test results.
        fdsRef = h5py.File("kalmanFilterModel.%s.h5"%expID, "r")
        fdsGen = h5py.File("kalmanFilterModel.h5", "r")
        for name in ["X", "Y", "Z"]:
            for key in fdsRef[name]:
                rowRef = fdsRef[name][key][:]
                rowExp = fdsGen[name][key][:]
                tstOK = np.isclose(rowRef, rowExp).all()
                if not tstOK:
                    tstOK = np.isclose(rowRef, rowExp)
                    for idx, tst in enumerate(tstOK):
                        if not tst:
                            print("Check test KO - %s, %s, idx %s -"%(name, key, idx),
                                  fdsRef[name]["T"][idx], rowRef[idx], rowExp[idx], flush=True)
                            self.assertTrue(tst)
                        else:
                            print("Check test OK - %s, %s, idx %s -"%(name, key, idx),
                                  fdsRef[name]["T"][idx], rowRef[idx], rowExp[idx], flush=True)
        fdsRef.close()
        fdsGen.close()

# Main program.
if __name__ == '__main__':
    unittest.main(failfast=True)
