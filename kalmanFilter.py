#!/usr/bin/env python3

"""Kalman filter MVC (Model-View-Controller)"""

import sys
from PyQt5.QtWidgets import QApplication

from kalmanFilterController import kalmanFilterController

# Main program.
if __name__ == "__main__":
    # Check for python3.
    assert sys.version_info.major == 3, "this script is a python3 script."

    # Create application and controls GUI.
    app = QApplication(sys.argv)
    ctrWin = kalmanFilterController()
    ctrWin.showMaximized()

    # End main program.
    sys.exit(app.exec_())
