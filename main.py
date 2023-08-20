import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow

os.environ['QT_QPA_PLATFORM'] = 'windows'

print(sys.argv)
app = QApplication(sys.argv)

window = QMainWindow()

desktop = QApplication.desktop()
screen_rect = desktop.availableGeometry()
window.resize(int(screen_rect.width() * 0.8), int(screen_rect.height() * 0.8))

window.show()
sys.exit(app.exec_())
