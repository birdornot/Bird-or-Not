# Bird or Not
## Get requirements file
# pipreqs .
# pip3 freeze > requirements.txt

## Terminal: convert python to exe file (add python to PATH first)
# pip install pyinstaller               # Bundles a Python application and all its dependencies into a single package.
# flags: --onefile (Onefile result); -w (without terminal); --add-data ({source}{os_separator}{destination})
# pyinstaller --onedir -w --clean --add-data="BD_factors.npy;." --add-data="form.ui;." --add-data="BD_weights.h5;." --add-data="Birds.png;." --add-data="Archaeopteryx.png;." --add-data C:/Users/inda7/AppData/Local/Programs/Python/Python310/Lib/site-packages/librosa/util/example_data;librosa/util/example_data --exclude-module torch --noconfirm Bird_or_Not.py
# Remove from shared drive for fast startup

import matplotlib.widgets
from PyQt6 import QtWidgets, uic, QtCore, QtGui
import sys
import os
import Bird_Detector

matplotlib.use('Qt5Agg')
absolute_path = "C:/Users/inda7/Documents/Audio Files/Agamon Hula 2021/"
relative_path = "Tag/"
file_path = absolute_path + relative_path
print("Base path: " + file_path)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        uic.loadUi("form.ui", self)
        self.browse_button = self.findChild(QtWidgets.QPushButton)
        self.label = self.findChild(QtWidgets.QLabel, "label")
        self.label.setPixmap(QtGui.QPixmap("Birds.png"))
        self.setWindowIcon(QtGui.QIcon('Archaeopteryx.png'))
        self.progress = self.findChild(QtWidgets.QProgressBar, "progressBar")
        self.progress.setVisible(False)

        self.browse_button.clicked.connect(self.browse)
        self.show()

    def browse(self):
        filepath, _ = QtWidgets.QFileDialog.getOpenFileNames(None, "Choose audio files", file_path, "WAV (*.wav) ;; MP3 (*.mp3)")

        if len(filepath) > 0:
            print(filepath)
            self.filepath = filepath
            self.progress.setVisible(True)
            self.worker = WorkerThread(filepath)
            self.worker.start()
            self.worker.finished.connect(self.evt_worker_finished)
            self.worker.update_progress.connect(self.evt_progress_bar)

    def evt_worker_finished(self):
        self.progress.setValue(100)
        self.browse_button.setText("What a thrill! \nPredictions exported to input path. Click to detect again.")
        QtCore.QCoreApplication.processEvents()

    def evt_progress_bar(self, val):
        file = os.path.splitext(os.path.basename(self.filepath[val]))[0]
        print(file)
        print("Progress is: ", val)
        self.progress.setValue(int(100*round(val/len(self.filepath),2)))
        self.browse_button.setText("Detecting birds in\n" + file)

class WorkerThread(QtCore.QThread):
    update_progress = QtCore.pyqtSignal(int)

    def __init__(self, filepath):
        QtCore.QThread.__init__(self)
        self.filepath = filepath

    def run(self):
        print("Starting bird detection for ", self.filepath)
        model, M, S = Bird_Detector.load_BD_model()
        for i, file in enumerate(self.filepath):
            print("Detecting birds in ", file)
            self.update_progress.emit(i)
            Bird_Detector.BD(file, model = model, M=M, S=S)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    app.exec()
