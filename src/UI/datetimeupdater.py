import datetime
from PyQt5.QtCore import QThread, pyqtSignal

class TimeUpdateThread(QThread):
    DateUpdate = pyqtSignal(str)
    ThreadActive = False

    def run(self):
        self.ThreadActive = True

        while (self.ThreadActive):
            currentTime = datetime.datetime.now()
            self.DateUpdate.emit(currentTime.strftime('%a %d %b %Y, %I:%M%p'))
            self.sleep(1)

    def stop(self):
        self.ThreadActive = False
        self.quit()
