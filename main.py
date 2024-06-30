import sys
import icon_rc
from PyQt5 import QtWidgets
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QFont
import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import *

# Placeholder for class labels, replace it with your actual class labels
class_list = ['person', 'car', 'bus', ...]

class ShowImage(QtWidgets.QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi('home.ui', self)
        self.gambar = None
        self.Image = None
        self.homeButton.clicked.connect(self.home)
        self.aboutButton.clicked.connect(self.about)
        self.uploadButton.clicked.connect(self.detect_screen)

    def home(self):
        home_window_instance = AboutWindow()
        widget.addWidget(home_window_instance)
        widget.setCurrentWidget(home_window_instance)

    def about(self):
        about_window_instance = AboutWindow()
        widget.addWidget(about_window_instance)
        widget.setCurrentWidget(about_window_instance)

    def detect_screen(self):
        fname, filter = QFileDialog.getOpenFileName(self, 'Open File', 'C:\\User\\', "Video Files(*.mp4)")
        detect_window_instance = DetectWindow(fname)
        widget.addWidget(detect_window_instance)
        widget.setCurrentWidget(detect_window_instance)
class AboutWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(AboutWindow, self).__init__()
        loadUi('about.ui', self)

class HomeWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(HomeWindow, self).__init__()
        loadUi('home.ui', self)


class DetectWindow(QtWidgets.QMainWindow):
    def __init__(self, video_path):
        super(DetectWindow, self).__init__()
        loadUi('detect.ui', self)
        self.video_path = video_path

        self.cap = cv2.VideoCapture(video_path)
        self.model = YOLO('yolov8s.pt')
        self.tracker = Tracker()

        self.label_video = self.findChild(QtWidgets.QLabel, 'label_video')  # Assuming the QLabel is named 'label_video' in detect.ui
        self.output_label = self.findChild(QtWidgets.QLabel, 'output_label')  # Assuming the QLabel is named 'output_label' in detect.ui

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.detect_and_display)
        self.timer.start(30)  # Update every 30 milliseconds

    def detect_and_display(self):
        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            return

        frame = cv2.resize(frame, (1020, 500))

        results = self.model.predict(frame)
        a = results[0].boxes.data
        px = pd.DataFrame(a).astype("float")
        bbox_list = []

        for index, row in px.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            d = int(row[5])

            if d < len(class_list):
                c = class_list[d]
                if 'person' in c:
                    bbox_list.append([x1, y1, x2, y2])

        bbox_id = self.tracker.update(bbox_list)
        detected_people = len(bbox_id)

        for i in range(detected_people):
            x1, y1, x2, y2, _ = bbox_id[i]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            cv2.circle(frame, (cx, cy), 4, (255, 0, 255), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Deteksi kerumunan
        for i in range(len(bbox_list)):
            num_overlapping_boxes = 0

            for j in range(len(bbox_list)):
                if i != j:
                    x1_i, y1_i, x2_i, y2_i = bbox_list[i]
                    x1_j, y1_j, x2_j, y2_j = bbox_list[j]

                    overlap_area = max(0, min(x2_i, x2_j) - max(x1_i, x1_j)) * max(0, min(y2_i, y2_j) - max(y1_i, y1_j))

                    if overlap_area > 0:
                        num_overlapping_boxes += 1

            threshold = 5

            if num_overlapping_boxes >= threshold:
                cx, cy = (int((x1_i + x2_i) / 2), int((y1_i + y2_i) / 2))
                cv2.putText(frame, "CROWD!!!", (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.rectangle(frame, (x1_i - 100, y1_i - 100), (x2_i + 100, y2_i + 100), (255, 0, 0), 2)


        # Convert the frame to QImage
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)

        # Convert QImage to QPixmap and set it to the QLabel
        pixmap = QPixmap.fromImage(q_image)
        self.label_video.setPixmap(pixmap)

        # Update QLabel for crowd count
        self.output_label.setStyleSheet("background-color: #1c6d66; border-radius: 20px; color: white; font-size: 16px;")
        self.output_label.setText(f"{detected_people}")

    def closeEvent(self, event):
        self.cap.release()
        cv2.destroyAllWindows()
        event.accept()

app = QtWidgets.QApplication(sys.argv)
widget = QtWidgets.QStackedWidget()
window = ShowImage()
widget.setWindowTitle('Deteksi Kerumunan')
window.show()
widget.addWidget(window)  # Add the window to the widget stack
widget.show()
widget.setFixedHeight(857)
widget.setFixedWidth(1266)
sys.exit(app.exec_())
