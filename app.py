import sys
import os

os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = ''
os.environ.pop('QT_QPA_PLATFORM_PLUGIN_PATH', None)

from PyQt5 import QtWidgets, uic, QtCore, QtGui
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

import cv2 as cv
import numpy as np

try:
    import boto3
    from image import RekognitionImage
    AWS_AVAILABLE = True
except ImportError:
    print("Warning: AWS modules not available")
    AWS_AVAILABLE = False

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    print("Warning: pyttsx3 not available, text-to-speech disabled")
    TTS_AVAILABLE = False

from pprint import pprint


# Video capture thread
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    detection_signal = pyqtSignal(list)
    
    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.frame_counter = 0
        self.FRAME_SKIP = 60
        self.client = None
        self.aws_enabled = False
        
    def setup_aws(self, access_id, secret_key):
        """Setup AWS Rekognition client"""
        if not AWS_AVAILABLE:
            print("AWS modules not available")
            return False
            
        try:
            self.client = boto3.client(
                'rekognition',
                region_name='us-east-2',
                aws_access_key_id=access_id,
                aws_secret_access_key=secret_key
            )
            self.aws_enabled = True
            return True
        except Exception as e:
            print(f"AWS setup error: {e}")
            self.aws_enabled = False
            return False
    
    def send_to_AWS(self, img_bytes):
        """Send frame to AWS Rekognition for object detection"""
        if not self.aws_enabled or self.client is None or not AWS_AVAILABLE:
            return []
        
        try:
            identified_objects = []
            image_obj = RekognitionImage(
                {'Bytes': img_bytes}, 
                f"Frame {self.frame_counter}", 
                self.client
            )
            labels = image_obj.detect_labels(max_labels=10, min_confidence=55)
            for label in labels:
                identified_objects.append(label['Name'])
            return identified_objects
        except Exception as e:
            print(f"AWS detection error: {e}")
            return []
    
    def get_encode_image(self, frame):
        """Encode frame as JPEG bytes"""
        try:
            success, buffer = cv.imencode('.jpg', frame)
            if success:
                return buffer.tobytes()
        except Exception as e:
            print(f"Encoding error: {e}")
        return None
    
    def run(self):
        """Main video capture loop"""
        cap = cv.VideoCapture(0)
        
        if not cap.isOpened():
            print("Cannot open camera")
            return
        
        print("Video capture started")
        
        while self._run_flag:
            ret, frame = cap.read()
            if ret:
                # Emit frame for display
                self.change_pixmap_signal.emit(frame)
                
                # Process frame for object detection every N frames
                if self.aws_enabled and self.frame_counter % self.FRAME_SKIP == 0:
                    img_bytes = self.get_encode_image(frame)
                    if img_bytes:
                        detected = self.send_to_AWS(img_bytes)
                        if detected:
                            self.detection_signal.emit(detected)
                
                self.frame_counter += 1
            else:
                print("Failed to capture frame")
                break
        
        cap.release()
        print("Video capture stopped")
    
    def stop(self):
        """Stop the video thread"""
        self._run_flag = False
        self.wait()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        
        # Initialize text-to-speech engine
        self.tts_engine = None
        self.tts_enabled = False
        if TTS_AVAILABLE:
            try:
                self.tts_engine = pyttsx3.init(driverName='espeak')
                self.tts_engine.setProperty('rate', 150)  # Speed of speech
                self.tts_enabled = True
                print("Text-to-speech initialized")
            except Exception as e:
                print(f"TTS initialization error: {e}")
                self.tts_enabled = False
        
        # Load the UI file
        try:
            uic.loadUi('UI.ui', self)
            print("UI loaded successfully")
        except Exception as e:
            print(f"Error loading UI: {e}")
            import glob
            ui_files = glob.glob("*.ui")
            print("Available .ui files:")
            for f in ui_files:
                print(f"  - {f}")
            raise
        
        # Setup video thread
        self.video_thread = VideoThread()
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.detection_signal.connect(self.update_detections)
        
        # Connect buttons
        try:
            self.pushButton.clicked.connect(self.start_video)  # Video tab confirm
            self.pushButton_2.clicked.connect(self.confirm_audio)  # Audio tab confirm
            self.pushButton_3.clicked.connect(self.confirm_aws)  # AWS tab confirm
            self.pushButton_4.clicked.connect(self.toggle_settings)  # Settings button
        except AttributeError as e:
            print(f"Warning: Some buttons not found in UI: {e}")
        
        # Setup graphics view for video display
        self.scene = QtWidgets.QGraphicsScene()
        self.graphicsView.setScene(self.scene)
        
        # Initialize detection list
        self.detected_objects = []
        
        # Set window title
        self.setWindowTitle("See For Me - Object Detection")
        
        # Hide the tab widget initially (settings hidden by default)
        try:
            self.tabWidget.hide()
            self.settings_visible = False
        except:
            print("Warning: tabWidget not found")
            self.settings_visible = True
        
        # Auto-start video if checkbox is already checked
        try:
            if self.checkBox.isChecked():
                QTimer.singleShot(500, self.start_video)
        except:
            pass
        
        print("MainWindow initialized")
        
    def start_video(self):
        """Start video capture when confirm is clicked"""
        try:
            if self.checkBox.isChecked():
                self.video_thread.start()
                try:
                    self.kled.setState(1)  # Turn on video LED
                except:
                    print("Warning: kled widget not available")
                print("Video started")
            else:
                QtWidgets.QMessageBox.warning(
                    self, "Warning", "Please check 'Ready' before starting video"
                )
        except Exception as e:
            print(f"Error starting video: {e}")
    
    def update_image(self, cv_img):
        """Update the video frame in the graphics view"""
        try:
            # Convert CV image to Qt format
            rgb_image = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # Scale to fit graphics view while maintaining aspect ratio
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(
                self.graphicsView.width() - 2,
                self.graphicsView.height() - 2,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            
            # Display in graphics view
            self.scene.clear()
            self.scene.addPixmap(scaled_pixmap)
            self.graphicsView.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        except Exception as e:
            print(f"Error updating image: {e}")
    
    def update_detections(self, detected):
        """Update UI with detected objects"""
        try:
            self.detected_objects = detected
            
            # Update text browsers with detection info
            if detected:
                # Show first detected item (or count if multiple)
                top_item = detected[0]
                if len(detected) == 1:
                    item_text = f"Item: {top_item}"
                else:
                    item_text = f"Items: {len(detected)} objects detected"
                self.textBrowser.setPlainText(item_text)
                
                # Show all detected items in description with better formatting
                description = ", ".join(detected)
                full_description = f"Detected: {description}"
                self.textBrowser_2.setPlainText(full_description)
                
                # QTextBrowser has word wrap enabled by default, no need to set it
                
                # Announce the top detected item via text-to-speech
                if self.tts_enabled and self.tts_engine:
                    try:
                        announcement = f"I see {top_item}"
                        print(f"Announcing: {announcement}")
                        # Run TTS in a separate thread to avoid blocking
                        self.tts_engine.say(announcement)
                        self.tts_engine.runAndWait()
                    except Exception as e:
                        print(f"TTS error: {e}")
                
            print(f"Detected objects: {detected}")
        except Exception as e:
            print(f"Error updating detections: {e}")
    
    def confirm_aws(self):
        """Setup AWS credentials"""
        try:
            if not AWS_AVAILABLE:
                QtWidgets.QMessageBox.warning(
                    self, "Error", "AWS modules not available. Install boto3 and image.py"
                )
                return
                
            access_id = self.lineEdit.text()
            secret_key = self.lineEdit_2.text()
            
            if access_id and secret_key:
                success = self.video_thread.setup_aws(access_id, secret_key)
                if success:
                    try:
                        self.kled_3.setState(1)  # Turn on AWS LED
                    except:
                        print("Warning: kled_3 widget not available")
                    print("AWS configured successfully")
                    QtWidgets.QMessageBox.information(
                        self, "Success", "AWS Rekognition configured successfully!"
                    )
                else:
                    QtWidgets.QMessageBox.warning(
                        self, "Error", "Failed to configure AWS. Check credentials."
                    )
            else:
                QtWidgets.QMessageBox.warning(
                    self, "Warning", "Please enter both Access ID and Secret Key"
                )
        except Exception as e:
            print(f"Error confirming AWS: {e}")
            QtWidgets.QMessageBox.critical(
                self, "Error", f"AWS configuration error: {str(e)}"
            )
    
    def confirm_audio(self):
        """Confirm audio settings"""
        try:
            if self.checkBox_3.isChecked() and self.checkBox_4.isChecked():
                try:
                    self.kled_2.setState(1)  # Turn on audio LED
                except:
                    print("Warning: kled_2 widget not available")
                
                # Enable TTS if available
                if TTS_AVAILABLE and self.tts_engine:
                    self.tts_enabled = True
                    # Test announcement
                    self.tts_engine.say("Audio configured")
                    self.tts_engine.runAndWait()
                    print("Audio configured with TTS enabled")
                else:
                    print("Audio configured but TTS not available")
            else:
                QtWidgets.QMessageBox.warning(
                    self, "Warning", "Please check both Device Output and Speakers"
                )
        except Exception as e:
            print(f"Error confirming audio: {e}")
    
    def toggle_settings(self):
        """Toggle settings panel visibility"""
        try:
            if self.settings_visible:
                self.tabWidget.hide()
                self.settings_visible = False
                self.pushButton_4.setText("Settings")
                print("Settings hidden")
            else:
                self.tabWidget.show()
                self.settings_visible = True
                self.pushButton_4.setText("Hide Settings")
                print("Settings shown")
        except Exception as e:
            print(f"Error toggling settings: {e}")
            # Fallback to dialog if tabWidget issues
            self.open_settings()
    
    def open_settings(self):
        """Open settings dialog (fallback method)"""
        QtWidgets.QMessageBox.information(
            self, "Settings", "Settings dialog - implement as needed"
        )
    
    def closeEvent(self, event):
        """Clean up when closing the application"""
        print("Closing application...")
        self.video_thread.stop()
        event.accept()


def main():
    # Create application
    app = QtWidgets.QApplication(sys.argv)
    
    print("Starting See For Me application...")
    print(f"Qt Platform: {app.platformName()}")
    
    # Create and show window
    try:
        window = MainWindow()
        window.show()
        print("Window displayed")
    except Exception as e:
        print(f"Error creating window: {e}")
        return 1
    
    # Run application
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main())