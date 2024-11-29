import sys
import openai
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QFileDialog, 
                           QVBoxLayout, QWidget, QLabel, QTextEdit, QProgressBar)
from PyQt5.QtCore import Qt, QTimer
from sk import Hamedkey

def create_window():
    # Create the main window
    window = QMainWindow()
    window.setWindowTitle('Audio Transcription')
    window.setGeometry(100, 100, 600, 400)
    
    # Create central widget and layout
    central_widget = QWidget()
    window.setCentralWidget(central_widget)
    layout = QVBoxLayout(central_widget)
    
    # Add welcome label
    welcome_label = QLabel('Welcome to Audio Transcription App')
    welcome_label.setAlignment(Qt.AlignCenter)
    welcome_label.setStyleSheet('font-size: 16px; margin: 10px;')
    layout.addWidget(welcome_label)
    
    # Add status label
    status_label = QLabel('')
    status_label.setAlignment(Qt.AlignCenter)
    layout.addWidget(status_label)
    
    # Add progress bar (initially hidden)
    progress_bar = QProgressBar()
    progress_bar.setRange(0, 0)  # Infinite progress bar
    progress_bar.hide()
    layout.addWidget(progress_bar)
    
    # Add text area for transcription
    transcription_text = QTextEdit()
    transcription_text.setPlaceholderText("Transcription will appear here...")
    transcription_text.setReadOnly(True)
    layout.addWidget(transcription_text)
    
    # Create upload button with styling
    upload_btn = QPushButton('Upload Audio File')
    upload_btn.setStyleSheet('''
        QPushButton {
            padding: 10px;
            font-size: 14px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            min-width: 150px;
        }
        QPushButton:hover {
            background-color: #45a049;
        }
        QPushButton:disabled {
            background-color: #cccccc;
            color: #666666;
        }
    ''')
    layout.addWidget(upload_btn, alignment=Qt.AlignCenter)
    
    def transcribe_audio(file_path):
        try:
            client = openai.OpenAI(api_key=Hamedkey)
            audio_file = Path(file_path)
            
            with audio_file.open("rb") as audio:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio,
                    response_format="text"
                )
            
            # Update UI after transcription
            progress_bar.hide()
            transcription_text.setText(transcript)
            status_label.setText("Transcription completed!")
            status_label.setStyleSheet('color: green;')
            upload_btn.setEnabled(True)
            
        except Exception as e:
            progress_bar.hide()
            status_label.setText(f"Error: {str(e)}")
            status_label.setStyleSheet('color: red;')
            upload_btn.setEnabled(True)
    
    def upload_audio():
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter("Audio Files (*.mp3 *.wav *.m4a *.ogg);;All Files (*.*)")
        
        if file_dialog.exec_():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                file_path = selected_files[0]
                
                # Disable upload button and show processing state
                upload_btn.setEnabled(False)
                status_label.setText("Transcription in progress... Please wait.")
                status_label.setStyleSheet('color: blue;')
                progress_bar.show()
                
                # Use QTimer to process in the next event loop iteration
                QTimer.singleShot(100, lambda: transcribe_audio(file_path))
    
    # Connect button to upload function
    upload_btn.clicked.connect(upload_audio)
    
    return window

def main():
    # Check if QApplication already exists
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    window = create_window()
    window.show()
    
    # Use a different approach for Jupyter vs regular Python
    if 'ipykernel' in sys.modules:
        # We're in Jupyter
        return window
    else:
        # We're in regular Python
        sys.exit(app.exec_())

if __name__ == '__main__':
    main()