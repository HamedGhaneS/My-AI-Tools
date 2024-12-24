"""
Audio Transcription GUI Application
Created by: Hamed Ghane
Date: November 29, 2024

This script creates a GUI application for transcribing audio files to text using OpenAI's Whisper model.
It provides a user-friendly interface with progress indication and status updates.

Requirements:
- PyQt5 (pip install PyQt5)
- openai (pip install openai)

API Key Setup:
1. Option 1 (Recommended for GitHub): 
   - Create a separate file (e.g., 'sk.py')
   - Add your OpenAI API key: HamedKey = "your-api-key-here"
   - Import it as: from sk import Hamedkey
   
2. Option 2 (Direct usage):
   - Replace 'Hamedkey' below with your actual OpenAI API key
   
"""

import sys
import openai
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QFileDialog, 
                           QVBoxLayout, QWidget, QLabel, QTextEdit, QProgressBar)
from PyQt5.QtCore import Qt, QTimer
from sk import Hamedkey  # Replace this with your API key import method

def create_window():
    """
    Creates and configures the main application window with all UI elements.
    Returns the configured window object.
    """
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
    
    # Add status label for showing process updates
    status_label = QLabel('')
    status_label.setAlignment(Qt.AlignCenter)
    layout.addWidget(status_label)
    
    # Add a progress bar (initially hidden) to show the processing state
    progress_bar = QProgressBar()
    progress_bar.setRange(0, 0)  # Infinite progress bar
    progress_bar.hide()
    layout.addWidget(progress_bar)
    
    # Add text area for displaying transcription results
    transcription_text = QTextEdit()
    transcription_text.setPlaceholderText("Transcription will appear here...")
    transcription_text.setReadOnly(True)
    layout.addWidget(transcription_text)
    
    # Create and style upload button
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
        """
        Handles the audio transcription process using OpenAI's Whisper model.
        Updates the UI with results or error messages.
        """
        try:
            # Initialize OpenAI client with API key
            client = openai.OpenAI(api_key=Hamedkey)  # Replace 'Hamedkey' with your API key variable
            audio_file = Path(file_path)
            
            # Perform transcription
            with audio_file.open("rb") as audio:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio,
                    response_format="text"
                )
            
            # Update UI with successful transcription
            progress_bar.hide()
            transcription_text.setText(transcript)
            status_label.setText("Transcription completed!")
            status_label.setStyleSheet('color: green;')
            upload_btn.setEnabled(True)
            
        except Exception as e:
            # Handle and display any errors
            progress_bar.hide()
            status_label.setText(f"Error: {str(e)}")
            status_label.setStyleSheet('color: red;')
            upload_btn.setEnabled(True)
    
    def upload_audio():
        """
        Handles the file selection process and initiates transcription.
        Shows progress indicators and updates UI state.
        """
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter("Audio Files (*.mp3 *.wav *.m4a *.ogg);;All Files (*.*)")
        
        if file_dialog.exec_():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                file_path = selected_files[0]
                
                # Update UI to show processing state
                upload_btn.setEnabled(False)
                status_label.setText("Transcription in progress... Please wait.")
                status_label.setStyleSheet('color: blue;')
                progress_bar.show()
                
                # Use QTimer to prevent UI freezing
                QTimer.singleShot(100, lambda: transcribe_audio(file_path))
    
    # Connect button to upload function
    upload_btn.clicked.connect(upload_audio)
    
    return window

def main():
    """
    Initializes and runs the application.
    """
    app = QApplication(sys.argv)
    window = create_window()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
