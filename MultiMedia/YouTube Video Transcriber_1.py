"""
YouTube Video Transcription GUI Application with Subtitle Generation

Modified by: Hamed Ghane
Last Modified: December 21, 2024

This script creates a GUI application that:
1. Takes a YouTube URL from the user
2. Attempts to retrieve the video's transcript from YouTube directly
3. If no transcript is found, downloads the video's audio and uses OpenAI's Whisper model
4. Allows choosing between Persian or English transcription
5. Generates both transcript text and SRT subtitle files
6. Displays results in the GUI with progress and status updates

Additional requirements (same as original, plus):
- All original requirements still apply
- Outputs .srt subtitle files in the user's Documents folder
"""

# System and OS operations
import sys
import os
import openai
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

# PyQt5 imports for GUI components
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout,
    QWidget, QLabel, QTextEdit, QProgressBar, QLineEdit,
    QHBoxLayout, QComboBox, QMessageBox, QCheckBox
)
from PyQt5.QtCore import Qt, QTimer

# Additional functionality imports
from youtube_transcript_api import YouTubeTranscriptApi
import subprocess
import re
from googletrans import Translator
import time

# Configure logging
log_path = os.path.join(os.path.expanduser('~'), 'Documents', 'transcription_app.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()
    ]
)

def get_video_id(url: str) -> Optional[str]:
    """Extract YouTube video ID from various URL formats."""
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"(?:youtu\.be\/)([0-9A-Za-z_-]{11})",
        r"(?:embed\/)([0-9A-Za-z_-]{11})",
        r"(?:shorts\/)([0-9A-Za-z_-]{11})"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def create_srt_content(transcript_entries: List[Dict[str, Any]]) -> str:
    """
    Convert transcript entries to SRT subtitle format.
    
    Args:
        transcript_entries: List of transcript entries with 'text' and timing info
        
    Returns:
        Formatted SRT subtitle content as string
    """
    def format_time(seconds: float) -> str:
        """Convert seconds to SRT timestamp format: HH:MM:SS,mmm"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        milliseconds = int((seconds % 1) * 1000)
        seconds = int(seconds)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    srt_content = []
    for i, entry in enumerate(transcript_entries, 1):
        start = entry.get('start', 0)
        duration = entry.get('duration', 3)
        end = entry.get('end', start + duration)
        
        srt_entry = f"{i}\n{format_time(start)} --> {format_time(end)}\n{entry['text']}\n"
        srt_content.append(srt_entry)
    
    return "\n".join(srt_content)

def download_audio(video_id: str, progress_callback=None) -> Tuple[bool, str]:
    """Download audio from YouTube video using yt-dlp."""
    try:
        output_path = os.path.join(os.path.expanduser('~'), 'Documents', f'audio_{video_id}.mp3')
        
        command = [
            'yt-dlp',
            '--extract-audio',
            '--audio-format', 'mp3',
            '-o', output_path,
            '--newline',
            f'https://www.youtube.com/watch?v={video_id}'
        ]
        
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output and progress_callback:
                if '[download]' in output and '%' in output:
                    try:
                        percent = float(output.split('%')[0].split()[-1])
                        progress_callback(percent)
                    except (ValueError, IndexError):
                        pass
        
        if process.returncode == 0:
            return True, output_path
        return False, process.stderr.read()
        
    except Exception as e:
        return False, str(e)

def translate_text(text: str, target_lang: str = 'fa') -> str:
    """Translate text using Google Translate with chunking and retry logic."""
    try:
        translator = Translator()
        chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
        translated_chunks = []
        
        for chunk in chunks:
            for attempt in range(3):
                try:
                    translation = translator.translate(chunk, dest=target_lang)
                    translated_chunks.append(translation.text)
                    break
                except Exception as e:
                    if attempt == 2:
                        raise
                    time.sleep(1)
        
        return ' '.join(translated_chunks)
    except Exception as e:
        logging.error(f"Translation error: {e}")
        return f"Translation failed: {str(e)}"

def whisper_transcribe(audio_file_path: str, language: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Transcribe audio using OpenAI's Whisper API with timestamp information.
    
    Returns:
        Tuple containing full text transcript and list of timed segments
    """
    try:
        from sk import Hamedkey
        openai.api_key = Hamedkey
    except ImportError:
        raise ImportError("OpenAI API key not found. Please create sk.py with your API key.")
    
    lang_code = 'fa' if language == 'Persian' else 'en'
    
    for attempt in range(3):
        try:
            with open(audio_file_path, "rb") as audio:
                response = openai.Audio.transcribe(
                    model="whisper-1",
                    file=audio,
                    language=lang_code,
                    response_format="verbose_json"
                )
                
                full_text = response['text']
                segments = response['segments']
                
                transcript_entries = [
                    {
                        'text': segment['text'],
                        'start': segment['start'],
                        'end': segment['end']
                    }
                    for segment in segments
                ]
                
                return full_text, transcript_entries
                
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(1)

def save_subtitle_file(video_id: str, transcript_entries: List[Dict[str, Any]], language: str) -> str:
    """
    Save transcript entries as an SRT subtitle file.
    
    Returns:
        Path to the saved subtitle file
    """
    lang_code = 'fa' if language == 'Persian' else 'en'
    output_path = os.path.join(
        os.path.expanduser('~'),
        'Documents',
        f'subtitle_{video_id}_{lang_code}.srt'
    )
    
    srt_content = create_srt_content(transcript_entries)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(srt_content)
    
    return output_path

def create_window():
    """Create and configure the main application window with all UI elements."""
    window = QMainWindow()
    window.setWindowTitle('YouTube Audio Transcription')
    window.setGeometry(100, 100, 800, 600)
    
    central_widget = QWidget()
    window.setCentralWidget(central_widget)
    main_layout = QVBoxLayout(central_widget)
    
    welcome_label = QLabel('YouTube Audio Transcription')
    welcome_label.setAlignment(Qt.AlignCenter)
    welcome_label.setStyleSheet('font-size: 16px; margin: 10px; font-weight: bold;')
    main_layout.addWidget(welcome_label)
    
    input_layout = QHBoxLayout()
    
    url_input = QLineEdit()
    url_input.setPlaceholderText("Enter YouTube URL here...")
    input_layout.addWidget(url_input)
    
    language_combo = QComboBox()
    language_combo.addItems(["English", "Persian"])
    language_combo.setToolTip("Select transcription language")
    input_layout.addWidget(language_combo)
    main_layout.addLayout(input_layout)
    
    # Add subtitle generation checkbox
    subtitle_checkbox = QCheckBox("Generate Subtitle File (.srt)")
    subtitle_checkbox.setChecked(True)
    subtitle_checkbox.setStyleSheet('margin: 5px;')
    main_layout.addWidget(subtitle_checkbox)
    
    status_label = QLabel('')
    status_label.setAlignment(Qt.AlignCenter)
    main_layout.addWidget(status_label)
    
    progress_bar = QProgressBar()
    progress_bar.setRange(0, 100)
    progress_bar.hide()
    main_layout.addWidget(progress_bar)
    
    transcript_text = QTextEdit()
    transcript_text.setPlaceholderText("Transcription will appear here...")
    transcript_text.setReadOnly(True)
    main_layout.addWidget(transcript_text)
    
    button_layout = QHBoxLayout()
    
    transcribe_btn = QPushButton('Transcribe')
    transcribe_btn.setStyleSheet('''
        QPushButton {
            padding: 12px;
            font-size: 14px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            min-width: 200px;
        }
        QPushButton:hover {
            background-color: #45a049;
        }
        QPushButton:disabled {
            background-color: #cccccc;
            color: #666666;
        }
    ''')
    
    exit_btn = QPushButton('Exit')
    exit_btn.setFixedWidth(100)
    exit_btn.setStyleSheet('''
        QPushButton {
            padding: 8px;
            font-size: 12px;
            background-color: #f44336;
            color: white;
            border: none;
            border-radius: 4px;
            min-width: 80px;
        }
        QPushButton:hover {
            background-color: #d32f2f;
        }
    ''')
    
    button_layout.addWidget(transcribe_btn, alignment=Qt.AlignCenter)
    button_layout.addSpacing(40)
    button_layout.addWidget(exit_btn, alignment=Qt.AlignRight)
    button_layout.addSpacing(10)
    main_layout.addLayout(button_layout)
    
    def update_progress(percent):
        progress_bar.setValue(int(percent))
        QApplication.processEvents()
    
    def update_status(message: str, color: str = 'black'):
        status_label.setText(message)
        status_label.setStyleSheet(f'color: {color};')
        QApplication.processEvents()

    def process_transcription(video_id: str, target_language: str):
        """Handle the complete transcription process workflow."""
        audio_file = None
        try:
            update_status("Checking for YouTube transcript...", 'blue')
            
            try:
                # Try getting YouTube transcript first
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                transcript = transcript_list.find_transcript(['en'])
                transcript_entries = transcript.fetch()
                
                text = " ".join([entry['text'] for entry in transcript_entries])
                
                if target_language == "Persian":
                    update_status("Translating to Persian...", 'blue')
                    text = translate_text(text, 'fa')
                    
                    # Translate subtitle entries
                    for entry in transcript_entries:
                        entry['text'] = translate_text(entry['text'], 'fa')
                
                transcript_text.setText(text)
                
                # Generate subtitle file if requested
                if subtitle_checkbox.isChecked():
                    subtitle_path = save_subtitle_file(video_id, transcript_entries, target_language)
                    update_status(f"Transcript and subtitle file saved: {subtitle_path}", 'green')
                else:
                    update_status("Transcript retrieved successfully!", 'green')
                
            except Exception as yt_error:
                # If YouTube transcript fails, use Whisper
                update_status("No transcript available. Downloading audio...", 'blue')
                progress_bar.show()
                
                success, result = download_audio(video_id, update_progress)
                
                if success:
                    audio_file = result
                    update_status("Transcribing with Whisper...", 'blue')
                    text, transcript_entries = whisper_transcribe(audio_file, target_language)
                    transcript_text.setText(text)
                    
                    # Generate subtitle file if requested
                    if subtitle_checkbox.isChecked():
                        subtitle_path = save_subtitle_file(video_id, transcript_entries, target_language)
                        update_status(f"Transcription and subtitle file saved: {subtitle_path}", 'green')
                    else:
                        update_status("Transcription completed!", 'green')
                else:
                    raise Exception(f"Failed to download audio: {result}")
                    
        except Exception as e:
            update_status(f"Error: {str(e)}", 'red')
            logging.error(f"Transcription error: {e}")
        
        finally:
            if audio_file and os.path.exists(audio_file):
                try:
                    os.remove(audio_file)
                except Exception as e:
                    logging.warning(f"Could not delete audio file: {e}")
            
            progress_bar.hide()
            transcribe_btn.setEnabled(True)
    
    def start_transcription():
        """Initialize the transcription process after validation."""
        url = url_input.text().strip()
        if not url:
            update_status("Please enter a YouTube URL", 'red')
            return

        video_id = get_video_id(url)
        if not video_id:
            update_status("Invalid YouTube URL", 'red')
            return

        transcribe_btn.setEnabled(False)
        progress_bar.setValue(0)
        progress_bar.show()
        
        language = language_combo.currentText()
        QTimer.singleShot(100, lambda: process_transcription(video_id, language))
    def exit_application():
        """
        Safely exit the application with cleanup.
        
        Performs cleanup of temporary files and ensures proper application shutdown.
        """
        try:
            # Clean up any remaining temporary files
            docs_path = os.path.join(os.path.expanduser('~'), 'Documents')
            for file in os.listdir(docs_path):
                if (file.startswith('audio_') and file.endswith('.mp3')) or \
                   (file.startswith('subtitle_') and file.endswith('.srt')):
                    try:
                        file_path = os.path.join(docs_path, file)
                        if os.path.exists(file_path):
                            os.remove(file_path)
                    except Exception as e:
                        logging.warning(f"Could not delete file {file}: {e}")
        except Exception as e:
            logging.warning(f"Error during cleanup: {e}")
        
        try:
            # Properly close application
            QApplication.closeAllWindows()
            QApplication.quit()
        except Exception as e:
            logging.error(f"Error during application exit: {e}")
            sys.exit(0)
    
    # Connect button signals to their handlers
    transcribe_btn.clicked.connect(start_transcription)
    exit_btn.clicked.connect(exit_application)
    window.closeEvent = lambda event: exit_application()
    
    return window

def main():
    """
    Main application entry point.
    
    Initializes the QApplication, creates the main window, and starts the event loop.
    Handles unexpected errors gracefully with user notification.
    """
    try:
        # Initialize or get existing QApplication instance
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        
        # Create and display main window
        window = create_window()
        window.show()
        
        # Start event loop
        return app.exec_()
    except Exception as e:
        # Handle any unhandled exceptions
        QMessageBox.critical(None, "Error", f"An unexpected error occurred: {str(e)}")
        logging.error(f"Application error: {e}")
        return 1

# Application entry point
if __name__ == '__main__':
    sys.exit(main())
