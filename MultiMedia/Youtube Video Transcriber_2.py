"""
YouTube Video Transcription GUI Application with Subtitle Generation

Modified by: Hamed Ghane
Last Modified: December 28, 2024

1. Updated OpenAI API format
2. Replace Google Translation with OpenAI

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

def translate_text(text: str, target_lang: str = 'fa', context: str = None) -> str:
    """
    Translate text using OpenAI's GPT model for subtitle translation.
    
    This function is designed to handle subtitle translation with special 
    consideration for timing, context, and natural language flow. It maintains
    the original meaning while ensuring the translation works well in subtitle format.
    
    Args:
        text: The subtitle text to translate
        target_lang: Target language code (default: 'fa' for Persian)
        context: Surrounding subtitle lines for contextual understanding
    
    Returns:
        Translated text formatted appropriately for subtitles
    """
    try:
        from sk import Hamedkey
        client = openai.OpenAI(api_key=Hamedkey)
        
        if target_lang == 'fa':
            system_prompt = """You are a professional subtitle translator working with English to Persian translation.
            Follow these subtitle translation principles:
            1. Keep translations concise and readable at subtitle speed
            2. Maintain natural conversational flow in the target language
            3. Preserve the original tone (formal/informal/humorous)
            4. Ensure translations fit well in subtitle format
            5. Consider cultural context while staying true to the original meaning
            6. Use appropriate Persian language conventions and punctuation
            7. Maintain consistency across connected dialogue"""
        else:
            raise ValueError(f"Unsupported target language: {target_lang}")
        
        # Provide context to help maintain conversation flow
        user_prompt = f"Translate this subtitle line into natural, conversational Persian:\n\nLine: {text}"
        if context:
            user_prompt += f"\n\nSurrounding context: {context}"
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,  # Keep relatively low for consistency
            max_tokens=200,   # Suitable for subtitle length
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        
        # Clean up the translation to ensure proper formatting
        translated_text = response.choices[0].message.content.strip()
        translated_text = ' '.join(translated_text.split())  # Normalize whitespace
        
        return translated_text
        
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
    """
    Creates and configures the main application window with all UI elements.
    
    This function sets up the entire GUI, including all widgets, layouts,
    and event handlers. It stores important widgets as window attributes
    to ensure they remain accessible throughout the application lifecycle.
    """
    # Create the main window
    window = QMainWindow()
    window.setWindowTitle('YouTube Audio Transcription')
    window.setGeometry(100, 100, 800, 600)
    
    # Create and store the central widget
    window._central_widget = QWidget()
    window.setCentralWidget(window._central_widget)
    main_layout = QVBoxLayout(window._central_widget)
    
    # Create and store the welcome label
    window._welcome_label = QLabel('YouTube Audio Transcription')
    window._welcome_label.setAlignment(Qt.AlignCenter)
    window._welcome_label.setStyleSheet('font-size: 16px; margin: 10px; font-weight: bold;')
    main_layout.addWidget(window._welcome_label)
    
    # Create and store the input layout
    input_layout = QHBoxLayout()
    window._url_input = QLineEdit()
    window._url_input.setPlaceholderText("Enter YouTube URL here...")
    input_layout.addWidget(window._url_input)
    
    window._language_combo = QComboBox()
    window._language_combo.addItems(["English", "Persian"])
    window._language_combo.setToolTip("Select transcription language")
    input_layout.addWidget(window._language_combo)
    main_layout.addLayout(input_layout)
    
    # Create and store the subtitle checkbox
    window._subtitle_checkbox = QCheckBox("Generate Subtitle File (.srt)")
    window._subtitle_checkbox.setChecked(True)
    window._subtitle_checkbox.setStyleSheet('margin: 5px;')
    main_layout.addWidget(window._subtitle_checkbox)
    
    # Create and store status label and progress bar
    window._status_label = QLabel('')
    window._status_label.setAlignment(Qt.AlignCenter)
    main_layout.addWidget(window._status_label)
    
    window._progress_bar = QProgressBar()
    window._progress_bar.setRange(0, 100)
    window._progress_bar.hide()
    main_layout.addWidget(window._progress_bar)
    
    # Create and store the transcript text area
    window._transcript_text = QTextEdit()
    window._transcript_text.setPlaceholderText("Transcription will appear here...")
    window._transcript_text.setReadOnly(True)
    main_layout.addWidget(window._transcript_text)
    
    # Create button layout
    button_layout = QHBoxLayout()
    
    # Create and store the transcribe button
    window._transcribe_btn = QPushButton('Transcribe')
    window._transcribe_btn.setStyleSheet('''
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
    
    # Create and store the exit button
    window._exit_btn = QPushButton('Exit')
    window._exit_btn.setFixedWidth(100)
    window._exit_btn.setStyleSheet('''
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
    
    button_layout.addWidget(window._transcribe_btn, alignment=Qt.AlignCenter)
    button_layout.addSpacing(40)
    button_layout.addWidget(window._exit_btn, alignment=Qt.AlignRight)
    button_layout.addSpacing(10)
    main_layout.addLayout(button_layout)
    
    # Define helper functions that use window attributes
    def update_progress(percent):
        """Updates the progress bar value and processes pending events."""
        window._progress_bar.setValue(int(percent))
        QApplication.processEvents()
    
    def update_status(message: str, color: str = 'black'):
        """Updates the status label with the given message and color."""
        window._status_label.setText(message)
        window._status_label.setStyleSheet(f'color: {color};')
        QApplication.processEvents()
    
    # Store helper functions as window attributes
    window.update_progress = update_progress
    window.update_status = update_status
    
    # Define the process_transcription function
    def process_transcription(video_id: str, target_language: str):
        """Handles the complete transcription process workflow."""
        audio_file = None
        try:
            update_status("Checking for YouTube transcript...", 'blue')
            
            try:
                # Try getting YouTube transcript first
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                transcript = transcript_list.find_transcript(['en'])
                transcript_entries = transcript.fetch()
                
                if target_language == "Persian":
                    update_status("Translating to Persian...", 'blue')
                    
                    # Translate each subtitle with context
                    translated_entries = []
                    for i, entry in enumerate(transcript_entries):
                        # Get context from surrounding entries
                        context_range = 2
                        start_idx = max(0, i - context_range)
                        end_idx = min(len(transcript_entries), i + context_range + 1)
                        
                        context = " ".join([
                            e['text'] for e in transcript_entries[start_idx:end_idx]
                            if transcript_entries.index(e) != i
                        ])
                        
                        # Update progress
                        progress = int((i + 1) * 100 / len(transcript_entries))
                        update_status(f"Translating subtitles... {progress}%", 'blue')
                        
                        # Translate with context
                        translated_text = translate_text(entry['text'], 'fa', context)
                        
                        translated_entries.append({
                            'text': translated_text,
                            'start': entry['start'],
                            'duration': entry.get('duration', 3),
                            'end': entry.get('end', entry['start'] + entry.get('duration', 3))
                        })
                    
                    # Update GUI with translated text
                    full_translated_text = "\n".join([entry['text'] for entry in translated_entries])
                    window._transcript_text.setText(full_translated_text)
                    
                    # Save subtitle file if requested
                    if window._subtitle_checkbox.isChecked():
                        subtitle_path = save_subtitle_file(video_id, translated_entries, target_language)
                        update_status(f"Transcript and subtitle file saved: {subtitle_path}", 'green')
                    else:
                        update_status("Translation completed!", 'green')
                else:
                    # For English, use original text
                    text = " ".join([entry['text'] for entry in transcript_entries])
                    window._transcript_text.setText(text)
                    
                    if window._subtitle_checkbox.isChecked():
                        subtitle_path = save_subtitle_file(video_id, transcript_entries, target_language)
                        update_status(f"Transcript and subtitle file saved: {subtitle_path}", 'green')
                    else:
                        update_status("Transcript retrieved successfully!", 'green')
                        
            except Exception as yt_error:
                # If YouTube transcript fails, use Whisper
                update_status("No transcript available. Downloading audio...", 'blue')
                window._progress_bar.show()
                
                success, result = download_audio(video_id, update_progress)
                
                if success:
                    audio_file = result
                    update_status("Transcribing with Whisper...", 'blue')
                    text, transcript_entries = whisper_transcribe(audio_file, target_language)
                    window._transcript_text.setText(text)
                    
                    if window._subtitle_checkbox.isChecked():
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
            
            window._progress_bar.hide()
            window._transcribe_btn.setEnabled(True)
    
    def start_transcription():
        """Initializes the transcription process after validation."""
        url = window._url_input.text().strip()
        if not url:
            update_status("Please enter a YouTube URL", 'red')
            return
        
        video_id = get_video_id(url)
        if not video_id:
            update_status("Invalid YouTube URL", 'red')
            return
        
        window._transcribe_btn.setEnabled(False)
        window._progress_bar.setValue(0)
        window._progress_bar.show()
        
        language = window._language_combo.currentText()
        QTimer.singleShot(100, lambda: process_transcription(video_id, language))
    
    def exit_application():
        """
        Safely exits the application with cleanup.
        Only removes temporary audio files, preserves SRT files.
        """
        try:
            # Clean up only temporary audio files
            docs_path = os.path.join(os.path.expanduser('~'), 'Documents')
            for file in os.listdir(docs_path):
                if file.startswith('audio_') and file.endswith('.mp3'):
                    try:
                        file_path = os.path.join(docs_path, file)
                        if os.path.exists(file_path):
                            os.remove(file_path)
                    except Exception as e:
                        logging.warning(f"Could not delete file {file}: {e}")
        except Exception as e:
            logging.warning(f"Error during cleanup: {e}")
        
        try:
            QApplication.closeAllWindows()
            QApplication.quit()
        except Exception as e:
            logging.error(f"Error during application exit: {e}")
            sys.exit(0)
    
    # Connect button signals to their handlers
    window._transcribe_btn.clicked.connect(start_transcription)
    window._exit_btn.clicked.connect(exit_application)
    window.closeEvent = lambda event: exit_application()
    
    return window

def main():
    """
    Main application entry point.
    
    This function initializes the Qt application, creates the main window,
    and starts the event loop. It provides comprehensive error handling
    and proper application lifecycle management.
    """
    try:
        # Create a new QApplication instance with empty arguments
        # This is more reliable than trying to reuse an existing instance
        app = QApplication([])
        
        # Create our main window and store it to prevent garbage collection
        window = create_window()
        
        # Make the window visible to the user
        window.show()
        
        # Start the Qt event loop and return its exit code
        # This keeps the application running until the user closes it
        return app.exec_()
        
    except Exception as e:
        # If something goes wrong, show an error message to the user
        # but only if we still have a working QApplication instance
        if QApplication.instance():
            QMessageBox.critical(None, "Error", 
                               f"An unexpected error occurred: {str(e)}")
        
        # Log the error for debugging purposes
        logging.error(f"Application error: {e}")
        return 1

# Application entry point
if __name__ == '__main__':
    sys.exit(main())
