import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import torch
from ml_project import SimpleCNN, load_and_clean_audio, generate_spectrogram, SpectrogramDataset

class VoiceRecognitionApp(QWidget):
    def __init__(self, model, device):
        super().__init__()
        self.model = model
        self.device = device
        self.recording = False
        self.audio_data = []

        # Create UI elements
        self.label = QLabel("Press 'Record' to start recording", self)
        self.record_button = QPushButton('Record', self)
        self.stop_button = QPushButton('Stop', self)
        self.upload_button = QPushButton('Upload WAV File', self)

        # Connect buttons to their actions
        self.record_button.clicked.connect(self.start_recording)
        self.stop_button.clicked.connect(self.stop_recording)
        self.upload_button.clicked.connect(self.upload_wav_file)

        # Set up layout
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.record_button)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.upload_button)
        self.setLayout(layout)

    def start_recording(self):
        self.recording = True
        self.audio_data = [] 
        self.label.setText("Recording...")

        def callback(indata, frames, time, status):
          if status:
            print(f"Recording error: {status}")
          if self.recording:
            self.audio_data.append(indata.copy())  # audio data real-time

        self.stream = sd.InputStream(callback=callback, channels=1, samplerate=16000)
        self.stream.start() 

    def stop_recording(self):
      self.recording = False
      self.stream.stop()  # stop the stream
      self.stream.close()  # close the stream

      self.label.setText("Processing the recording...")

     # audio to numpy
      audio_data_np = np.concatenate(self.audio_data, axis=0)

      # saving audio wav for preprocessing
      wav.write("recorded_audio.wav", 16000, audio_data_np.astype(np.float32))

    # Process the recorded audio file
      predicted_class = self.process_audio_file("recorded_audio.wav")
      self.label.setText(f"Predicted class is {predicted_class}")
      self.audio_data = [] 

    def upload_wav_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Upload WAV File", "", "WAV Files (*.wav)", options=options)
        if file_path:
            self.label.setText(f"Processing {file_path}...")
            predicted_class = self.process_audio_file(file_path)
            self.label.setText(f"Predicted class is {predicted_class}")

    def process_audio_file(self, file_path):
      """Process the audio file and return the predicted class."""
    # Load and clean using fucntions from ml_project.py for consistency
      y, sr = load_and_clean_audio(file_path, sr=16000, trim_silence=True, cut_length=10)
      if y is None:
        return "Error: Unable to load audio"

    # Generate spectograms using fucntions from ml_project.py for consistency
      spectrogram = generate_spectrogram(y, sr=sr, n_mels=64)
      if spectrogram is None:
        return "Error: Unable to generate spectrogram"

    # Making the dimensions [1, 1, height, width], as CNN requires
      spectrogram = np.expand_dims(spectrogram, axis=0)  # add first conv1 dimensions (1, 64, 313)
      spectrogram = np.expand_dims(spectrogram, axis=0)  # add second conv2 dimension (1, 1, 64, 313)

    # Spectogram -> tensor, and no DataLoader because there are no batches anymore
      spectrogram_tensor = torch.tensor(spectrogram, dtype=torch.float32)
      spectrogram_tensor = spectrogram_tensor.to(self.device)

    # Model work work work work work -> prediction
      self.model.eval()
      with torch.no_grad():
        output = self.model(spectrogram_tensor)
        _, predicted_class = torch.max(output, 1)
        return predicted_class.item()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # Load your trained model here
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN()  # Use the same model structure used in training
    model_path = '/Users/sunsosun/Desktop/ML_DEPLOY/best_model.pth'
    model = torch.load(model_path)
    model.eval() 

    # Initialize the app
    window = VoiceRecognitionApp(model, device)
    window.setWindowTitle('Voice Recognition')
    window.resize(300, 200)
    window.show()

    sys.exit(app.exec_())
# F1, F7, F8, M3, M6, M8