from piano_transcription_inference import PianoTranscription, sample_rate, load_audio
import glob
import os

def transcribe(audio_path, output_midi_path):
    # Load audio
    audio, _ = load_audio(audio_path, sr=sample_rate, mono=True)

    # Transcriptor
    transcriptor = PianoTranscription(device='cuda', checkpoint_path=None)

    # Transcribe and write out to MIDI file
    transcriptor.transcribe(audio, output_midi_path)


files = glob.glob('/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/audio/lmd_matched_mp3/**/*.mp3', recursive=True)

for file in files:

    dir = os.path.dirname(file.replace("/audio/lmd_matched_mp3", "/midi/audio2midi"))
    os.makedirs(dir, exist_ok=True)

    output = file.replace("/audio/lmd_matched_mp3", "/midi/audio2midi").replace(".mp3", ".mid")
    transcribe(file, output)