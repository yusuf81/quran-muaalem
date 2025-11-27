import sys
import os
import numpy as np
import librosa
import torch

# Add src to path
sys.path.append("/root/quran-muaalem/src")

from quran_muaalem.gradio_app import preprocess_waveform, process_multi_verse_audio, load_segmenter

def test_silence_trimming():
    print("Testing Silence Trimming...")
    sr = 16000
    # Create a dummy wave: 2s silence + 1s tone + 2s silence
    silence = np.zeros(2 * sr)
    tone = np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr))
    wave = np.concatenate([silence, tone, silence])
    
    # Process
    processed_wave, _ = preprocess_waveform(wave, sr, enable_preprocess=True, enable_debug=False)
    
    # Expected: ~1s silence + 1s tone + ~1s silence = ~3s
    # The original trim would remove all silence, leaving 1s.
    # Our new logic adds 1s padding.
    # So we expect roughly 1s + 1s + 1s = 3s.
    
    duration = len(processed_wave) / sr
    print(f"Original Duration: {len(wave)/sr}s")
    print(f"Processed Duration: {duration}s")
    
    if 2.8 <= duration <= 3.2:
        print("✅ Silence Trimming Test Passed")
    else:
        print("❌ Silence Trimming Test Failed")

def test_multi_verse_processing():
    print("\nTesting Multi-Verse Processing...")
    # Use existing asset
    audio_path = "/root/quran-muaalem/assets/test.mp3"
    
    if not os.path.exists(audio_path):
        print(f"❌ Audio file not found: {audio_path}")
        return

    # Mock inputs
    sura_idx = 1 # Al-Fatiha
    start_aya = 1
    end_aya = 7
    full_sura_toggle = False
    enable_preprocess = True
    
    # Run process
    try:
        output = process_multi_verse_audio(
            audio_path,
            sura_idx,
            start_aya,
            end_aya,
            full_sura_toggle,
            enable_preprocess
        )
        
        if "Hasil Analisis" in output:
             print("✅ Multi-Verse Processing Test Passed (Output generated)")
             # print(output[:500]) # Print first 500 chars
        else:
             print("❌ Multi-Verse Processing Test Failed (Unexpected output)")
             print(output)
             
    except Exception as e:
        print(f"❌ Multi-Verse Processing Test Failed with Exception: {e}")

if __name__ == "__main__":
    test_silence_trimming()
    test_multi_verse_processing()
