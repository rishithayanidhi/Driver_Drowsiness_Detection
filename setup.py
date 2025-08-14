#!/usr/bin/env python3
"""
Complete Setup Script for Driver Drowsiness Detection System
Run with: python setup.py
"""

import os
import sys
import subprocess
import urllib.request
import zipfile
import shutil
from pathlib import Path

def run_command(command, description=""):
    """Execute shell command with error handling"""
    print(f"\n{'='*50}")
    print(f"🔧 {description}")
    print(f"{'='*50}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print("✅ SUCCESS!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False

def detect_gpu():
    """Detect GPU and determine appropriate PyTorch installation"""
    print("\n🔍 Detecting GPU...")
    
    try:
        # Check for NVIDIA GPU
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected!")
            # Get CUDA version
            try:
                cuda_result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
                if 'release 11' in cuda_result.stdout:
                    return "cu118"
                elif 'release 12' in cuda_result.stdout:
                    return "cu121" 
                else:
                    return "cu118"  # Default
            except:
                return "cu118"  # Default CUDA version
    except:
        pass
    
    print("ℹ️  No NVIDIA GPU detected, using CPU version")
    return "cpu"

def install_pytorch(gpu_version):
    """Install PyTorch with appropriate GPU support"""
    if gpu_version == "cpu":
        command = "pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu"
        description = "Installing PyTorch (CPU version)"
    else:
        command = f"pip install torch torchvision --index-url https://download.pytorch.org/whl/{gpu_version}"
        description = f"Installing PyTorch (GPU version - {gpu_version})"
    
    return run_command(command, description)

def install_packages():
    """Install all required packages"""
    packages = [
        "opencv-python>=4.8.0",
        "numpy>=1.24.0", 
        "matplotlib>=3.7.0",
        "pygame>=2.5.0",
        "Pillow>=9.5.0",
        "scikit-learn>=1.3.0",
        "tqdm>=4.65.0"
    ]
    
    command = f"pip install {' '.join(packages)}"
    return run_command(command, "Installing additional packages")

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    dirs = [
        "models",
        "haar_cascade_files", 
        "data/train/close",
        "data/train/open",
        "data/valid/close", 
        "data/valid/open"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {dir_path}")

def download_haar_cascades():
    """Download Haar cascade files"""
    print("\n⬇️  Downloading Haar cascade files...")
    
    base_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/"
    files = {
        "haarcascade_frontalface_alt.xml": "haarcascade_frontalface_alt.xml",
        "haarcascade_lefteye_2splits.xml": "haarcascade_lefteye_2splits.xml", 
        "haarcascade_righteye_2splits.xml": "haarcascade_righteye_2splits.xml"
    }
    
    for filename, url_filename in files.items():
        url = base_url + url_filename
        filepath = f"haar_cascade_files/{filename}"
        
        try:
            urllib.request.urlretrieve(url, filepath)
            print(f"✅ Downloaded: {filename}")
        except Exception as e:
            print(f"❌ Failed to download {filename}: {e}")

def create_alarm_sound():
    """Create a simple alarm sound"""
    print("\n🔊 Creating alarm sound...")
    
    try:
        import numpy as np
        import wave
        
        # Generate a simple beep sound
        duration = 2.0  # seconds
        sample_rate = 22050
        frequency1 = 440  # A4 note
        frequency2 = 880  # A5 note (octave higher)
        
        # Create time array
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        
        # Generate beep waveform (mix of two frequencies)
        wave_data = np.sin(frequency1 * 2 * np.pi * t) + 0.5 * np.sin(frequency2 * 2 * np.pi * t)
        
        # Normalize and convert to 16-bit integers
        wave_data = np.int16(wave_data / np.max(np.abs(wave_data)) * 32767)
        
        # Write WAV file
        with wave.open('alarm.wav', 'w') as wav_file:
            wav_file.setnchannels(1)  # mono
            wav_file.setsampwidth(2)  # 2 bytes per sample
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(wave_data.tobytes())
        
        print("✅ Alarm sound created: alarm.wav")
        
    except Exception as e:
        print(f"⚠️  Could not create alarm sound: {e}")
        print("Creating basic alarm file...")
        
        # Create a basic WAV file with simple tone
        try:
            with open('alarm.wav', 'wb') as f:
                # Write basic WAV header and some tone data
                header = b'RIFF\x24\x08\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x22\x56\x00\x00\x44\xAC\x00\x00\x02\x00\x10\x00data\x00\x08\x00\x00'
                tone_data = b'\x00\x00' * 1000  # Simple silence, but valid WAV
                f.write(header + tone_data)
            print("✅ Basic alarm.wav file created")
        except:
            print("❌ Could not create alarm file. Please add your own alarm.wav")

def test_installation():
    """Test if everything is installed correctly"""
    print("\n🧪 Testing installation...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
        
        import numpy as np
        import matplotlib.pyplot as plt
        import pygame
        from PIL import Image
        print("✅ All packages imported successfully!")
        
        # Test camera access
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✅ Camera access: OK")
            cap.release()
        else:
            print("⚠️  Camera access: Failed (check camera permissions)")
            
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 DRIVER DROWSINESS DETECTION - COMPLETE SETUP")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return
    
    print(f"✅ Python version: {sys.version}")
    
    # Upgrade pip
    run_command("python -m pip install --upgrade pip", "Upgrading pip")
    
    # Detect GPU and install PyTorch
    gpu_version = detect_gpu()
    if not install_pytorch(gpu_version):
        print("❌ PyTorch installation failed!")
        return
    
    # Install other packages
    if not install_packages():
        print("❌ Package installation failed!")
        return
    
    # Create directories
    create_directories()
    
    # Download Haar cascades
    download_haar_cascades()
    
    # Create alarm sound
    create_alarm_sound()
    
    # Test installation
    if test_installation():
        print("\n" + "=" * 60)
        print("🎉 SETUP COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\n📝 Next steps:")
        print("1. Add your training data to data/train/ and data/valid/ folders")
        print("2. Run: python model.py (to train the model)")
        print("3. Run: python main.py (to start drowsiness detection)")
        print("\n✅ Your system is ready for drowsiness detection!")
    else:
        print("\n❌ Setup completed with some issues. Please check the errors above.")

if __name__ == "__main__":
    main()