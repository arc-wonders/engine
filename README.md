# 🎵 Advanced Music Component Separation Tool

A powerful Streamlit application for separating audio files into individual components (vocals, drums, bass, instruments) using state-of-the-art AI models like Demucs and Spleeter.

## 📋 Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Supported Models](#supported-models)
- [Usage Guide](#usage-guide)
- [Troubleshooting](#troubleshooting)
- [Dependencies](#dependencies)
- [Contributing](#contributing)

## ✨ Features

### 🎛️ Audio Separation
- **Multiple AI Models**: Demucs (recommended) and Spleeter support
- **Flexible Stem Options**: 2, 4, 5, or 6-stem separation
- **High Quality Output**: Professional-grade audio separation
- **Batch Processing**: Handle multiple files efficiently

### 🖥️ User Interface
- **Web-based Interface**: Easy-to-use Streamlit application
- **Real-time Progress**: Live updates during processing
- **Audio Preview**: Play original and separated components
- **Download Options**: Save individual components as WAV files

### 🔧 Advanced Features
- **Auto-Installation**: Automatic tool installation from the UI
- **File Validation**: Smart audio file format detection
- **Metadata Display**: Duration, sample rate, and file size info
- **Error Handling**: Comprehensive error messages and solutions

## 🚀 Installation

### Prerequisites
- Python 3.8-3.11 (recommended: 3.10)
- FFmpeg installed on your system
- At least 4GB RAM (8GB+ recommended for large files)

### Option 1: Quick Install (Recommended)
```bash
# Clone or download the repository
git clone <repository-url>
cd music-separator

# Install with compatible dependencies
pip install "numpy>=1.19.0,<2.0.0" "tensorflow>=2.8.0,<2.16.0" "spleeter>=2.4.0" "demucs>=4.0.0" "torch>=1.13.0" "torchaudio>=0.13.0" "librosa>=0.8.0" "soundfile>=0.10.0" "streamlit>=1.28.0" "scipy>=1.7.0" "matplotlib>=3.3.0" "pandas>=1.3.0" "ffmpeg-python>=0.2.0"
```

### Option 2: Step-by-Step Install
```bash
# 1. Uninstall conflicting packages
pip uninstall numpy tensorflow spleeter demucs librosa soundfile -y

# 2. Install compatible NumPy first
pip install "numpy>=1.19.0,<2.0.0"

# 3. Install TensorFlow with NumPy 1.x compatibility
pip install "tensorflow>=2.8.0,<2.16.0"

# 4. Install Spleeter
pip install "spleeter>=2.4.0"

# 5. Install Demucs (recommended)
pip install "demucs>=4.0.0" "torch>=1.13.0" "torchaudio>=0.13.0"

# 6. Install audio processing libraries
pip install "librosa>=0.8.0" "soundfile>=0.10.0"

# 7. Install Streamlit
pip install "streamlit>=1.28.0"
```

### Option 3: Virtual Environment (Safest)
```bash
# Create virtual environment
python -m venv music_separator_env

# Activate (Windows)
music_separator_env\Scripts\activate
# Activate (Mac/Linux)
source music_separator_env/bin/activate

# Install dependencies
pip install --upgrade pip
pip install "numpy>=1.19.0,<2.0.0" "tensorflow>=2.8.0,<2.16.0" "spleeter>=2.4.0" "demucs>=4.0.0" "torch>=1.13.0" "torchaudio>=0.13.0" "librosa>=0.8.0" "soundfile>=0.10.0" "streamlit>=1.28.0"
```

### FFmpeg Installation
**Windows:**
```bash
# Using chocolatey
choco install ffmpeg

# Or download from https://ffmpeg.org/download.html
```

**Mac:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt update
sudo apt install ffmpeg
```

## 🎯 Quick Start

1. **Launch the Application**
   ```bash
   streamlit run music_separator.py
   ```

2. **Open Your Browser**
   - Navigate to `http://localhost:8501`
   - The application will open automatically

3. **Upload Audio File**
   - Supported formats: MP3, WAV, FLAC, M4A, OGG
   - File size limit: 200MB (configurable)
   - Duration: Works best with files under 10 minutes

4. **Choose Separation Method**
   - **Demucs** (recommended): Higher quality, slower processing
   - **Spleeter**: Faster processing, good quality

5. **Start Separation**
   - Click "Separate Audio Components"
   - Wait for processing (2-10 minutes depending on file size)
   - Download individual components

## 🤖 Supported Models

### Demucs Models
| Model | Stems | Quality | Speed | Description |
|-------|-------|---------|-------|-------------|
| `htdemucs` | 4 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Recommended for most users |
| `htdemucs_ft` | 4 | ⭐⭐⭐⭐⭐ | ⭐⭐ | Fine-tuned version |
| `htdemucs_6s` | 6 | ⭐⭐⭐⭐ | ⭐⭐ | Separates piano and guitar |
| `mdx_extra` | 4 | ⭐⭐⭐⭐ | ⭐⭐⭐ | Alternative architecture |

### Spleeter Models
| Model | Stems | Components | Use Case |
|-------|-------|------------|----------|
| 2-stem | 2 | Vocals, Accompaniment | Simple vocal isolation |
| 4-stem | 4 | Vocals, Drums, Bass, Other | Balanced separation |
| 5-stem | 5 | Vocals, Drums, Bass, Piano, Other | Detailed separation |

## 📖 Usage Guide

### Basic Workflow
1. **File Upload**: Drag and drop or select audio file
2. **Model Selection**: Choose separation method and model
3. **Processing**: Monitor progress bar and status
4. **Results**: Preview and download separated components
5. **Analysis**: View component metadata and future AI insights

### Advanced Options
- **GPU Acceleration**: Automatically detected for Demucs
- **Batch Processing**: Process multiple files sequentially
- **Quality Settings**: Adjust sample rate and bit depth
- **Custom Models**: Load user-trained models (advanced)

### Best Practices
- **File Preparation**: Use high-quality source files (320kbps+ MP3 or lossless)
- **Length Optimization**: Shorter files (3-5 minutes) process faster
- **Model Selection**: Use Demucs for quality, Spleeter for speed
- **System Resources**: Close other applications during processing

## 🔧 Troubleshooting

### Common Issues

#### NumPy Compatibility Error
```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x
```
**Solution:**
```bash
pip uninstall numpy tensorflow spleeter -y
pip install "numpy>=1.19.0,<2.0.0" "tensorflow>=2.8.0,<2.16.0" "spleeter>=2.4.0"
```

#### CUDA/GPU Issues
```
GPU not available, retrying with CPU...
```
**Solution:**
- Install CUDA-compatible PyTorch: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
- Or use CPU-only version (slower but works everywhere)

#### FFmpeg Not Found
```
FFmpeg not found in PATH
```
**Solution:**
- Install FFmpeg system-wide
- Restart terminal/command prompt
- Verify with `ffmpeg -version`

#### Memory Issues
```
Out of memory error
```
**Solution:**
- Use shorter audio files
- Close other applications
- Increase virtual memory/swap file
- Use CPU processing instead of GPU

#### Installation Conflicts
```
Package version conflicts
```
**Solution:**
```bash
# Create fresh environment
python -m venv fresh_env
source fresh_env/bin/activate  # or fresh_env\Scripts\activate on Windows
pip install --upgrade pip
# Install from requirements
```

### Performance Optimization
- **CPU**: Use all available cores automatically
- **GPU**: NVIDIA GPUs with CUDA support recommended
- **RAM**: 8GB+ recommended for large files
- **Storage**: SSD recommended for faster I/O

## 📦 Dependencies

### Core Libraries
```
streamlit>=1.28.0          # Web interface
numpy>=1.19.0,<2.0.0      # Numerical computing (IMPORTANT: <2.0.0)
librosa>=0.8.0             # Audio analysis
soundfile>=0.10.0          # Audio I/O
```

### AI Models
```
spleeter>=2.4.0            # Facebook's separation model
tensorflow>=2.8.0,<2.16.0 # Deep learning framework
demucs>=4.0.0              # Meta's separation model
torch>=1.13.0              # PyTorch framework
torchaudio>=0.13.0         # Audio processing for PyTorch
```

### Utilities
```
scipy>=1.7.0               # Scientific computing
matplotlib>=3.3.0          # Plotting
pandas>=1.3.0              # Data manipulation
ffmpeg-python>=0.2.0       # FFmpeg Python wrapper
```

## 🎨 Component Analysis Features

### Current Features
- **Metadata Display**: Duration, sample rate, file size
- **Audio Preview**: Play original and separated components
- **Download Options**: Save components as WAV files
- **Progress Tracking**: Real-time separation progress

### Future Features (Planned)
- **Spectral Analysis**: Frequency domain visualization
- **Component Interaction**: Cross-component attention analysis
- **Musical Scoring**: AI-powered quality assessment
- **Harmonic Analysis**: Chord and key detection
- **Rhythm Detection**: Beat and tempo analysis

## 🔒 System Requirements

### Minimum Requirements
- **OS**: Windows 10, macOS 10.14, or Linux
- **CPU**: Dual-core processor
- **RAM**: 4GB
- **Storage**: 2GB free space
- **Python**: 3.8+

### Recommended Requirements
- **OS**: Windows 11, macOS 12+, or Ubuntu 20.04+
- **CPU**: Quad-core processor or better
- **RAM**: 8GB+
- **GPU**: NVIDIA GPU with CUDA support (optional)
- **Storage**: 10GB+ free space (SSD recommended)
- **Python**: 3.10

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

### Development Setup
```bash
# Clone repository
git clone <repository-url>
cd music-separator

# Create development environment
python -m venv dev_env
source dev_env/bin/activate  # or dev_env\Scripts\activate on Windows

# Install in development mode
pip install -e .
pip install -r requirements-dev.txt

# Run tests
pytest

# Run linting
flake8 music_separator.py
black music_separator.py
```

## 🙏 Acknowledgments

- **Demucs**: Meta Research for the state-of-the-art separation models
- **Spleeter**: Deezer for pioneering open-source music separation
- **Streamlit**: For the excellent web framework
- **Community**: All contributors and users who provide feedback

## 📞 Support

- **Issues**: Report bugs on [GitHub Issues](link-to-issues)
- **Discussions**: Join our [GitHub Discussions](link-to-discussions)
- **Email**: Contact [email@example.com](mailto:email@example.com)
- **Discord**: Join our [Discord Server](link-to-discord)

## 🔗 Related Projects

- [Demucs](https://github.com/facebookresearch/demucs) - Official Demucs repository
- [Spleeter](https://github.com/deezer/spleeter) - Official Spleeter repository
- [Librosa](https://librosa.org/) - Audio analysis library
- [Streamlit](https://streamlit.io/) - Web app framework

---

**Happy Music Separation!** 🎵✨
