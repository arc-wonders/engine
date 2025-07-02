import streamlit as st
import os
import tempfile
import shutil
from pathlib import Path
import subprocess
import sys
from typing import Dict, Optional, Tuple
import time

# Import audio processing libraries
try:
    import librosa
    import soundfile as sf
except ImportError:
    st.error("Required audio libraries not installed. Please install librosa and soundfile.")
    st.stop()

class MusicSeparator:
    """
    Handles music separation using Spleeter and Demucs.
    Modular design for easy integration with future AI scoring components.
    """
    
    def __init__(self):
        self.temp_dir = None
        self.separated_components = {}
        
    def setup_temp_directory(self) -> str:
        """Create temporary directory for processing files."""
        if self.temp_dir is None:
            self.temp_dir = tempfile.mkdtemp(prefix="music_separation_")
        return self.temp_dir
        
    def validate_audio_file(self, file_path: str) -> bool:
        """Validate uploaded audio file format and integrity."""
        try:
            # Check file extension
            valid_extensions = ['.mp3', '.wav', '.flac', '.m4a', '.ogg']
            if not any(file_path.lower().endswith(ext) for ext in valid_extensions):
                return False
                
            # Try to load audio metadata
            librosa.load(file_path, sr=None, duration=1.0)
            return True
        except Exception:
            return False
    
    def separate_audio_with_demucs(self, input_file_path: str, progress_callback=None, model="htdemucs") -> Dict[str, str]:
        """
        Separate using Demucs with enhanced model options and error handling.
        
        Args:
            input_file_path: Path to input audio file
            progress_callback: Optional callback for progress updates
            model: Demucs model to use ("htdemucs", "htdemucs_ft", "htdemucs_6s", "mdx_extra")
        """
        temp_dir = self.setup_temp_directory()
        output_dir = os.path.join(temp_dir, "demucs_separated")
        
        try:
            if progress_callback:
                progress_callback(f"Initializing Demucs with {model} model...")
            
            # Prepare Demucs command with enhanced options
            cmd = [
                sys.executable, "-m", "demucs.separate",
                "--name", model,
                "--out", output_dir,
                "--float32",  # Better quality
                "--clip-mode", "rescale",  # Handle clipping
                input_file_path
            ]
            
            if progress_callback:
                progress_callback("Running Demucs separation (this may take several minutes)...")
            
            # Execute with longer timeout for larger files
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                # Try with fallback options if initial attempt fails
                if "cuda" in result.stderr.lower() or "gpu" in result.stderr.lower():
                    if progress_callback:
                        progress_callback("GPU not available, retrying with CPU...")
                    cmd.extend(["--device", "cpu"])
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                
                if result.returncode != 0:
                    raise Exception(f"Demucs separation failed: {result.stderr}")
            
            if progress_callback:
                progress_callback("Processing separated components...")
            
            # Find separated files (Demucs directory structure)
            input_filename = Path(input_file_path).stem
            separated_folder = os.path.join(output_dir, model, input_filename)
            
            # Standard 4-stem output
            component_files = {
                "vocals": os.path.join(separated_folder, "vocals.wav"),
                "drums": os.path.join(separated_folder, "drums.wav"),
                "bass": os.path.join(separated_folder, "bass.wav"),
                "other": os.path.join(separated_folder, "other.wav")
            }
            
            # Handle 6-stem models if available
            if model == "htdemucs_6s":
                component_files.update({
                    "piano": os.path.join(separated_folder, "piano.wav"),
                    "guitar": os.path.join(separated_folder, "guitar.wav")
                })
            
            # Verify components exist
            missing_components = []
            for component, file_path in component_files.items():
                if not os.path.exists(file_path):
                    missing_components.append(component)
            
            if missing_components:
                raise Exception(f"Missing components: {', '.join(missing_components)}")
            
            self.separated_components = component_files
            
            if progress_callback:
                progress_callback("Demucs separation completed successfully!")
                
            return component_files
            
        except subprocess.TimeoutExpired:
            raise Exception("Demucs separation process timed out (10 minutes). Try with a shorter audio file.")
        except Exception as e:
            raise Exception(f"Demucs separation failed: {str(e)}")
    
    def separate_audio_with_spleeter(self, input_file_path: str, progress_callback=None, stems=4) -> Dict[str, str]:
        """
        Separate audio using Spleeter with enhanced configuration options.
        
        Args:
            input_file_path: Path to input audio file
            progress_callback: Optional callback for progress updates
            stems: Number of stems (2, 4, or 5)
        """
        temp_dir = self.setup_temp_directory()
        output_dir = os.path.join(temp_dir, "spleeter_separated")
        
        try:
            if progress_callback:
                progress_callback(f"Initializing Spleeter ({stems}-stem model)...")
            
            # Configure model based on stems
            model_configs = {
                2: "spleeter:2stems-16kHz",
                4: "spleeter:4stems-16kHz", 
                5: "spleeter:5stems-16kHz"
            }
            
            if stems not in model_configs:
                stems = 4  # Default fallback
            
            cmd = [
                sys.executable, "-m", "spleeter", "separate",
                "-p", model_configs[stems],
                "-o", output_dir,
                input_file_path
            ]
            
            if progress_callback:
                progress_callback("Running Spleeter separation...")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                raise Exception(f"Spleeter failed: {result.stderr}")
            
            if progress_callback:
                progress_callback("Processing separated components...")
            
            # Find separated files
            input_filename = Path(input_file_path).stem
            separated_folder = os.path.join(output_dir, input_filename)
            
            # Configure components based on model
            if stems == 2:
                component_files = {
                    "vocals": os.path.join(separated_folder, "vocals.wav"),
                    "accompaniment": os.path.join(separated_folder, "accompaniment.wav")
                }
            elif stems == 5:
                component_files = {
                    "vocals": os.path.join(separated_folder, "vocals.wav"),
                    "drums": os.path.join(separated_folder, "drums.wav"),
                    "bass": os.path.join(separated_folder, "bass.wav"),
                    "piano": os.path.join(separated_folder, "piano.wav"),
                    "other": os.path.join(separated_folder, "other.wav")
                }
            else:  # 4 stems (default)
                component_files = {
                    "vocals": os.path.join(separated_folder, "vocals.wav"),
                    "drums": os.path.join(separated_folder, "drums.wav"),
                    "bass": os.path.join(separated_folder, "bass.wav"),
                    "other": os.path.join(separated_folder, "other.wav")
                }
            
            # Verify all components exist
            for component, file_path in component_files.items():
                if not os.path.exists(file_path):
                    raise Exception(f"Component {component} not generated")
            
            self.separated_components = component_files
            
            if progress_callback:
                progress_callback("Spleeter separation completed!")
                
            return component_files
            
        except subprocess.TimeoutExpired:
            raise Exception("Spleeter separation process timed out (5 minutes)")
        except Exception as e:
            raise Exception(f"Spleeter separation failed: {str(e)}")
    
    def separate_audio(self, input_file_path: str, progress_callback=None, method="auto", **kwargs) -> Dict[str, str]:
        """
        Separate audio into components using specified or best available method.
        
        Args:
            input_file_path: Path to input audio file
            progress_callback: Optional callback for progress updates
            method: "spleeter", "demucs", or "auto"
            **kwargs: Additional arguments for specific methods
            
        Returns:
            Dictionary mapping component names to file paths
        """
        # Determine method
        if method == "auto":
            tools = check_separation_tools()
            if tools["demucs"]:
                method = "demucs"
            elif tools["spleeter"]:
                method = "spleeter"
            else:
                raise Exception("No separation tools available. Please install Spleeter or Demucs.")
        
        if method == "demucs":
            model = kwargs.get("model", "htdemucs")
            return self.separate_audio_with_demucs(input_file_path, progress_callback, model)
        elif method == "spleeter":
            stems = kwargs.get("stems", 4)
            return self.separate_audio_with_spleeter(input_file_path, progress_callback, stems)
        else:
            raise Exception(f"Unknown separation method: {method}")
    
    def get_audio_info(self, file_path: str) -> Tuple[float, int]:
        """Get audio duration and sample rate for component analysis."""
        try:
            y, sr = librosa.load(file_path, sr=None)
            duration = len(y) / sr
            return duration, sr
        except Exception:
            return 0.0, 0
    
    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None
            self.separated_components = {}

def check_separation_tools():
    """Check which separation tools are available with version info."""
    tools = {"spleeter": False, "demucs": False, "spleeter_version": None, "demucs_version": None}
    
    # Check Spleeter
    try:
        result = subprocess.run([sys.executable, "-m", "spleeter", "--help"], 
                              capture_output=True, timeout=10, text=True)
        tools["spleeter"] = result.returncode == 0
        if tools["spleeter"]:
            # Try to get version
            try:
                version_result = subprocess.run([sys.executable, "-c", "import spleeter; print(spleeter.__version__)"], 
                                              capture_output=True, timeout=5, text=True)
                if version_result.returncode == 0:
                    tools["spleeter_version"] = version_result.stdout.strip()
            except:
                pass
    except Exception:
        pass
    
    # Check Demucs
    try:
        result = subprocess.run([sys.executable, "-c", "import demucs"], 
                              capture_output=True, timeout=10)
        tools["demucs"] = result.returncode == 0
        if tools["demucs"]:
            # Try to get version
            try:
                version_result = subprocess.run([sys.executable, "-c", "import demucs; print(demucs.__version__)"], 
                                              capture_output=True, timeout=5, text=True)
                if version_result.returncode == 0:
                    tools["demucs_version"] = version_result.stdout.strip()
            except:
                pass
    except Exception:
        pass
    
    return tools

def install_separation_tool(tool="spleeter"):
    """Install separation tool with progress tracking."""
    try:
        if tool == "spleeter":
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "spleeter>=2.4.0", "tensorflow>=2.5.0"
            ], capture_output=True, timeout=180, text=True)
        elif tool == "demucs":
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "demucs>=4.0.0", "torch", "torchaudio"
            ], capture_output=True, timeout=300, text=True)
        else:
            return False, "Unknown tool"
            
        if result.returncode == 0:
            return True, "Installation successful"
        else:
            return False, result.stderr
    except subprocess.TimeoutExpired:
        return False, "Installation timed out"
    except Exception as e:
        return False, str(e)

def main():
    st.set_page_config(
        page_title="Music Component Separation Tool",
        page_icon="🎵",
        layout="wide"
    )
    
    st.title("🎵 Advanced Music Component Separation Tool")
    st.markdown("**Separate audio into individual components using Demucs or Spleeter for detailed analysis**")
    
    # Initialize session state
    if 'separator' not in st.session_state:
        st.session_state.separator = MusicSeparator()
    if 'separated_files' not in st.session_state:
        st.session_state.separated_files = None
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    
    # Enhanced sidebar with tool management
    with st.sidebar:
        st.header("🔧 System Configuration")
        tools = check_separation_tools()
        
        # Tool status display
        st.subheader("Available Tools")
        if tools["demucs"]:
            version_info = f" (v{tools['demucs_version']})" if tools["demucs_version"] else ""
            st.success(f"✅ Demucs{version_info}")
            st.caption("High-quality separation with GPU acceleration")
        else:
            st.error("❌ Demucs not installed")
            
        if tools["spleeter"]:
            version_info = f" (v{tools['spleeter_version']})" if tools["spleeter_version"] else ""
            st.success(f"✅ Spleeter{version_info}")
            st.caption("Fast separation with good quality")
        else:
            st.error("❌ Spleeter not installed")
        
        # Installation section
        if not tools["demucs"] or not tools["spleeter"]:
            st.subheader("📥 Install Tools")
            
            col1, col2 = st.columns(2)
            with col1:
                if not tools["demucs"] and st.button("Install Demucs", help="Recommended for best quality"):
                    with st.spinner("Installing Demucs (this may take several minutes)..."):
                        success, message = install_separation_tool("demucs")
                        if success:
                            st.success("Demucs installed successfully!")
                            st.rerun()
                        else:
                            st.error(f"Installation failed: {message}")
            
            with col2:
                if not tools["spleeter"] and st.button("Install Spleeter", help="Faster processing"):
                    with st.spinner("Installing Spleeter..."):
                        success, message = install_separation_tool("spleeter")
                        if success:
                            st.success("Spleeter installed successfully!")
                            st.rerun()
                        else:
                            st.error(f"Installation failed: {message}")
        
        # Tool and model selection
        if tools["demucs"] or tools["spleeter"]:
            st.subheader("⚙️ Separation Settings")
            
            # Method selection
            method_options = []
            if tools["demucs"]:
                method_options.append("Demucs (Recommended)")
            if tools["spleeter"]:
                method_options.append("Spleeter (Fast)")
            
            selected_method = st.selectbox("Separation Method:", method_options)
            separation_method = "demucs" if "Demucs" in selected_method else "spleeter"
            
            # Model-specific options
            if separation_method == "demucs" and tools["demucs"]:
                st.subheader("🎛️ Demucs Models")
                demucs_models = {
                    "htdemucs": "HT-Demucs (Recommended) - 4 stems",
                    "htdemucs_ft": "HT-Demucs Fine-tuned - 4 stems", 
                    "htdemucs_6s": "HT-Demucs 6-stem - Vocals, Drums, Bass, Piano, Guitar, Other",
                    "mdx_extra": "MDX Extra - Alternative architecture"
                }
                
                selected_model = st.selectbox(
                    "Model:",
                    options=list(demucs_models.keys()),
                    format_func=lambda x: demucs_models[x],
                    help="Different models offer various trade-offs between quality and processing time"
                )
                
            elif separation_method == "spleeter" and tools["spleeter"]:
                st.subheader("🎛️ Spleeter Configuration")
                stem_options = {
                    2: "2-stem: Vocals + Accompaniment",
                    4: "4-stem: Vocals, Drums, Bass, Other", 
                    5: "5-stem: Vocals, Drums, Bass, Piano, Other"
                }
                
                selected_stems = st.selectbox(
                    "Stems:",
                    options=list(stem_options.keys()),
                    format_func=lambda x: stem_options[x],
                    index=1  # Default to 4-stem
                )
        else:
            separation_method = "auto"
            selected_model = "htdemucs"
            selected_stems = 4
    
    # File upload section
    st.header("📁 Upload Audio File")
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['mp3', 'wav', 'flac', 'm4a', 'ogg'],
        help="Upload your music file for component separation. Supported formats: MP3, WAV, FLAC, M4A, OGG"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_input_path = os.path.join(
            st.session_state.separator.setup_temp_directory(),
            uploaded_file.name
        )
        
        with open(temp_input_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Validate file
        if not st.session_state.separator.validate_audio_file(temp_input_path):
            st.error("❌ Invalid audio file. Please upload a valid audio file.")
            return
        
        # Display file info
        duration, sample_rate = st.session_state.separator.get_audio_info(temp_input_path)
        file_size_mb = len(uploaded_file.getvalue()) / 1024 / 1024
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("⏱️ Duration", f"{duration:.1f} seconds")
        with col2:
            st.metric("🔊 Sample Rate", f"{sample_rate:,} Hz")
        with col3:
            st.metric("💾 File Size", f"{file_size_mb:.1f} MB")
        
        # Warnings for large files
        if duration > 300:  # 5 minutes
            st.warning("⚠️ Large file detected. Processing may take several minutes.")
        if file_size_mb > 50:
            st.warning("⚠️ Large file size. Consider using a shorter audio clip for faster processing.")
        
        # Original audio player
        st.subheader("🎵 Original Audio")
        st.audio(uploaded_file.getvalue())
        
        # Separation button
        separation_disabled = st.session_state.processing or not (tools["demucs"] or tools["spleeter"])
        
        if st.button("🔧 Separate Audio Components", type="primary", disabled=separation_disabled):
            if not (tools["demucs"] or tools["spleeter"]):
                st.error("Please install Demucs or Spleeter first!")
                return
                
            st.session_state.processing = True
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            start_time = time.time()
            
            def update_progress(message):
                elapsed = time.time() - start_time
                status_text.text(f"{message} (Elapsed: {elapsed:.1f}s)")
                
                # Update progress bar
                if "Initializing" in message:
                    progress_bar.progress(20)
                elif "Running" in message:
                    progress_bar.progress(40)
                elif "Processing" in message:
                    progress_bar.progress(80)
                elif "completed" in message or "complete" in message:
                    progress_bar.progress(100)
            
            try:
                # Prepare separation parameters
                kwargs = {}
                if separation_method == "demucs":
                    kwargs["model"] = selected_model
                elif separation_method == "spleeter":
                    kwargs["stems"] = selected_stems
                
                # Perform separation
                separated_files = st.session_state.separator.separate_audio(
                    temp_input_path, 
                    progress_callback=update_progress,
                    method=separation_method,
                    **kwargs
                )
                
                st.session_state.separated_files = separated_files
                
                elapsed_total = time.time() - start_time
                st.success(f"✅ Audio separation completed successfully in {elapsed_total:.1f} seconds!")
                
            except Exception as e:
                st.error(f"❌ Separation failed: {str(e)}")
                st.session_state.separated_files = None
            
            finally:
                st.session_state.processing = False
                progress_bar.empty()
                status_text.empty()
    
    # Display separated components
    if st.session_state.separated_files:
        st.header("🎼 Separated Audio Components")
        st.markdown("Preview and download individual components:")
        
        # Component organization
        components_info = {
            "vocals": {"icon": "🎤", "name": "Vocals", "description": "Lead and backing vocals"},
            "drums": {"icon": "🥁", "name": "Drums", "description": "Drum kit and percussion"},
            "bass": {"icon": "🎸", "name": "Bass", "description": "Bass guitar and low frequencies"},
            "other": {"icon": "🎹", "name": "Other Instruments", "description": "Remaining instruments"},
            "accompaniment": {"icon": "🎵", "name": "Accompaniment", "description": "All instruments (no vocals)"},
            "piano": {"icon": "🎹", "name": "Piano", "description": "Piano and keyboard"},
            "guitar": {"icon": "🎸", "name": "Guitar", "description": "Electric and acoustic guitar"}
        }
        
        # Create responsive columns
        num_components = len(st.session_state.separated_files)
        cols = st.columns(min(num_components, 3))
        
        for idx, (component_key, file_path) in enumerate(st.session_state.separated_files.items()):
            if os.path.exists(file_path):
                col = cols[idx % len(cols)]
                component_info = components_info.get(component_key, {
                    "icon": "🎵", "name": component_key.title(), "description": f"{component_key} component"
                })
                
                with col:
                    st.subheader(f"{component_info['icon']} {component_info['name']}")
                    st.caption(component_info['description'])
                    
                    # Audio player
                    with open(file_path, "rb") as audio_file:
                        audio_bytes = audio_file.read()
                        st.audio(audio_bytes)
                    
                    # Component metadata
                    duration, sr = st.session_state.separator.get_audio_info(file_path)
                    size_mb = len(audio_bytes) / 1024 / 1024
                    
                    col_info1, col_info2 = st.columns(2)
                    with col_info1:
                        st.metric("Duration", f"{duration:.1f}s")
                    with col_info2:
                        st.metric("Size", f"{size_mb:.1f}MB")
                    
                    # Download button
                    st.download_button(
                        label=f"📥 Download {component_info['name']}",
                        data=audio_bytes,
                        file_name=f"{component_key}.wav",
                        mime="audio/wav",
                        use_container_width=True
                    )
                    
                    # Future analysis placeholder
                    with st.expander(f"🔮 Analysis Preview"):
                        st.info(f"Future AI analysis for {component_info['name']} will appear here")
        
        # Global analysis section
        st.header("🧠 Component Analysis Dashboard")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info("🔮 **Future Features:**")
            st.markdown("""
            - Cross-component attention analysis
            - Musical interaction scoring  
            - Harmonic relationship mapping
            - Rhythm pattern detection
            - Dynamic range analysis
            """)
        
        with col2:
            st.info("📊 **Analysis Capabilities:**")
            st.markdown("""
            - Spectral analysis per component
            - Energy distribution visualization
            - Frequency domain insights
            - Temporal pattern recognition  
            - Component interaction metrics
            """)
    
    # Footer with cleanup
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if st.session_state.separated_files or st.session_state.separator.temp_dir:
            if st.button("🧹 Clear All Files", use_container_width=True):
                st.session_state.separator.cleanup()
                st.session_state.separated_files = None
                st.success("✅ All temporary files cleared!")
                st.rerun()

if __name__ == "__main__":
    main()