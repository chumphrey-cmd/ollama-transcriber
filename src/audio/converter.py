import os
import subprocess
from pydub import AudioSegment

def check_ffmpeg():
    """Verify FFmpeg installation. Checks system PATH."""
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        print("FFmpeg not found. Please ensure ffmpeg is installed and in your PATH.\n"
              "  Linux:   sudo apt install ffmpeg   (or your distro's package manager)\n"
              "  macOS:   brew install ffmpeg\n"
              "  Windows: choco install ffmpeg")
        return False  # Return False if FFmpeg is not found

def get_supported_formats():
    """Returns a list of supported audio formats."""
    return ['mp3', 'wav', 'ogg', 'm4a', 'flac', 'aac', 'wma']

def validate_format(format_str):
    """Validates if the provided format is supported."""
    return format_str.lower() in get_supported_formats()

def convert_audio(input_file, output_format, output_file):
    """Converts audio file.

    Args:
        input_file: Path to input audio file.
        output_format: Desired output format.
        output_file: Path to output file.

    Raises:
        RuntimeError: If FFmpeg is not found.
        FileNotFoundError: If the input file is not found.
        ValueError: If the output format is not supported.
    """

    if not check_ffmpeg():
        raise RuntimeError("FFmpeg is not installed or not in PATH.")

    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    if not validate_format(output_format):
        raise ValueError(f"Unsupported output format: {output_format}. Supported formats are: {', '.join(get_supported_formats())}")

    original_filename = os.path.splitext(os.path.basename(input_file))[0]


    output_dir = os.path.dirname(output_file) # Extract directory

    if not output_file: # Handle if output file is None
        output_file = os.path.join(output_dir, f"{original_filename}.{output_format}")
    elif os.path.isdir(output_file): # Handle directory as output
        output_file = os.path.join(output_file, f"{original_filename}.{output_format}")
    else: # Handle file as output
        output_file = os.path.join(output_dir, f"{original_filename}.{output_format}")


    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        audio = AudioSegment.from_file(input_file)
        audio.export(output_file, format=output_format)
        return True
    except Exception as e:
        print(f"Conversion error: {e}")  # Keep print statements for this module
        return False