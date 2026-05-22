from html import parser
import logging
import sys
from pathlib import Path
import whisper
import argparse
import torch
from torch.xpu import is_available

# Add project root to Python path for proper import resolution
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Ensure the user is running a supported Python version (3.8 to 3.11)
if sys.version_info < (3, 8) or sys.version_info >= (3, 12):
    print("ERROR: Unsupported Python Version")
    print(f"You are running Python {sys.version_info.major}.{sys.version_info.minor}.")
    print("Ollama Transcriber requires Python 3.8, 3.9, 3.10, or 3.11.")
    print("Please use the correct version or activate your virtual environment.\n")
    sys.exit(1)

# Import custom modules for audio processing, transcription, and summarization
from src.utils.config import ConfigManager
from src.audio.converter import convert_audio
from src.transcription.transcribe import transcribe_audio
from src.summary.summarize import TranscriptSummarizer
from src.utils.input_handler import select_audio_file


def parse_arguments():
    """
    Parse command line arguments for the audio transcription and summarization tool.
    """
    parser = argparse.ArgumentParser(
        description="Audio Transcription and Summarization Tool",
        epilog="""
Examples:
    # Use GUI to select audio file or directory
    python main.py --gui
    
    # Process specific audio file
    python main.py --audio path/to/recording.mp3
    
    # Process all audio files in a directory
    python main.py --audio path/to/recordings_folder/
    
    # Full example with all options
    python main.py --audio path/to/recording.mp3 --output path/to/summaries --transcript medium --llm mistral:latest
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--audio", type=str, help="Path to audio file or directory to transcribe and summarize"
    )
    parser.add_argument(
        "--output", type=str, help="Path to output directory for saving summaries"
    )
    parser.add_argument(
        "--llm",
        type=str,
        help="Name of Ollama model to use for summarization (default: from config.yaml)",
    )
    parser.add_argument(
        "--transcript",
        type=str,
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model selection for transcription (default: from config.yaml)",
    )
    parser.add_argument(
        "--language",
        type=str,
        help='Language code for transcription (e.g., "en"). Use "auto" for detection.',
    )
    parser.add_argument(
        "--gui", action="store_true", help="Launch GUI file picker to select audio file"
    )

    return parser.parse_args()


def ensure_audio_format(audio_file_path: Path, config: dict) -> Path:
    """
    Checks the audio file format and converts it if necessary.
    Returns the path to the valid/converted audio file.
    """
    converted_audio_dir = Path(config["audio_processing"]["converted_audio_directory"])
    output_format = config["audio"]["output_format"]
    converted_audio_dir.mkdir(parents=True, exist_ok=True)

    print(f"Checking audio format for {audio_file_path.name}")
    logging.info(f"Checking audio format for {audio_file_path.name}")

    if audio_file_path.suffix.lower() != f".{output_format}":
        converted_audio_path = converted_audio_dir / f"{audio_file_path.stem}.{output_format}"
        logging.info(f"Converting {audio_file_path} to {output_format}")
        print(f"Converting audio to {output_format} format...")

        if not convert_audio(str(audio_file_path), output_format, str(converted_audio_path)):
            raise ValueError(f"Audio conversion failed for {audio_file_path}")
        
        logging.info(f"Audio converted successfully to: {converted_audio_path}")
        print("Audio converted successfully")
        return converted_audio_path
    else:
        logging.info("Audio already in correct format. Skipping conversion.")
        print("Audio already in correct format. Skipping conversion.")
        return audio_file_path


def process_file(file_path: Path, config: dict, model, summarizer, target_language: str):
    """
    Executes the conversion, transcription, and summarization pipeline for a single file.
    """

    try:
        # Step 1: Audio Processing and Conversion
        processed_audio_path = ensure_audio_format(file_path, config)

        # Step 2: Audio Transcription
        logging.info("Starting audio transcription...")
        print("Starting audio transcription...")

        transcription_dir = Path(config["transcription"]["transcription_directory"])
        transcription_dir.mkdir(parents=True, exist_ok=True)

        transcribe_audio(
            str(processed_audio_path),
            str(transcription_dir),
            model,
            language=target_language,
        )

        transcript_path = transcription_dir / f"{processed_audio_path.stem}.txt"
        logging.info(f"Transcription saved to: {transcript_path}")
        logging.getLogger().handlers[0].flush()
        print(f"Transcription saved to: {transcript_path}")

        # Step 3: Transcript Summarization
        logging.info("Starting summary generation...")
        print("Starting summary generation...")

        summary_path = summarizer.process_transcript(
            transcript_path=str(transcript_path), audio_path=str(processed_audio_path)
        )
        
        logging.info(f"Summary generated and saved to: {summary_path}")
        print(f"Summary generated and saved to: {summary_path}")
        print(f"--- Completed: {file_path.name} ---\n")

    except Exception as e:
        logging.error(f"Failed processing {file_path.name}: {e}")
        print(f"Error processing {file_path.name}: {e}. Skipping to next file.")


def main():
    logging.getLogger().setLevel(logging.INFO)
    logging.info("Starting up application...")
    
    try:
        # Configuration Loading
        args = parse_arguments()
        config_manager = ConfigManager()
        config = config_manager.config

        # Resolve Inputs
        if args.gui:
            file_path = select_audio_file()
            if file_path:
                config["paths"]["audio_file"] = file_path
            else:
                print("No audio file selected. Exiting.")
                sys.exit(1)
        elif args.audio:
            config["paths"]["audio_file"] = args.audio

        # Override configurations
        if args.output:
            config["transcription"]["meeting_summary_directory"] = args.output
        if args.llm:
            config["llm"]["model_name"] = args.llm
        if args.transcript:
            config["transcription"]["model_selection"] = args.transcript

        input_path = Path(config["paths"]["audio_file"])
        if not input_path.exists():
            logging.error(f"Input path not found: {input_path}")
            print(f"Error: Input path not found: {input_path}")
            sys.exit(1)

        # Build list of target files
        target_files = []
        supported_extensions = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".mp4"}
               
        if input_path.is_dir():
            print(f"Directory detected. Scanning for audio files in: {input_path}")
            
            # Use a set to automatically deduplicate paths on Windows
            unique_files = set() 
            
            for ext in supported_extensions:
                unique_files.update(input_path.rglob(f"*{ext}"))
                unique_files.update(input_path.rglob(f"*{ext.upper()}"))
                
            if not unique_files:
                print(f"No supported audio files found in directory: {input_path}")
                sys.exit(1)
            
            # Convert back to a list and sort alphabetically
            target_files = sorted(list(unique_files))
            
            print(f"Found {len(target_files)} unique audio files to process.")
        else:
            target_files = [input_path]        
                           
            
        # Cache already processed files by name
        output_dir = Path(config["transcription"]["meeting_summary_directory"])
        processed_stems = set()
        
        if output_dir.exists():
            print("Scanning output directory for existing summaries...")
            for summary_file in output_dir.iterdir():
                if summary_file.is_file() and "_summary_" in summary_file.name:
                    # Extract the original filename stem (everything before "_summary_")
                    original_stem = summary_file.name.split("_summary_")[0]
                    processed_stems.add(original_stem)
            
            if processed_stems:
                print(f"Found {len(processed_stems)} previously processed files. These will be skipped.")
        

        # Check if GPU (CUDA) is available, otherwise try XPU, else default to CPU
        if torch.cuda.is_available():
            device = "cuda"         
        else:
            if torch.xpu.is_available():
                device = "xpu"
            else:
                device = "cpu"

        # Initialize Models ONCE
        logging.info(f"Loading Whisper model: {config['transcription']['model_selection']} on {device}")
        print(f"Loading Whisper model '{config['transcription']['model_selection']}' into memory...")
        try:
            whisper_model = whisper.load_model(config["transcription"]["model_selection"], device=device)
            print("Whisper model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading Whisper model: {e}")
            print(f"Fatal Error: Could not load Whisper model: {e}")
            sys.exit(1)

        logging.info(f"Initializing LLM Summarizer with model: {config['llm']['model_name']}")
        print("Initializing Summarizer...")
        try:
            summarizer = TranscriptSummarizer(config)
        except Exception as e:
            logging.error(f"Error initializing summarizer: {e}")
            print(f"Fatal Error: Could not initialize summarizer: {e}")
            sys.exit(1)

        # Determine target language
        target_language = args.language if args.language else config["transcription"].get("language", "en")
        if target_language.lower() == "auto":
            target_language = None
        print(f"Using language: {target_language or 'auto-detect'}\n")

      
        # Process Queue
        total_files = len(target_files)
        pad_width = len(str(total_files)) 

        for idx, file_path in enumerate(target_files, start=1):
            # Create the formatted indicator string (e.g., "0034/2355")
            progress = f"{idx:0{pad_width}d}/{total_files}"
            
            # Check against our cached set of processed stems
            if file_path.stem in processed_stems:
                print(f"[{progress}] --- Skipping: {file_path.name} (Summary already exists) ---")
                logging.info(f"[{progress}] Skipped {file_path.name}: Already processed.")
                continue
                
            print(f"\n[{progress}] --- Processing: {file_path.name} ---")
            logging.info(f"[{progress}] Processing file: {file_path}")
            
            process_file(file_path, config, whisper_model, summarizer, target_language)
            
            # Add the stem to the cache immediately after processing
            processed_stems.add(file_path.stem)        
            
        print("All operations completed.")
        logging.info("Batch processing complete.")
            
    except Exception as e:
        logging.error(f"Fatal pipeline error: {e}")
        print(f"Fatal Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()