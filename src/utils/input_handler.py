def select_audio_file():
    """
    Opens a file dialog for selecting an audio file.

    Requires python3-tk (tkinter) to be installed on Linux:
        sudo apt install python3-tk

    Returns:
        str: Path to selected audio file, or None if canceled
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        print("Error: tkinter is not installed. The --gui option requires it.")
        print("  Linux:   sudo apt install python3-tk")
        print("  Or use --audio <path> instead.")
        return None

    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(
        title='Select Audio File',
        filetypes=[
            ('Audio Files', '*.mp3 *.wav *.m4a *.flac'),
            ('All Files', '*.*')
        ]
    )
    return file_path if file_path else None
