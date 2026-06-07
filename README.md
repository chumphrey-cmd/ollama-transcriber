# Purpose

This project offers a privacy-focused solution for transcribing and summarizing audio recordings through entirely local processing on your machine. Using OpenAI's Whisper for transcription and local LLMs via Ollama for summarization, it processes audio files (**MP3/WAV**) entirely on your machine, ensuring sensitive content never leaves your environment.

The tool automatically generates structured summaries including:

- Executive overview
- Detailed content breakdown
- Action items
- Meeting metadata

> [!NOTE]
> This project is functional on **Linux** and **Windows 11**.

---

## Linux Setup: Using Python Virtual Environment (Recommended)

### Prerequisites

Install the system-level dependencies using your package manager:

```bash
# Debian/Ubuntu
sudo apt install python3 python3-venv python3-pip ffmpeg

# Fedora
sudo dnf install python3 python3-pip ffmpeg-free

# Arch
sudo pacman -S python python-pip ffmpeg
```

You also need [Ollama](https://ollama.com/download) installed and a model pulled (e.g., `ollama pull llama3.1:8b`).

### Automated Setup

From the root of the cloned repository, run the install script:

```bash
# CPU-only PyTorch (works on any machine)
./install.sh

# Or, if you have an NVIDIA GPU with CUDA:
./install.sh --cuda
```

The script creates a virtual environment, installs all Python dependencies, and verifies your setup.

<details>
<summary><strong>Linux Manual Setup (Click to expand)</strong></summary>

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# CPU-only PyTorch:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# Or CUDA PyTorch (see https://pytorch.org/get-started/locally/):
# pip install torch torchvision torchaudio

pip install -r requirements.txt
python pytorch_verify.py
```

</details>

---

## Windows Automated Setup: Using Python Virtual Environment (Recommended)

If you are on Windows, you can use the included PowerShell script to automatically install Chocolatey, FFmpeg, Python 3.10, and all required Python dependencies. 

**Important:** You must run this script directly from the root folder of the cloned repository.

1. Open PowerShell as an **Administrator**.

2. Navigate to your cloned project directory:

```bash
cd path\to\Ollama-Transcriber
```

3. Run the setup script using the following command (this temporarily bypasses Windows execution policies to allow the script to run):

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\install.ps1
```

The script will automatically check for missing dependencies, set up your Python virtual environment, and verify your GPU access.

> [!IMPORTANT]
> **Restart Required:** If you have VS Code, Cmder, Command Prompt, or any other terminal open while running the setup script, you **must completely close and restart** those applications (and sometimes reboot your computer) after the installation finishes. 
> 
> Otherwise, your terminal will not recognize the newly installed `ffmpeg` command, and audio processing will fail.

<details>
<summary><strong>Windows Manual Setup: Direct Install (Click to expand)</strong></summary>

### Select Python Interpreter Version

This project requires **Python 3.8 or later**. It is highly recommended to set up a virtual environment (`python -m venv venv`) before proceeding.

### Install `ffmpeg` Globally as PowerShell Administrator

`ffmpeg` is required for Whisper to process audio files. Follow the instructions [HERE](https://chocolatey.org/install#individual) to install Chocolatey via PowerShell Administration, then install `ffmpeg`:

```powershell
choco install ffmpeg

```

### Requirements Installation

```bash
python -m pip install -r requirements.txt --no-warn-script-location

```

### Enable Long Paths

From an Administrator PowerShell window, run the following:

```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force

```

### Download PyTorch with CUDA Support for GPU Acceleration

1. If you have an NVIDIA GPU, determine your compute platform by running: `nvidia-smi.exe`
2. Identify your "CUDA Version".
3. Navigate to: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
4. Select options specific to your environment and run the provided install command.
5. Once installation is complete, verify your setup:
```bash
python pytorch_verify.py

```



</details>

---

## Usage

### LLM Customization

* [Install Ollama on your system](https://ollama.com/download) and download your preferred model.
* Modify the **`config.yaml`** file located in **`src/utils/config.yaml`** and specify the model you are using.
* Refer to the [Ollama documentation](https://github.com/ollama/ollama/tree/main/docs) for details on other available options like `num_ctx`, `num_predict`, `top_k`, `repeat_penalty`, and `num_gpu`.

```yaml
llm:
  model_name: "mistral:latest" # Choose your Ollama model (e.g., "mistral:latest", "llama3.1:8b")
  options:
    temperature: 0.3 # Controls response creativity (0.0-1.0). Higher values are more creative.
    top_p: 0.5 # Controls similarity sampling (accuracy) when generating a response (0.1-1).
```

### Begin Ollama Server

Before running the transcriber, ensure your local Ollama server is running:

```bash
ollama serve
```

### Run Project

To run the project, ensure your virtual environment is active, then use the `main.py` script:

```bash
python main.py [OPTIONS]
```

<details>
<summary><strong>Click here to see all available CLI commands and flags</strong></summary>

* `python main.py --gui`: Use the graphical user interface (GUI) to select an audio file.
* `python main.py --audio path/to/recording.mp3`: Process a specific audio file with default settings.
* `python main.py --audio path/to/recording.mp3 --language es`: Specify the language of the audio file (e.g., Spanish) using ISO codes.
* `python main.py --audio path/to/recording.mp3 --output path/to/output --transcript medium`: Specify the output directory and the Whisper model size for transcription.
* `python main.py --audio path/to/recording.mp3 --llm mistral:latest`: Use a specific LLM model for summarization.
* `python main.py --audio path/to/recording.mp3 --output path/to/summaries --transcript medium --language es --llm mistral:latest`: Full example utilizing multiple customized flags.
* `python main.py --help`: Display the help menu with all available options.

</details>

The results of the processing will be stored in a `results` directory created in the same location where you run `main.py`. This directory will contain:

* `converted_audio/`: Stores the audio files converted to the required format (if necessary).
* `transcribed_text/`: Holds the raw `.txt` transcriptions of the audio files.
* `meeting_summaries/`: Contains the generated LLM meeting summary files.

### Supported Languages

Whisper supports nearly 100 languages. Pass the 2-letter ISO code using the `--language` flag (e.g., `--language es`).

<details>
<summary><strong>Click here to expand the full list of language codes</strong></summary>

| Code | Language | Code | Language | Code | Language |
| --- | --- | --- | --- | --- | --- |
| `en` | English | `es` | Spanish | `fr` | French |
| `de` | German | `it` | Italian | `pt` | Portuguese |
| `nl` | Dutch | `ja` | Japanese | `ko` | Korean |
| `zh` | Chinese | `ru` | Russian | `ar` | Arabic |
| `hi` | Hindi | `tr` | Turkish | `pl` | Polish |

*(Note: You can find the complete list of all 90+ supported ISO-639-1 codes in the [official Whisper documentation](https://github.com/openai/whisper#available-models-and-languages).)*

</details>

### System Prompt Customization

Modify the `config.yaml` file located in `src/utils/config.yaml` to customize the exact structure and focus of the AI summary:

```yaml
prompts:
  summary_prompt: | 
    Analyze the provided transcript and create a comprehensive Summary Report that captures all essential information.

    Structure the summary as follows:

    1. **EXECUTIVE OVERVIEW**
    - Synthesize core meeting purpose and outcomes

    2. **KEY DISCUSSION POINTS**
    - Present main topics chronologically with timestamps

    3. **ACTION ITEMS AND RESPONSIBILITIES**
    - List concrete tasks with clear ownership and deliverables

    4. **CONCLUSIONS AND NEXT STEPS**
    - Summarize achieved outcomes against objectives
  ```