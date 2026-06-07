from pathlib import Path
import requests
import logging
from typing import Dict, Optional
from datetime import datetime
from time import sleep
from tqdm import tqdm
from pydub import AudioSegment
import os

class TranscriptSummarizer:
    """Handles transcript summarization using LLM."""

    def __init__(self, config):
        """Initialize with configuration and safe provider fallbacks."""
        self.config = config
        llm_config = config.get("llm", {})
        
        self.model_name = llm_config.get("model_name", "llama3.1:8b")
        self.max_retries = llm_config.get("max_retries", 5)
        self.retry_delay = llm_config.get("retry_delay", 3)
        self.llm_options = llm_config.get("options", {})

        # 1. Determine the provider
        if "provider" in llm_config:
            self.provider = llm_config["provider"].lower()
        elif "api_mode" in llm_config:
            self.provider = llm_config["api_mode"].lower()
        elif "api_url" in llm_config and "/v1/" in llm_config["api_url"]:
            self.provider = "lm_studio"
        else:
            self.provider = "ollama"

        # 2. Build the API URL dynamically
        if "api_url" in llm_config and not any(p in llm_config for p in ["ollama", "lm_studio", "openai"]):
            self.api_url = llm_config["api_url"]
        else:
            provider_settings = llm_config.get(self.provider, {})
            base_url = provider_settings.get("base_url", "http://localhost:11434")
            endpoint = provider_settings.get("endpoint", "/api/generate")
            self.api_url = f"{base_url.rstrip('/')}{endpoint}"

        # 3. Pull optional API keys
        self.api_key = llm_config.get("openai", {}).get("api_key", None)

        logging.info(f"TranscriptSummarizer initialized. Provider: {self.provider} | URL: {self.api_url}")

    def _read_transcript(self, transcript_path: str) -> str:
        """Read transcript file.

        Args:
            transcript_path (str): Path to the transcript file

        Returns:
            str: Content of the transcript file
        """
        try:
            logging.info(f"Reading transcript from: {transcript_path}")
            with open(transcript_path, "r", encoding="utf-8") as file:
                return file.read()
        except Exception as e:
            logging.error(f"Failed to read transcript from {transcript_path}: {e}")
            raise

    def process_transcript(self, transcript_path: str, audio_path: str) -> Path:
        """Main processing pipeline for transcript summarization.

        Args:
            transcript_path (str): Path to the transcript file
            audio_path (str): Path to the original audio file

        Returns:
            Path: Path to the generated summary file
        """
        try:
            logging.info("Starting transcript processing")
            self.audio_path = audio_path

            transcript = self._read_transcript(transcript_path)            

            with tqdm(total=1, desc="Generating summary", unit="summary") as pbar:
                summary = self._generate_summary(transcript)
                pbar.update(1)

            metadata = self._prepare_metadata(audio_path)
            formatted_doc = self._format_document(
                {
                    "summary": summary, 
                    "transcription": transcript
                }, 
                metadata
            )
            return self._save_document(formatted_doc)

        except Exception as e:
            logging.error(f"Transcript processing failed: {e}")
            raise

    def _generate_summary(self, text: str) -> str:
        """Generate summary using LLM with support for multiple API providers."""
        prompt = self.config["prompts"]["summary_prompt"]
        full_prompt = f"{prompt}\n\nText: {text}"

        for attempt in range(self.max_retries):
            try:
                logging.info(
                    f"Attempt {attempt + 1} to generate summary (Provider: {self.provider})"
                )

                # Both OpenAI and LM Studio use the exact same payload structure
                if self.provider in ["openai", "lm_studio"]:
                    payload = self._build_openai_payload(full_prompt)
                    
                    headers = {"Content-Type": "application/json"}
                    if self.api_key and self.provider == "openai":
                        headers["Authorization"] = f"Bearer {self.api_key}"

                    response = requests.post(
                        self.api_url,
                        json=payload,
                        timeout=600,
                        headers=headers,
                    )
                    response.raise_for_status()
                    result = self._parse_openai_response(response.json())

                else:  # Default to Ollama mode
                    payload = self._build_ollama_payload(full_prompt)
                    response = requests.post(self.api_url, json=payload, timeout=600)
                    response.raise_for_status()
                    result = response.json()["response"]

                if not result.strip():
                    raise ValueError("Empty LLM response")

                return result

            except Exception as e:
                logging.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    sleep(self.retry_delay)
                else:
                    raise

    def _build_ollama_payload(self, prompt: str) -> Dict:
        """Build request payload for Ollama API."""
        return {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": self.llm_options,
        }

    def _build_openai_payload(self, prompt: str) -> Dict:
        """Build request payload for OpenAI/LLMStudio compatible API."""
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "temperature": self.llm_options.get("temperature", 0.7),
        }

        # Optionally include other OpenAI-compatible parameters
        if "max_tokens" in self.llm_options:
            payload["max_tokens"] = self.llm_options["max_tokens"]
        if "top_p" in self.llm_options:
            payload["top_p"] = self.llm_options["top_p"]

        return payload

    def _parse_openai_response(self, response_data: Dict) -> str:
        """Parse response from OpenAI/LLMStudio compatible API."""
        try:
            # Handle chat completions format
            if "choices" in response_data and len(response_data["choices"]) > 0:
                choice = response_data["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    return choice["message"]["content"]
                elif "text" in choice:
                    return choice["text"]
            # Fallback: attempt to extract from common alternative structures
            if "response" in response_data:  # Some LLMStudio instances may return this
                return response_data["response"]
            raise ValueError(f"Unexpected response structure: {response_data.keys()}")
        except KeyError as e:
            raise ValueError(f"Failed to parse OpenAI-style response: {e}")

    def _prepare_metadata(self, audio_path: str) -> Dict[str, str]:
        """Prepare document metadata."""
        try:
            logging.info("Preparing metadata")
            metadata = {
                "date": datetime.now().strftime(
                    self.config["document_format"]["metadata"]["date_format"]
                ),
                "duration": self._get_audio_duration(audio_path),
            }

            defaults = self.config["document_format"]["metadata"]["defaults"]
            for field in self.config["document_format"]["metadata"]["fields"]:
                metadata[field] = defaults.get(field, "Not specified")

            return metadata
        except Exception as e:
            logging.error("Error preparing metadata")
            raise

    def _get_audio_duration(self, audio_path: str) -> str:
        """Get formatted audio duration."""
        try:
            logging.info(f"Getting audio duration for: {audio_path}")
            audio = AudioSegment.from_file(audio_path)
            duration = len(audio) / 1000.0

            hours = int(duration // 3600)
            minutes = int((duration % 3600) // 60)
            seconds = int(duration % 60)

            if hours > 0:
                return f"{hours}h {minutes}m {seconds}s"
            elif minutes > 0:
                return f"{minutes}m {seconds}s"
            return f"{seconds}s"

        except Exception as e:
            logging.warning(f"Could not determine audio duration: {e}")
            return "Duration unavailable"

    def _format_document(
        self, summaries: Dict[str, str], metadata: Dict[str, str]
    ) -> str:
        """Format the summary document."""
        try:
            logging.info("Formatting document")
            metadata_section = self._format_metadata(metadata)
            
            try:
                rel_audio_path = os.path.relpath(self.audio_path)
            except ValueError:
                # Fallback to original path if relative resolution fails (e.g., across different drives on Windows)
                rel_audio_path = self.audio_path

            return self.config["document_format"]["template"].format(
                metadata_section=metadata_section,
                summary=summaries["summary"],
                transcription=summaries["transcription"],
                audio_path=rel_audio_path,
                generation_timestamp=datetime.now().strftime(
                    self.config["document_format"]["metadata"]["date_format"]
                ),
            )
        except Exception as e:
            logging.error("Error formatting document")
            raise

    def _format_metadata(self, metadata: Dict[str, str]) -> str:
        """Format metadata section."""
        try:
            logging.info("Formatting metadata section")
            lines = [self.config["document_format"]["metadata"]["header"]]
            for field in self.config["document_format"]["metadata"]["fields"]:
                lines.append(
                    f"- {field.title()}: {metadata.get(field, 'Not specified')}"
                )
            return "\n".join(lines + ["\n"])
        except Exception as e:
            logging.error("Error formatting metadata section")
            raise

    def _save_document(self, formatted_text: str) -> Path:
        """
        Save the formatted document using the audio filename and timestamp.

        Args:
            formatted_text: The formatted document content to save

        Returns:
            Path: The path to the saved document
        """
        try:
            logging.info("Saving document")
            output_dir = Path(self.config["transcription"]["meeting_summary_directory"])
            output_dir.mkdir(parents=True, exist_ok=True)

            audio_filename = Path(self.audio_path).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{audio_filename}_summary_{timestamp}.{self.config['output']['format']}"

            output_path = output_dir / filename

            with open(output_path, "w", encoding="utf-8") as file:
                file.write(formatted_text)
            logging.info(f"Document saved: {output_path}")
            return output_path
        except Exception as e:
            logging.error(f"Failed to save document: {e}")
            raise

