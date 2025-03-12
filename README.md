# F5-TTS-Gradio-Example

This project provides a simple example of integrating F5-TTS (a state-of-the-art text-to-speech model) with a Gradio
interface, along with an API for use in other applications.

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/cansik/f5-tts-gradio-example.git
   cd f5-tts-gradio-example
   ```

2. **Create a Virtual Environment** (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

To launch the Gradio interface:

```bash
python main.py
```

This will start a local web server. Open the provided URL (typically `http://127.0.0.1:7860/`) in your web browser to
access the interface.

## Parameters

The `main.py` script accepts several command-line arguments:

- `-r` or `--reference-audio`: Path to the reference audio file. This audio is used to capture the desired voice
  characteristics for synthesis.

- `-t` or `--reference_text`: Path to a text file containing the transcript of the reference audio.

- `--model`: Path or name of the pre-trained F5-TTS model to use. Defaults to `"lucasnewman/f5-tts-mlx"`.

- `--method`: Sampling method for audio generation. Options include:
    - `euler`: Euler method.
    - `midpoint`: Midpoint method.
    - `rk4`: Runge-Kutta method (default).

Example usage:

```bash
python main.py -r path/to/reference_audio.wav -t path/to/reference_text.txt --model your_model_name --method rk4
```

## About

Copyright (c) 2025 Florian Bruggisser