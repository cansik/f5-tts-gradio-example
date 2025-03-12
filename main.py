import argparse
import datetime
import pkgutil
from pathlib import Path
from typing import Optional, Literal, Tuple

import gradio as gr
import mlx.core as mx
import numpy as np
import soundfile as sf
from f5_tts_mlx import F5TTS
from f5_tts_mlx.generate import TARGET_RMS, SAMPLE_RATE, FRAMES_PER_SEC, estimated_duration, split_sentences
from f5_tts_mlx.utils import convert_char_to_pinyin
from tqdm import tqdm


def trim_silence(audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """
    Trim silence from the beginning and end of an audio array.

    Parameters:
        audio (np.ndarray): The input audio array.
        threshold (float): Threshold amplitude below which is considered silence.

    Returns:
        np.ndarray: Trimmed audio array.
    """
    if audio.size == 0:
        return audio

    # Find indices where the absolute amplitude exceeds the threshold
    non_silent_indices = np.where(np.abs(audio) > threshold)[0]

    # Check if there is no non-silent segment
    if non_silent_indices.size == 0:
        return np.array([], dtype=audio.dtype)

    # Find start and end of non-silent audio
    start_index = non_silent_indices[0]
    end_index = non_silent_indices[-1] + 1  # +1 to include the endpoint

    return audio[start_index:end_index]


class F5TTSGenerator:
    def __init__(
            self,
            model_name: str = "lucasnewman/f5-tts-mlx",
            ref_audio_path: Optional[str] = None,
            ref_audio_text: Optional[str] = None,
            quantization_bits: Optional[int] = None,
    ):
        # Load the model; convert weights unless it's the default model.
        convert_weights = model_name != "lucasnewman/f5-tts-mlx"
        self.f5tts = F5TTS.from_pretrained(
            model_name, convert_weights=convert_weights, quantization_bits=quantization_bits
        )

        # Load reference audio and set reference text.
        if ref_audio_path is None:
            data = pkgutil.get_data("f5_tts_mlx", "tests/test_en_1_ref_short.wav")
            tmp_ref_audio_file = "/tmp/ref.wav"
            with open(tmp_ref_audio_file, "wb") as f:
                f.write(data)
            audio, sr = sf.read(tmp_ref_audio_file)
            self.ref_audio_text = ref_audio_text or "Some call me nature, others call me mother nature."
        else:
            audio, sr = sf.read(ref_audio_path)
            if sr != SAMPLE_RATE:
                raise ValueError("Reference audio must have a sample rate of 24kHz")
            self.ref_audio_text = ref_audio_text

        self.audio = mx.array(audio)
        self.audio_duration = self.audio.shape[0] / SAMPLE_RATE
        print(f"Loaded reference audio with duration: {self.audio_duration:.2f} seconds")

        # Normalize the reference audio if its RMS is below the target.
        rms = mx.sqrt(mx.mean(mx.square(self.audio)))
        if rms < TARGET_RMS:
            self.audio = self.audio * TARGET_RMS / rms

    def _compute_duration(self, text: str, provided_duration: Optional[float], estimate_duration: bool, speed: float) -> \
            Optional[int]:
        """
        Compute the duration in frames. If a duration is provided, convert it.
        If estimate_duration is True, estimate the duration using the helper.
        Otherwise, return None.
        """
        if provided_duration is not None:
            return int(provided_duration * FRAMES_PER_SEC)
        elif estimate_duration:
            return int(estimated_duration(self.audio, self.ref_audio_text, text, speed) * FRAMES_PER_SEC)
        else:
            return None

    def _sample_text(
            self,
            text: str,
            duration_frames: Optional[int],
            steps: int,
            method: Literal["euler", "midpoint"],
            speed: float,
            cfg_strength: float,
            sway_sampling_coef: float,
            seed: Optional[int],
    ) -> mx.array:
        """
        Generate a wave sample for a single text segment.
        It combines the reference text with the provided text, converts it to pinyin,
        and calls the model to sample audio.
        """
        converted_text = convert_char_to_pinyin([self.ref_audio_text + " " + text])
        wave, _ = self.f5tts.sample(
            mx.expand_dims(self.audio, axis=0),
            text=converted_text,
            duration=duration_frames,
            steps=steps,
            method=method,
            speed=speed,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            seed=seed,
        )
        # Trim the reference audio portion from the beginning.
        wave = wave[self.audio.shape[0]:]
        mx.eval(wave)
        return wave

    def generate(
            self,
            generation_text: str,
            duration: Optional[float] = None,
            estimate_duration: bool = False,
            steps: int = 8,
            method: Literal["euler", "midpoint"] = "rk4",
            cfg_strength: float = 2.0,
            sway_sampling_coef: float = -1.0,
            speed: float = 1.0,
            seed: Optional[int] = None,
            output_path: Optional[str] = None,
    ) -> Tuple[int, np.ndarray]:
        """
        Generate audio for the given text and save it to the output path if provided.
        If the input text contains multiple sentences (and no fixed duration is given),
        each sentence is processed individually and concatenated.
        """
        start_time = datetime.datetime.now()
        sentences = split_sentences(generation_text)

        if len(sentences) <= 1 or duration is not None:
            # Single-generation mode.
            dur = self._compute_duration(generation_text, duration, estimate_duration, speed)
            wave = self._sample_text(generation_text, dur, steps, method, speed, cfg_strength, sway_sampling_coef, seed)
        else:
            # Multi-generation mode: process each sentence individually.
            wave_parts = []
            for sentence in tqdm(sentences, desc="Generating sentences"):
                dur = self._compute_duration(sentence, duration, estimate_duration, speed)
                wave_parts.append(
                    self._sample_text(sentence, dur, steps, method, speed, cfg_strength, sway_sampling_coef, seed)
                )
            wave = mx.concat(wave_parts, axis=0)

        generated_duration = wave.shape[0] / SAMPLE_RATE
        elapsed = datetime.datetime.now() - start_time
        print(f"Generated {generated_duration:.2f}s of audio in {elapsed}.")

        data = np.array(wave)

        if output_path is not None:
            # Convert the MXNet array to a NumPy array before writing to disk.
            sf.write(output_path, data, SAMPLE_RATE)

        return SAMPLE_RATE, data


generator: Optional[F5TTSGenerator] = None


def process(text: str, quality_steps: int, sampling_method: str):
    sr, wave = generator.generate(text,
                                  steps=quality_steps,
                                  method=sampling_method)  # noqa

    # sometimes the generate wave has a long start silence
    # here we trim the wave to just the spoken part
    trimmed_wave = trim_silence(wave)
    return sr, trimmed_wave


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--reference-audio", type=str, help="Path to the reference audio file.")
    parser.add_argument("-t", "--reference_text", type=str, help="Path to the reference audio text file.")
    parser.add_argument("--model", type=str, default="lucasnewman/f5-tts-mlx", help="Path or name of the model.")
    parser.add_argument("--method", type=str, default="rk4", help="Sampling method (one-of: euler, midpoint, rk4).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    reference_audio_path = Path(args.reference_audio) if args.reference_audio is not None else None
    reference_text_path = Path(args.reference_text) if args.reference_text is not None else None
    model_path = Path(args.model)

    # read textfiles
    if reference_text_path is not None:
        reference_text = reference_text_path.read_text(encoding="utf-8")
    else:
        reference_text = None

    # load model and embed voice
    global generator
    generator = F5TTSGenerator(
        model_name=str(model_path),
        ref_audio_path=str(reference_audio_path) if reference_audio_path is not None else None,
        ref_audio_text=reference_text
    )

    # create gradio and launch
    demo = gr.Interface(title="F5 TTS Demo",
                        fn=process,
                        inputs=[
                            gr.Text(label="Text"),
                            gr.Slider(label="Quality Steps", minimum=1, maximum=20, value=8),
                            gr.Dropdown(label="Sampling Method", choices=["euler", "midpoint", "rk4"], value="rk4")
                        ],
                        outputs=[
                            gr.Audio(label="Output", format="wav")
                        ],
                        flagging_mode="never"
                        )
    demo.launch()


if __name__ == "__main__":
    main()
