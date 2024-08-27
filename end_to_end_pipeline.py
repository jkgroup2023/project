import numpy as np
import whisper
import webrtcvad
import asyncio
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import edge_tts

# Step 1: Voice-to-Text Conversion (Using Whisper with VAD)
def load_and_preprocess_audio(audio_path, vad_threshold=0.5):
    model = whisper.load_model("small.en")
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)

    # Convert the audio to PCM 16-bit format required by webrtcvad
    audio_pcm = (audio * 32767).astype(np.int16)

    # VAD integration
    vad = webrtcvad.Vad()
    vad.set_mode(2)  # Sensitivity level: 0 (least aggressive) to 3 (most aggressive)

    # Split audio into frames for VAD
    frame_duration = 20  # ms
    frame_size = int(16000 * frame_duration / 1000)
    frames = [audio_pcm[i:i + frame_size].tobytes() for i in range(0, len(audio_pcm), frame_size)]

    # Apply VAD to remove non-speech segments
    speech_frames = []
    for frame in frames:
        if len(frame) == frame_size * 2 and vad.is_speech(frame, sample_rate=16000):  # Ensure frame size and VAD check
            speech_frames.append(np.frombuffer(frame, dtype=np.int16))

    # Concatenate speech frames back into one audio array
    if speech_frames:
        processed_audio = np.concatenate(speech_frames).astype(np.float32) / 32767  # Convert back to float32
    else:
        processed_audio = audio  # Fallback to original audio if no speech detected

    # Transcribe the audio using Whisper
    result = model.transcribe(processed_audio, language="en")
    return result['text']


# Step 2: Text Input into LLM (GPT-2)
def query_llm(transcribed_text):
    # Load GPT-2 model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    model = GPT2LMHeadModel.from_pretrained("distilgpt2")

    # Prepare input text from previous transcription
    input_ids = tokenizer.encode(transcribed_text, return_tensors='pt')

    # Generate response with a max of 50 tokens (~2 sentences)
    output_ids = model.generate(input_ids, max_length=50, num_return_sequences=1)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Restrict output to 2 sentences
    output_text = ' '.join(output_text.split('.')[:2]) + '.'
    return output_text


# Step 3: Text-to-Speech Conversion (Edge TTS)
async def text_to_speech(output_text, output_audio_path, voice="en-US-JennyNeural", pitch="+0Hz", rate="+0%"):
    # Initialize the Edge TTS engine and generate speech
    tts = edge_tts.Communicate(output_text, voice=voice, pitch=pitch, rate=rate)
    await tts.save(output_audio_path)
    print(f"Speech saved to {output_audio_path}")


# Main function to combine all steps
async def main(audio_path, output_audio_path):
    # Step 1: Convert voice to text
    transcribed_text = load_and_preprocess_audio(audio_path)
    print(f"Transcribed Text: {transcribed_text}")

    # Step 2: Pass the text into LLM and get response
    response_text = query_llm(transcribed_text)
    print(f"Generated Response: {response_text}")

    # Step 3: Convert the LLM response back into speech
    await text_to_speech(response_text, output_audio_path, voice="en-US-JennyNeural", pitch="+0Hz", rate="+0%")


# Run the pipeline
if __name__ == "__main__":
    audio_path = "input_audio.wav"  # Input audio file
    output_audio_path = "output_audio.mp3"  # Output speech file

    asyncio.run(main(audio_path, output_audio_path))
