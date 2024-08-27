import asyncio
import edge_tts

class TextToSpeech:
    @staticmethod
    async def synthesize_speech(text, voice="en-US-AriaNeural", rate="+0%", pitch="+0Hz", output_file="output2.wav"):
        communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
        with open(output_file, 'wb') as file:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    await file.write(chunk["data"])
        return file