
import whisper
import torch
import os
import time
import io
import speech_recognition as sr
from transformers import pipeline
from tempfile import NamedTemporaryFile
from pydub import AudioSegment

abs_path = os.path.dirname(__file__)

temp_file = NamedTemporaryFile().name
temp_file = os.path.join(abs_path, "../media/king-fast.mp3")

audio_model_name = None
audio_model = None

benchmark_start = 0

def benchmark(name=None):
	global benchmark_start
	if (name != None):
		print(name)
		benchmark_start = time.time()
	else:
		dif = time.time() - benchmark_start
		print("Elapsed time: %.2fs" % dif)

def transcribe(model, file):
	global audio_model_name
	global audio_model
	if (audio_model_name != model):
		audio_model_name = model
		#benchmark("Loading model: " + model)
		model_time = time.time()
		audio_model = whisper.load_model(model)
		model_time = time.time() - model_time
		#benchmark()
	#benchmark("Transcribing: " + file)
	transcribe_time = time.time()
	result = audio_model.transcribe(os.path.join(abs_path, "../media", file), fp16=torch.cuda.is_available(), language="no")
	transcribe_time = time.time() - transcribe_time
	text = result['text'].strip()
	print(model + " (" + "%.2fs" % model_time + " load) (" + "%.2fs" % transcribe_time + " transcribe) (" + "%.2fs" % (model_time + transcribe_time) + " total)")
	#print(model_time)
	#benchmark()
	#print("Result:")
	print(text)
	print(" ")

def transcribe_nb(model, file):
	model_time = time.time()
	asr = pipeline(
		"automatic-speech-recognition",
		"NbAiLab/" + model,
		device="cuda:0"
	)
	model_time = time.time() - model_time
	transcribe_time = time.time()
	result = asr(
		os.path.join(abs_path, "../media", file),
		chunk_length_s=28,
		generate_kwargs={"task": "transcribe", "language": "no"}
	)
	transcribe_time = time.time() - transcribe_time
	# print("Result:")
	print(model + " (" + "%.2fs" % model_time + " load) (" + "%.2fs" % transcribe_time + " transcribe) (" + "%.2fs" % (model_time + transcribe_time) + " total)")
	print(result["text"].strip())
	# benchmark()
	print(" ")





transcribe("small", "king.mp3")
transcribe("medium", "king.mp3")
transcribe("large", "king.mp3")
transcribe("large-v3", "king.mp3")

transcribe_nb("nb-whisper-small-beta", "king.mp3")
transcribe_nb("nb-whisper-medium-beta", "king.mp3")
transcribe_nb("nb-whisper-large-beta", "king.mp3")



## ATTEMPT TO COMPRESS AUDIO

# benchmark("Generating: king-fast.mp3")

# sound = AudioSegment.from_file(os.path.join(abs_path, "../media/king.mp3"))
# sound.export(temp_file, format="mp3", parameters=["-ac","2","-ar","8000"])

# """
# with open(os.path.join(abs_path, "../media", "king.mp3"), "rb") as fd:
# 	contents = fd.read()

# audio_data = sr.AudioData(contents, 8000, 2)
# wav_data = io.BytesIO(audio_data.get_wav_data())

# with open(temp_file, 'w+b') as f:
# 	f.write(wav_data.read())
# """

# transcribe("small", temp_file)
