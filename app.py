# import subprocess
# import os
# def find_folder_name(fileName):
#     root_dir = os.getcwd()
#     for dirpath, _, filenames in os.walk(root_dir):
#         for filename in filenames:
#             if filename == fileName + ".txt"    :
#                 with open(filename, 'r') as file:
#                     contents = file.read()
#                 return contents
#     return None

# def audioToSpeech(audioLocation):
#     command = f"whisper {audioLocation} --task translate"
#     try:
#         # Execute the command and capture the output
#         output = subprocess.check_output(command, shell=True)
#         fileNameWithExtension = os.path.basename(audioLocation)
#         fileNameWithoutExtension = fileNameWithExtension.split('.')[0]
#         return find_folder_name(fileNameWithoutExtension)
#     except subprocess.CalledProcessError as e:
#         print("Error:", e)


import whisper


def audioToSpeech(audioLocation):
    model = whisper.load_model("medium")
    audio = whisper.load_audio(audioLocation)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)
    print(result.text)

audioToSpeech("audio1/Recording3.mp3")


