from flask import Flask, render_template, request, redirect
import speech_recognition as sr
import pickle
from werkzeug.utils import secure_filename
import os
#Install all the Reqiuired Libraries and Packages 
import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc , logfbank
import librosa as lr
import glob
from scipy import signal
#import noisereduce as nr
from glob import glob
import librosa
#All the Required Packages and Libraies are installed.
import soundfile
app = Flask(__name__)

model = pickle.load(open('Emotion_Voice_Detection_Model.pkl', 'rb'))

UPLOAD_FOLDER = 'C:/Users/ACER/Speech_Emotion/output'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'wav'}

@app.route("/", methods=["GET", "POST"])
def index():
	return render_template('index.html')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    dir = 'C:/Users/ACER/Speech_Emotion/output'
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return render_template('redir.html')
            #return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            return render_template('redir.html')
            #flash('No selected file')
            #return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            os.rename(UPLOAD_FOLDER +'/' + filename, UPLOAD_FOLDER+ '/' + 'output10.wav')
            #return 'file uploaded successfully'
        else:
            return render_template('redir.html')
        file = 'C:/Users/ACER/Speech_Emotion/output/output10.wav'
        ans =[]
        new_feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
        #print(type(new_feature))
        ans.append(new_feature)
        ans = np.array(ans)
        # data.shape
        data=model.predict(ans)
        result=data[0]
        return render_template('result.html',datatorender=result)

    #return result

def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
        result=np.hstack((result, mel))
    return result

if __name__ == "__main__":
    app.run(debug=True, threaded=True)