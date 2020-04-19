import librosa
import yaml
import os
import numpy as np
from python_speech_features import mfcc
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle

def feature_extraction(path, MFCC_dim=13):
    audio, sr = librosa.load(path)
    energy = librosa.feature.rms(audio)
    frames = np.nonzero(energy >= np.max(energy) / 5)
    indices = librosa.core.frames_to_samples(frames)[1]
    audio = audio[indices[0]:indices[-1]] if indices.size else audio[0:0]
    feature = mfcc(audio, sr, numcep=MFCC_dim, nfft=551)
    return audio, feature

def visiualize(feature, audio):
    plt.figure(figsize=(12, 3))
    plt.plot(np.arange(len(audio)), audio)
    plt.title('Raw Audio Signal')
    plt.xlabel('Time')
    plt.ylabel('Audio Amplitude')
    plt.show()

    print('Shape of MFCC:', feature.shape)
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)
    im = ax.imshow(feature, cmap=plt.cm.jet, aspect='auto')
    plt.title('Normalized MFCC')
    plt.ylabel('Time')
    plt.xlabel('MFCC Coefficient')
    plt.colorbar(im, cax=make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05))
    ax.set_xticks(np.arange(0, 13, 2), minor=False)
    plt.show()

    return

if __name__=='__main__':

    config_file = open('config.yml')
    config = yaml.load(config_file, Loader=yaml.Loader)
    config_file.close()


    if 'mfcc.txt' in os.listdir():
        f = open('mfcc.txt', 'rb+')
        Speakers_Utterance = pickle.load(f)
        f.close()

        visiualize(Speakers_Utterance['SF1'][1][0],Speakers_Utterance['SF1'][0][0])

    else:
        source_path = config['source_audio_path']
        Speakers = os.listdir(source_path)
        for filename in Speakers:
            if len(filename) > 3:
                Speakers.remove(filename)

            # create speakers dir

        Speakers_Utterance = {}
        for speaker in Speakers:

            # print(speaker + " done!")
            speaker_source = os.path.join(source_path, speaker)
            speaker_utters_source = os.listdir(speaker_source)
            print(speaker_source)      # test
        #
            Utters_MFCC = []
            Utters_Audio = []
            for i in speaker_utters_source:
        #         # print(i)
                utters_source = os.path.join(speaker_source, i)
                speaker_utters_source_files = os.listdir(utters_source)
                # print(speaker_utters_source_files)
                for j in speaker_utters_source_files:
                    audio_flie = os.path.join(utters_source,j)
                    audio, feature = feature_extraction(audio_flie, 13)
                    Utters_MFCC.append(feature)
                    Utters_Audio.append(audio)
        #
                Speakers_Utterance[speaker] = [Utters_Audio, Utters_MFCC]
        #
        f = open('mfcc.txt','wb+')
        pickle.dump(Speakers_Utterance,f,0)
        f.close()