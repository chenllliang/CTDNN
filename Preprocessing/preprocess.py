import librosa
import yaml
import os
import numpy as np
from python_speech_features import mfcc
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
import threading
import time

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
    ax.set_xticks(np.arange(0, 13, 2), minor=False);
    plt.show()
    return


def process_per_worker(path_list,id)->dict:
    current_Mfccs={}
    for i in path_list:
        speaker_utterances=[]
        subfolders = os.listdir(i)
        for j in subfolders:
            path = os.path.join(i,j)
            for q in os.listdir(path):
                utters_source = os.path.join(path, q)
                audio, feature = feature_extraction(utters_source, 25)
                speaker_utterances.append(feature)
        
        current_Mfccs[i] = speaker_utterances
    f = open('Preprocessing/mfcc'+str(id)+'.txt','wb+')
    pickle.dump(current_Mfccs,f,0)
    f.close()

    print(id,'done!')
    
    


def preprocess_Voxceleb1_training_data_multithreads(Train_Folder_Path,num_workers=1):
    threading_pool = []
    Speakers = os.listdir(Train_Folder_Path)
    num_of_speakers = len(Speakers)
    jobs_per_worker = num_of_speakers//num_workers

    path_list=[]
    path_per_worker=[]
    #divide work for per thread
    for i,j in enumerate(Speakers):
        path_per_worker.append(os.path.join(Train_Folder_Path,j))
        if (i!=0 and i%jobs_per_worker==0) or i==len(Speakers)-1:
            
            path_list.append(path_per_worker)
            path_per_worker=[]

    for i in range(len(path_list)):
        x = threading.Thread(target = process_per_worker,args=(path_list[i],i))
        x.start()


    



if __name__=='__main__':

    config_file = open('Preprocessing/config.yml')
    config = yaml.load(config_file, Loader=yaml.Loader)
    config_file.close()

    preprocess_Voxceleb1_training_data_multithreads(config['source_audio_path'],5)


    # if 'mfcc.txt' in os.listdir():
    #     f = open('mfcc.txt', 'rb+')
    #     Speakers_Utterance = pickle.load(f)
    #     f.close()
    #     visiualize(Speakers_Utterance['SF1'][1][0],Speakers_Utterance['SF1'][0][0])

    # else:
    #     source_path = config['source_audio_path']
    #     Speakers = os.listdir(source_path)
    #     Speakers_Utterance = {}
    #     for speaker in Speakers:
    #         speaker_source = os.path.join(source_path, speaker)
    #         sub_folders=os.listdir(speaker_source)

    #         Utters_MFCC = []
    #         for i in sub_folders:
    #             utterances_folder = os.path.join(speaker_source,i)
    #             utterances = os.listdir(utterances_folder)

    #             for j in utterances:
    #                 utters_source = os.path.join(utterances_folder, j)
    #                 audio, feature = feature_extraction(utters_source, 25)
    #                 Utters_MFCC.append(feature)
    #         print(speaker,"done!")
    #         Speakers_Utterance[speaker] = Utters_MFCC

    #     f = open('mfcc.txt','wb+')
    #     pickle.dump(Speakers_Utterance,f,0)
    #     f.close()
