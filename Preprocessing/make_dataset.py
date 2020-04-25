from torch.utils import data
import torch
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def audio_padding_wrap(featuremap,maxlength):
    # make all input duration the same
    output = featuremap
    if featuremap.shape[0]<maxlength:
        padding_length = maxlength-featuremap.shape[0]
        padding_times = padding_length//featuremap.shape[0]+1
        for i in range(padding_times):
            output = np.concatenate([output,featuremap])
        new_length = output.shape[0]
        extra_length =new_length-maxlength
        output=np.delete(output,range(new_length-extra_length,new_length),axis=0)

    else:
        output = output[0:maxlength]
    return output

def val_train_test(Origin_Dataset_path,rate):
    '''
    :param Origin_Dataset_path: Dataset is conbined of n subfolders,each subfolder represent a class
    :param rate: the split rate of train,validation and test data , like [0.7,0.2,0.1], it should be a list have 3 elements and with a sum of 1
    :return: it will cut the origin folder in to 3 folders train,validation and test and return the list
    '''
    #Load all samples and divide into 3 lists

    Sample_classes = os.listdir(Origin_Dataset_path)
    #delete hidden files
    for filename in Sample_classes:
        if len(filename) > 3:
            Sample_classes.remove(filename)
    Samples = {}
    for one_class in Sample_classes:
        class_source = os.path.join(Origin_Dataset_path,one_class)
        samples_for_the_class = os.listdir(class_source)
        Samples[one_class] =[os.path.join(class_source,x) for x in samples_for_the_class ]

    return Samples


#
#         Speakers_Utterance = {}
#         for speaker in Speakers:
#             print(speaker + " done!")
#             speaker_source = os.path.join(source_path, speaker)
#             speaker_utters_source = os.listdir(speaker_source)
#
#             Utters_MFCC = []
#             Utters_Audio = []
#             for i in speaker_utters_source:
#                 utters_source = os.path.join(speaker_source, i)
#                 audio, feature = feature_extraction(utters_source, 13)
#                 Utters_MFCC.append(feature)
#                 Utters_Audio.append(audio)
#
#             Speakers_Utterance[speaker] = [Utters_Audio, Utters_MFCC]
#
#         f = open('mfcc.txt','wb+')
#         pickle.dump(Speakers_Utterance,f,0)
#         f.close()



class Make_VoxCeleb_Dataset(data.Dataset):
    def __init__(self):
        self.max_length = 0
        self.nums = 0
        self.length=[]
        self.Speakers_Utterance=[]
        self.line_up = 1000
        self.num_speakers=0

    def load_mfccs(self,mfccs_path):
        docs = os.listdir(mfccs_path)
        mfcc_list = [i for i in docs if 'mfcc' in i ]
        for i in mfcc_list:      
            with open(os.path.join(mfccs_path,i),'rb+') as f:
                speaker_utterance = pickle.load(f)
                for speaker in list(speaker_utterance.keys()):
                    self.num_speakers+=1
                    print("loading speaker",speaker)
                    for utter in speaker_utterance[speaker]:
                        self.Speakers_Utterance.append([speaker,utter])
                        self.nums+=1
                        self.length.append(utter.shape[0])
                        if utter.shape[0]>self.max_length:
                            self.max_length = utter.shape[0]

        print(self.max_length,sum(self.length)/len(self.length))

    def __getitem__(self, index):

        speaker_id = int(self.Speakers_Utterance[index][0][-4:])
        feature = self.Speakers_Utterance[index][1]
        feature = audio_padding_wrap(feature,self.line_up)
        feature = torch.from_numpy(feature).float()
        return  feature,speaker_id

    def __len__(self):
        return self.nums





class utter_dataset(data.Dataset):
    def __init__(self,nums_each):
        f = open('mfcc.txt', 'rb+')
        self.Speakers_Utterance = pickle.load(f)
        self.n = nums_each
        self.maxlength = 0
        for speaker in list(self.Speakers_Utterance.keys()):
            for utter in self.Speakers_Utterance[speaker][1]:
                if utter.shape[0]>self.maxlength:
                    self.maxlength = utter.shape[0]

        f.close()

    def __getitem__(self, index):
        speaker_id = index//self.n
        label = list(self.Speakers_Utterance.keys())[speaker_id]
        feature = self.Speakers_Utterance[label][1][index%162]
        feature = audio_padding_wrap(feature,self.maxlength)
        feature = torch.from_numpy(feature).float()
        return  feature,speaker_id

    def __len__(self):
        length=0
        for speaker in list(self.Speakers_Utterance.keys()):
            length+= len(self.Speakers_Utterance[speaker][1])
        return length




if __name__ == '__main__':
    x = Make_VoxCeleb_Dataset()
    x.load_mfccs('Preprocessing/')
    z,q = x[100]
    print(z,q)
    # if 'mfcc.txt' not in os.listdir():
    #     print('Please run preprocess.py first to preprocess all data')

    # else:
    #     u_dataset=utter_dataset(162)
    #     index=1000
    #     feature,label = u_dataset[index]
    #     number = index%162


    #     x,y,z=data.random_split(u_dataset,[800,200,620])


    #     print(label,' \'s ',str(number),'th audio')
    #     print(feature)

    #     print('Shape of MFCC:', feature.shape)
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)
    im = ax.imshow(z.transpose(0,1), cmap=plt.cm.jet, aspect='auto')
    plt.title('Normalized MFCC')
    plt.ylabel('MFCC Coefficient')
    plt.xlabel('Frames')
    plt.colorbar(im, cax=make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05))
    ax.set_xticks(np.arange(0, 100, 1000), minor=False);
    plt.show()

