import torch
import yaml
import torch.optim as optim
import torch.nn as nn
import sys
sys.path.append(".")
from Models.model import TDXN,CTDNN
from Models import FTDNN
from Preprocessing.make_dataset import Make_VoxCeleb_Dataset

if __name__=="__main__":

    config_file = open('TrainTest/config.yml')
    config = yaml.load(config_file, Loader=yaml.Loader)
    config_file.close()


    num_epoch=config['num_epoch']
    batch_size=config['batch_size']


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    Dataset = Make_VoxCeleb_Dataset()
    Dataset.load_mfccs(config['mfccs_folder'])

    trainloader = torch.utils.data.DataLoader(Dataset, batch_size=2,
                                            shuffle=True, num_workers=2)


    net = CTDNN(25,Dataset.num_speakers).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    print("start training")

    for epoch in range(num_epoch):

        running_loss = 0
        total = 0
        correct = 0

        for i,data in enumerate(trainloader):
            inputs ,labels = data
            inputs=inputs.to(device)
            labels=labels.to(device)
            out = net(inputs)

            loss = criterion(out,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(out, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum()
            if i % 10 == 0:
                print(epoch + 1, i + 1, 'loss:', running_loss, 'accuracy:{:.2%}'.format(correct.item() / total))
                running_loss = 0
                total = 0
                correct = 0


    torch.save(net.state_dict(), 'net.pth')





