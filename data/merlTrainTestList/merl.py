import scipy.io
import random
import math
import sys
import glob
import os
import os.path
from subprocess import call


class MERL_Shopping(object):
    classFile = './merlTrainTestList/classInd.txt'
    testFile = './merlTrainTestList/testlist01.txt'
    valFile = './merlTrainTestList/validationlist01.txt'
    trainFile = './merlTrainTestList/trainlist01.txt'
    inputVids = './Videos_MERL_Shopping_Dataset'
    outputVids = './output'
    labelFolder = './Labels_MERL_Shopping_Dataset'
    categories = dict()
    labels = ['ReachToShelf', 'RetractFromShelf',
              'HandInShelf', 'InspectProduct', 'InspectShelf']
    framerate = 30

    def __init__(self):
        self.checkDataFolder()

        self.checkOutputFolder()
        self.scanVideoFiles()
        self.pickRandom()

    def checkOutputFolder(self):
        try:
            os.stat(self.outputVids)
        except:
            os.mkdir(self.outputVids)

    def checkDataFolder(self):
        self.checkAndMakeFolder(folderPath='./merlTrainTestList')

    def checkAndMakeFolder(self, folderPath):
        try:
            os.stat(folderPath)
        except:
            os.mkdir(folderPath)

    def writeClassFile(self):
        # earse content file
        with open(self.classFile, 'w'):
            pass
        # write content file
        with open(self.classFile, 'a') as f:
            for i, key in enumerate(self.categories.keys()):
                line = str(i + 1) + ' ' + key + '\n'
                f.write(line)
        f.close
        print('write class file done')

    def writeOutputFile(self, list, file):
        # earse content file
        with open(file, 'w'):
            pass
        # write content file
        with open(file, 'a') as f:
            for filepath in list:
                filename = os.path.basename(filepath)
                cat = filename.split('.')[0]
                cat = cat.split('_')
                cat = cat[len(cat) - 1]
                line = cat + '/' + filename + '\n'
                f.write(line)
        f.close
        print('write file done')

    def checkFileExisted(self, filepath):
        return bool(os.path.exists(filepath))

    def checkAlreadySplit(self):
        return

    def spiltVideoByLabel(self, labelFilePath, src):
        # read matlab file
        mat = scipy.io.loadmat(labelFilePath)
        filename = os.path.basename(src)
        for i in range(len(mat['tlabs'])):
            label = self.labels[i]
            frames = mat['tlabs'][i][0]
            for i in range(len(frames)):
                # get start frame
                start_frame = int(frames[i][0])
                start_time = start_frame / self.framerate
                # get end frame
                end_frame = int(frames[i][1])
                end_time = (end_frame - start_frame) / self.framerate

                folder = self.outputVids + '/' + label
                dest_name = filename.split(
                    '.')[0] + '_' + str(start_frame) + '_' + str(end_frame) + '_' + label + '.mp4'

                dest = folder + '/' + dest_name

                # check folder existed or not
                self.checkAndMakeFolder(folder)
                if not self.checkFileExisted(filepath=dest):
                    call(["ffmpeg", '-i', src, '-ss',
                          str(start_time), '-t', str(end_time), dest])
                    print("Splited %s" % (src))
                    print(start_frame, end_frame, start_time, end_time, dest)

                if label in self.categories:
                    self.categories[label].append(
                        label + '/' + dest_name)
                else:
                    self.categories[label] = [
                        label + '/' + dest_name]

    # get all video files
    # read coarse file and then split video to multiparts with label
    def scanVideoFiles(self):
        for root, dirs, files in os.walk(self.inputVids):
            for file in files:
                if file.endswith(".mp4"):
                    mp4FilePath = root + '/' + file
                    filename = file.split('.')[0]
                    filename = filename.replace('_crop', '_label')
                    # generate coarse file path
                    labelFilePath = self.labelFolder + '/' + filename + '.mat'
                    # check coarse file existed or not?
                    if self.checkFileExisted(filepath=labelFilePath):
                        # read coarse file to get label
                        self.spiltVideoByLabel(
                            labelFilePath, src=mp4FilePath)
                    else:
                        print('not found %s %s' % (root, mp4FilePath))

    def diff(self, li1, li2):
        li_dif = [i for i in li1 + li2 if i not in li1 or i not in li2]
        return li_dif

    def getRandomList(self, dataList):
        if dataList != []:
            elem = random.choice(dataList)
            dataList.remove(elem)
        else:
            elem = None
            # print(dataList)
        return dataList, elem

    def getNumRandomList(self, datalist, num):
        newlist = []
        for x in range(0, num):
            datalist, item = self.getRandomList(datalist)
            newlist.append(item)
        return datalist, newlist

    def pickRandom(self):
        val_list = []
        test_list = []
        train_list = []
        # print("%s" % (self.categories))
        cats = self.categories
        for key in cats.keys():
            total = len(cats[key])
            if total >= 10:
                val_total = math.trunc(total * (1 / 10))
                if val_total == 0:
                    val_total = 1
                test_total = math.trunc(total * (3 / 10))
                if test_total == 0:
                    test_total = 1
                train_total = total - val_total - test_total
                cats[key], tmp_list = self.getNumRandomList(
                    cats[key], val_total)

                val_list = val_list + tmp_list

                cats[key], tmp_list = self.getNumRandomList(
                    cats[key], test_total)
                test_list = test_list + tmp_list
                train_list = train_list + cats[key]
                print(total, val_total, test_total, train_total)
        # print(test_list)
        if len(val_list) > 0:
            self.writeOutputFile(val_list, self.valFile)
        if len(test_list) > 0:
            self.writeOutputFile(test_list, self.testFile)
        if len(train_list) > 0:
            self.writeOutputFile(train_list, self.trainFile)

        self.writeClassFile()


MERL_Shopping()
