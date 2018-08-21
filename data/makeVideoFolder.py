import glob
import os
import os.path
import sys
import shutil


class MakeVideoFolder(object):
    """docstring for MakeVideoFolder"""
    folders = ['./train/', './test/', './validation/']

    def __init__(self):
        super(MakeVideoFolder, self).__init__()
        self.scanFolder()

    def scanFolder(self):
        for folder in self.folders:
            class_folders = glob.glob(folder + '*')
            for vid_class in class_folders:
                class_files = glob.glob(vid_class + '/*.*')
                for video_path in class_files:
                    classname, filename_no_ext, filename = self.get_video_parts(
                        video_path)
                    folder_name = filename_no_ext.split('-')[0]
                    destFolder = vid_class + '/' + folder_name
                    self.checkAndMakeFolder(destFolder)

                    shutil.move(video_path, destFolder + '/' + filename)
                    print('move file %s' % (filename_no_ext))

    def checkAndMakeFolder(self, folderPath):
        try:
            os.stat(folderPath)
        except:
            os.mkdir(folderPath)

    def get_video_parts(self, video_path):
        """Given a full path to a video, return its parts."""
        parts = video_path.split('/')
        filename = parts[3]
        filename_no_ext = filename.split('.')[0]
        classname = parts[2]
        return classname, filename_no_ext, filename


MakeVideoFolder()
