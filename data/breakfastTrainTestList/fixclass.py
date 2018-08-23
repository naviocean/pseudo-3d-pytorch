from random import shuffle


class FixClass(object):
    class_list = []
    train_file = './trainlist01.txt'
    class_file = './classInd.txt'

    def __init__(self):
        super(FixClass, self).__init__()
        # self.readFile()
        # self.writeClassFile()
        self.shuffeFile()

    def readFile(self):
        with open(self.train_file) as fin:
            for row in list(fin):
                class_name = row.strip().split('/')[0]
                if class_name not in self.class_list:
                    self.class_list.append(class_name)
        self.class_list.sort()
        print(len(self.class_list))

    def writeClassFile(self):
        # earse content file
        with open(self.class_file, 'w'): pass
        # write content file
        with open(self.class_file, 'a') as f:
            for i, key in enumerate(self.class_list):
                line = str(i + 1) + ' ' + key + '\n';
                f.write(line)
        f.close
        print('write class file done')

    def shuffeFile(self):
        files = ['./trainlist01.txt','testlist01.txt','validationlist01.txt']

        for i in range(len(files)):
            file = files[i]
            with open(file) as fin:
                rows = [row.strip() for row in list(fin)]

            shuffle(rows)
            # earse content file
            with open(file, 'w'): pass
            # write content file
            with open(file, 'a') as f:
                for row in rows:
                    line = row + '\n';
                    f.write(line)
            f.close
            print('write file done')


FixClass()
