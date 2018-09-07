from tensorboardX import SummaryWriter

class Visualizer(object):
    def __init__(self, logdir='./runs'):
        self.writer = SummaryWriter(logdir)

    def write_graph(self, model, dummy_input):
        self.writer.add_graph(model, (dummy_input, ))

    def write_summary(self, info_acc, info_loss, epoch):
        self.writer.add_scalars('Accuracy', info_acc, epoch)
        self.writer.add_scalars('Loss', info_loss, epoch)

    def write_histogram(self, model, step):
        for name, param in model.named_parameters():
            self.writer.add_histogram(name, param.clone().cpu().data.numpy(), step)

    def writer_close(self):
        self.writer.export_scalars_to_json("./all_scalars.json")
        self.writer.close()