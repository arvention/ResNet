import os
import torch
import time
import datetime
from utils import to_var

from model import FasterRCNN


class Solver(object):

    DEFAULTS = {}

    def __init__(self, version, data_loader, config):
        """
        Initializes a Solver object
        """

        # data loader
        self.__dict__.update(Solver.DEFAULTS, **config)
        self.version = version
        self.data_loader = data_loader

        self.build_model()

        # TODO: build tensorboard

        # start with a pre-trained model
        if self.pretrained_model:
            self.load_pretrained_model()

    def build_model(self):
        """
        Instantiates the model, loss criterion, and optimizer
        """

        # instantiate Faster R-CNN model
        self.model = FasterRCNN(self.input_channels, self.class_count)

        # TODO: instantiate loss criterion

        # TODO: instantiate optimizer

        # print network
        self.print_network(self.model, 'Faster R-CNN')

        if torch.cuda.is_available() and self.use_gpu:
            self.model.cuda()
            # TODO: set criterion to cuda

    def print_network(self, model, name):
        """
        Prints the structure of the network and the total number of parameters
        """
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
            print(name)
            print(model)
            print("The number of parameters: {}".format(num_params))

    def load_pretrained_model(self):
        """
        loads a pre-trained model from a .pth file
        """
        self.model.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}.pth'.format(self.pretrained_model))))
        print('loaded trained model ver {}'.format(self.pretrained_model))

    def print_loss_log(self, start_time, iters_per_epoch, e, i, loss):
        """
        Prints the loss and elapsed time for each epoch
        """
        total_iter = self.num_epochs * iters_per_epoch
        cur_iter = e * iters_per_epoch + i

        elapsed = time.time() - start_time
        total_time = (total_iter - cur_iter) * elapsed / (cur_iter + 1)
        epoch_time = (iters_per_epoch - i) * elapsed / (cur_iter + 1)

        epoch_time = str(datetime.timedelta(seconds=epoch_time))
        total_time = str(datetime.timedelta(seconds=total_time))
        elapsed = str(datetime.timedelta(seconds=elapsed))

        log = "Elapsed {}/{} -- {}, Epoch [{}/{}] Iter [{}/{}]," \
              "loss: {:.4f}".format(
               elapsed,
               epoch_time,
               total_time,
               e + 1,
               self.num_epochs,
               i + 1,
               iters_per_epoch,
               loss
               )

        # TODO: add tensorboard

        print(log)

    def save_model(self, e, i):
        """
        Saves a model per e epoch and i iteration
        """
        path = os.path.join(
            self.model_save_path,
            '{}_{}_{}.pth'.format(self.version, e + 1, i + 1)
        )
        torch.save(self.model.state_dict(), path)

    def train(self):
        """
        Training process
        """
        # TODO: add training process
        pass

    def model_step(self, images, labels):
        """
        A step for each iteration
        """

        # set model in training mode
        self.model.train()

        # empty the gradients of the model through the optimizer
        self.optimizer.zero_grad()

        # forward pass
        output = self.model(images)

        # compute loss
        loss = self.criterion(output, labels.squeeze())

        # compute gradients using back propagation
        loss.backward()

        # update parameters
        self.optimizer.step()

        # return loss
        return loss

    def train_evaluate(self, e):
        """
        Evaluates the performance of the model using the train dataset
        """
        top_1_correct, top_5_correct, total = self.eval(self.data_loader)
        log = "Epoch[{}/{}]--top_1_acc: {:.4f}--top_5_acc: {:.4f}".format(
            e + 1,
            self.num_epochs,
            top_1_correct / total,
            top_5_correct / total
        )
        print(log)
        return top_1_correct / total, top_5_correct / total

    def test(self):
        """
        Evaluates the performance of the model using the test dataset
        """
        top_1_correct, top_5_correct, total = self.eval(self.data_loader)
        log = "top_1_acc: {:.4f}--top_5_acc: {:.4f}".format(
            top_1_correct / total,
            top_5_correct / total
        )
        print(log)

    def eval(self, data_loader):
        """
        Returns the count of top 1 and top 5 predictions
        """

        self.model.eval()

        top_1_correct = 0
        top_5_correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in data_loader:

                images = to_var(images, self.use_gpu)
                labels = to_var(labels, self.use_gpu)

                output = self.model(images)
                total += labels.size()[0]

                # top 1
                # get the max for each instance in the batch
                _, top_1_output = torch.max(output.data, dim=1)

                top_1_correct += torch.sum(torch.eq(labels.squeeze(),
                                                    top_1_output))

                # top 5
                _, top_5_output = torch.topk(output.data, k=5, dim=1)
                for i, label in enumerate(labels):
                    if label in top_5_output[i]:
                        top_5_correct += 1

        return top_1_correct.item(), top_5_correct, total
