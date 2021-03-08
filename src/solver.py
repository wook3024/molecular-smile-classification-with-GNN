import sys
import timeit
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sklearn.metrics as metrics
import preprocess as pp
import constant
from adamp import AdamP, SGDP
from model import MolGNN
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from warmup_scheduler import GradualWarmupScheduler


np.random.seed(2345)
torch.manual_seed(2345)



class MolDataset(Dataset):
    '''Dataset class for the bionsight dataset.'''
    def __init__(self, dataset, batch_size):
        np.random.shuffle(dataset)
        self.dataset = dataset
        self.batch_size = batch_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        start, end = idx*self.batch_size, (idx+1)*self.batch_size
        data_batch = list(zip(*self.dataset[start:end]))
        return data_batch


class FocalLoss(nn.Module):
    ''' https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65938 '''
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class Solver():
    '''Solver for training and testing MolGNN.'''
    def __init__(self):
        '''Model configuration'''
        self.input_dim = constant.INPUT_DIM
        self.hidden_dim = constant.HIDDEN_DIM
        self.out_dim = constant.OUT_DIM
        self.layer_hidden = constant.LAYER_HIDDEN
        self.layer_output = constant.LAYER_OUTPUT
        self.skip_connection_type = constant.SKIP_CONNECTION_TYPE
        self.norm_type = constant.NORM_TYPE

        '''Training configuration'''
        self.batch_train = constant.BATCH_TRAIN
        self.batch_test = constant.BATCH_TEST
        self.lr = constant.LR
        self.lr_decay = constant.LR_DECAY
        self.decay_interval = constant.DECAY_INTERVAL
        self.iteration = constant.ITERATION
        self.loss_type = constant.LOSS_TYPE
        self.load_model_path = constant.LOAD_MODEL_PATH

        '''Miscellaneous'''
        self.mode = constant.MODE
        self.dataset = constant.DATASET
        self.model_type = constant.MODEL_TYPE
        self.use_augmentation = constant.USE_AUGMENTATION
        self.radius = constant.RADIUS
        self.checkpoint = constant.CHECKPOINT
        self.root_dir = constant.ROOT_DIR

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def create_datasets(self, dataset, radius, device, use_augmentation):
        '''Create bionsight datasets using preprocessing and augmentation'''
        createDatasets = pp.CreateDatasets(dataset, radius, device, use_augmentation)
        train_dataset, dev_dataset, test_dataset, N_fingerprints = \
            createDatasets.create_datasets(dataset, radius, device)

        train_dataset = MolDataset(train_dataset, self.batch_train)
        dev_dataset = MolDataset(dev_dataset, self.batch_test)
        test_dataset = MolDataset(test_dataset, self.batch_test)

        print('='*110)
        print('training data samples:', len(train_dataset))
        print('development data samples:', len(dev_dataset))
        print('test data samples:', len(test_dataset))

        del createDatasets
        return train_dataset, dev_dataset, test_dataset, N_fingerprints


    def build_model(self, N_fingerprints):
        '''Create a model and optimizer'''
        model = MolGNN(
                N_fingerprints, self.input_dim, self.hidden_dim, self.out_dim, 
                self.layer_hidden, self.layer_output, self.device, 
                self.loss_type, self.norm_type, self.skip_connection_type).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        # optimizer = SGDP(model.parameters(), lr=0.1, weight_decay=1e-5, momentum=0.9, nesterov=True)
        # scheduler_steplr = StepLR(optimizer, step_size=10, gamma=0.1)
        # scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler_steplr)
        print('='*110)
        print('model parameters:',
            sum([np.prod(p.size()) for p in model.parameters()]))
        return model, optimizer
    

    def classification_loss(self):
        '''Compute binary cross entropy or Flcal loss.'''
        if self.loss_type == 'BCELoss':
            return nn.BCEWithLogitsLoss()
        elif self.loss_type == 'FocalLoss':
            return FocalLoss(logits=True)


    def cal_complete_time(self, time, iteration):
        '''Calculate training completion time'''
        minutes = time * iteration / 60
        hours = int(minutes / 60)
        minutes = int(minutes - 60 * hours)
        print('='*110)
        print(f'training time: {hours}h {minutes}m')
        print('='*110)


    def make_result_file(self):
        '''Create folder path and create folder to save results'''
        os.makedirs(self.root_dir, exist_ok=True)
        file_result = f'{self.root_dir}/result.txt'
        result = 'epoch,loss,dev(log_loss/aucroc),test(log_loss/test_aucroc)'
        with open(file_result, 'w') as f:
            f.write(result + '\n')

        return file_result


    def clamp(self, num, min_value, max_value):
        '''Function to clip values between the minimum and maximum values'''
        return max(min(num, max_value), min_value)


    def test(self, dataset, model):
        '''Calculate log_logg, aucprc, aucroc metric using the trained model'''
        y_pred, y_test = [], []
        model.eval()
        for i in range(0, len(dataset)//self.batch_test):
            test_data = dataset[i]
            predicted_scores, correct_labels = model.forward(test_data, train=False)
            y_pred.extend(predicted_scores)
            y_test.extend(correct_labels)

        y_test = [float(item) for item in y_test]
        y_pred = [float(item) for item in y_pred]
        # y_pred = [float(clamp(item,0.25,1.0)) for item in y_pred]

        log_loss = metrics.log_loss(y_test, y_pred, labels=[0, 1])

        precision, recall, _ = metrics.precision_recall_curve(y_test, y_pred, pos_label=1)
        aucprc = metrics.auc(recall, precision)

        fpr_roc, tpr_roc, _ = metrics.roc_curve(y_test, y_pred, pos_label=1)
        aucroc = metrics.auc(fpr_roc, tpr_roc)
        return {"log_loss": round(log_loss,4), "aucprc": round(aucprc,4), "aucroc": round(aucroc,4)}


    def save_result(self, model, result, filename, epoch):
        '''Save the training results using the model you used'''
        with open(filename, 'a') as f:
            f.write(result + '\n')
        
        if (epoch-1) % 50 == 0:
            torch.save(model.state_dict(), 
                f'{self.root_dir}/{epoch}.pth')


    def load_model(self):
        '''Load to use pretrained model'''
        print('# load model...', self.load_model_path)
        load_model_state_dict = torch.load(self.load_model_path)
        del load_model_state_dict['embed_fingerprint.weight']

        model_state_dict = self.model.state_dict()
        pretrained_dict = {k: v for k, v in load_model_state_dict.items() \
            if k in model_state_dict}
        model_state_dict.update(pretrained_dict)
        self.model.load_state_dict(model_state_dict)


    def run(self):  
        '''Run the entire train or test process'''     
        train_dataset, dev_dataset, test_dataset, N_fingerprints = \
            self.create_datasets(self.dataset, self.radius, self.device, self.use_augmentation)
        model, optimizer = self.build_model(N_fingerprints)
        criterian = self.classification_loss()
    
        file_result = self.make_result_file()

        if self.load_model_path != "":
            load_model()

        if self.mode == 'train':
            start = timeit.default_timer()
            if self.mode == 'train':
                for epoch in range(1, self.iteration+1):
                    if epoch % self.decay_interval == 0:
                        '''Decay learning rates'''
                        optimizer.param_groups[0]['lr'] *= self.lr_decay

                    '''Train the model'''
                    loss_total = 0
                    model.train()
                    for i in range(0, len(train_dataset)//self.batch_train):
                        train_data = train_dataset[i]
                        predicted_values, correct_values = model.forward(train_data, train=True)
                        loss = criterian(predicted_values, correct_values.float())

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        loss_total += loss.item()

                    '''Miscellaneous'''
                    time = timeit.default_timer() - start
                    if epoch == 1:
                        self.cal_complete_time(time, self.iteration)

                    prediction_test = self.test(test_dataset, model)
                    prediction_dev = self.test(dev_dataset, model)
                    result = f"epoch:{epoch} loss:{round(loss_total,4)} \t" +\
                        f"dev(log_loss/aucoc):{prediction_dev['log_loss']}/{prediction_dev['aucroc']} \t" +\
                        f"test(log_loss/test_aucroc):{prediction_test['log_loss']}/{prediction_test['aucroc']}"
                    print(result)

                    self.save_result(model, str(result), file_result, epoch)

        if self.mode == 'test':
            results = self.test(dataset_test)
            print(results)