import numpy as np
import torch
import sklearn
import sklearn.metrics
import math
from torch.utils.data import TensorDataset, DataLoader
from collections import namedtuple
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

def get_dataloaders(data:tuple[np.ndarray], model_type:str, len_sequence:int,
                    batch_size:int, shuffle:bool=True) -> tuple[DataLoader]:
    """
    Creates dataloader for PyTorch neural networks.
    Args:
        data: Tuple of test and train data.
        model_type: Reshapes input data if 'GRU' or 'CNN'.
        len_sequence: Length of time sequence.
        batch_size: Size of batches used in model training.
        shuffle: Enable shuffling of data after each epoch.
    """
    # unpack input data
    x_train, x_test, y_train, y_test = data

    # calculate number of sensor features
    num_sensor_features = x_train.shape[1] // len_sequence

    # bring arrays into correct shape for given model architecture
    # CNN: (batch x channel x sequence)
    # GRU: (batch x sequence x features)
    # where channel and features correspond to the number of used
    # sensor features respectively
    gru_shape = (-1, num_sensor_features, len_sequence)
    cnn_shape = (-1, num_sensor_features, len_sequence)
    if model_type == 'GRU':
        x_train = x_train.reshape(gru_shape).swapaxes(1, 2)
        x_test = x_test.reshape(gru_shape).swapaxes(1, 2)
    elif model_type == 'CNN':
        x_train = x_train.reshape(cnn_shape)
        x_test = x_test.reshape(cnn_shape)

    # create torch Datasets
    train_set = TensorDataset(
        torch.tensor(x_train, dtype=torch.float),
        torch.tensor(y_train, dtype=bool)
    )
    test_set = TensorDataset(
        torch.tensor(x_test, dtype=torch.float),
        torch.tensor(y_test, dtype=bool)
    )

    # create torch DataLoaders
    train_loader = DataLoader(train_set, batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_set, batch_size, shuffle=shuffle)

    return train_loader, test_loader

def get_device(override:str|None=None) -> torch.device:
    """
    Returns most powerful available PyTorch device unless overriden.
    Args:
        override: Device that should be used instead.
    """
    # use 'override' device if given
    if override is not None:
        return torch.device(override)
    
    # choose most powerful available backend
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    return torch.device(device)

# define namedtuples for better code readability
TrainingReport = namedtuple('TrainingReport', "train test")
TrainingHistory = namedtuple('TrainingHistory', "weight_updates error loss")

def plot_training_run(training_history:TrainingReport, title:str=None) -> None:
    """
    Plot training progress in terms of accuracy and loss.
    Args:
        training_history: Output of the train_loop function.
        title: Plot title.
    """
    # set plotly theme and define color sequence
    plotly.io.templates.default = "plotly_white"
    color_sequence = px.colors.qualitative.G10

    # create figure with 2 subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Accuracy', 'Loss'])

    # add lines for training/test accuracy and loss
    fig.add_trace(
        go.Scatter(
            x=training_history.train.weight_updates,
            y=training_history.train.error,
            mode='lines',
            name='Training error',
            marker=dict(color=color_sequence[0])
        ),
        row=1,
        col=1
    )
    fig.add_trace(
        go.Scatter(
            x=training_history.test.weight_updates,
            y=training_history.test.error,
            mode='lines',
            name='Testing error',
            marker=dict(color=color_sequence[1])
        ),
        row=1,
        col=1
    )
    fig.add_trace(
        go.Scatter(
            x=training_history.train.weight_updates,
            y=training_history.train.loss,
            mode='lines',
            name='Training loss',
            marker=dict(color=color_sequence[2])
        ),
        row=1,
        col=2
    )
    fig.add_trace(
        go.Scatter(
            x=training_history.test.weight_updates,
            y=training_history.test.loss,
            mode='lines',
            name='Testing loss',
            marker=dict(color=color_sequence[3])
        ),
        row=1,
        col=2
    )
    # set the title and resize figure
    fig.update_layout(title_text=title, height=400, width=750)

    # update the axes labels
    fig.update_yaxes(title_text='Accuraccy', row=1, col=1)
    fig.update_xaxes(title_text='Weight updates', row=1, col=1)
    fig.update_yaxes(title_text='Loss', row=1, col=2)
    fig.update_xaxes(title_text='Weight updates', row=1, col=2)

    # display the figure
    fig.show()

class TorchMLP(torch.nn.Module):
    def __init__(self, input_size:int, output_size:int, hidden_layers:tuple):
        super().__init__()
        self.model_type = 'MLP'
        self.layers = torch.nn.Sequential()
        layer_sizes = (input_size, *hidden_layers)

        for i, size in enumerate(layer_sizes[1:], 1):
            prev_size = layer_sizes[i-1]
            self.layers.extend((
                torch.nn.Linear(prev_size, size),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(size, affine=False),
            ))

        self.layers.append(
            torch.nn.Linear(layer_sizes[-1], output_size),
        )

    def forward(self, x):
        return self.layers(x)

class TorchGRU(torch.nn.Module):
    def __init__(self, input_size:int, output_size:int, hidden_size:int, rnn_layers:int):
        super().__init__()
        self.model_type = 'GRU'
        self.rnn = torch.nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=rnn_layers,
                dropout=0.3,
                batch_first=True
        )
        self.linear = torch.nn.Linear(hidden_size, output_size)
        
    def forward(self, x, h):
        x, h = self.rnn(x, h)
        x = self.linear(x)
        return x, h

# Define the 1D CNN model
class CNN1D(torch.nn.Module):
    def __init__(self, input_size:int, output_size:int, hidden_size:int):
        super(CNN1D, self).__init__()
        self.model_type = 'CNN'
        self.conv1 = self.convbn(input_size,8,15)
        self.conv2 = self.convbn(8,8,11)
        self.conv3 = self.convbn(8,16,7)
        self.fc1 = torch.nn.Linear(112, hidden_size)
        self.bn = torch.nn.BatchNorm1d(hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)
        self.relu = torch.nn.ReLU()

    def convbn(self, ci,co,ksz,s=1,pz=1):
        return torch.nn.Sequential(
            torch.nn.Conv1d(ci,co,ksz,stride=s,padding=pz),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm1d(co),
            torch.nn.MaxPool1d(kernel_size=2, stride=2)
        ) # change pool size and stride to compress more info
          # large conv kernel size get better acc, better than ksz=3, but ksz=17, 15, 13 get 65 acc
          # current get 70 acc

    def forward(self, x):
        bsz = x.size(0)
        x = torch.nn.Dropout(0.2)(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(bsz, -1)
        # print(x.shape)
        x = self.relu(self.fc1(x))
        x = self.bn(x)
        x = torch.nn.Dropout(0.2)(x)
        x = self.fc2(x)
        return x

class ResidualBlock(torch.nn.Module):
    def __init__(self, input_size:int, output_size:int, stride:int=1, downsample:int=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(input_size, output_size, kernel_size = 3, stride = stride, padding = 1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(output_size),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv1d(output_size, output_size, kernel_size = 3, stride = 1, padding = 1),
            torch.nn.BatchNorm1d(output_size)
        )
        self.downsample = downsample
        self.relu = torch.nn.ReLU()
        self.out_channels = output_size

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet1D(torch.nn.Module):
    """
    Resnet architecture
    adapted from https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
    """
    def __init__(self, input_size, output_size, block, layers):
        super(ResNet1D, self).__init__()
        self.model_type = 'CNN'
        self.inplanes = 64
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(input_size, 64, kernel_size = 7, stride = 2, padding = 3),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU()
        )
        self.maxpool = torch.nn.MaxPool1d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
        self.avgpool = torch.nn.AvgPool1d(4, stride=1)
        self.fc = torch.nn.Linear(512, output_size)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = torch.nn.Sequential(
                torch.nn.Conv1d(self.inplanes, planes, kernel_size=1, stride=stride),
                torch.nn.BatchNorm1d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = torch.nn.Dropout(0.5)(x)
        x = self.fc(x)

        return x

def train_loop(model:torch.nn.Module, train_loader:DataLoader, test_loader:DataLoader,
               loss_fun, optimizer:torch.optim.Optimizer, num_epochs:int,
               device:torch.device) -> TrainingReport:
    """
    Function for training PyTorch neural networks.
    Args:
        model: Model to be trained.
        train_loader: Training data.
        test_loader: Testing data.
        loss_fun: Loss function for evaluating model.
        optimizer: Optimizer for training.
        num_epochs: Number of epochs.
        device: Device on which model should be trained.
    """
    # print out device that is used for training
    print(f"Training on {device.type.upper()}.")
    # move model to specified device
    model = model.to(device)
    # save total number of samples in training dataset
    size = len(train_loader.dataset)
    # save number of batches
    num_batches = len(train_loader)

    # initialize training history variables
    weight_updates = 0
    training_history = TrainingReport(
        train=TrainingHistory(weight_updates=[], error=[], loss=[]),
        test=TrainingHistory(weight_updates=[], error=[], loss=[])
    )

    # main training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        sample_count = 0
        for batch, (x, y) in enumerate(train_loader):
            # move data to specified device
            x = x.to(device)
            y = y.float().to(device)
            # initialize hidden states if using recurrent network
            if model.model_type == 'GRU':
                h0 = torch.zeros((model.rnn.num_layers, x.shape[0], model.rnn.hidden_size), device=device)
            # enable training mode of neural network
            model.train()

            # compute loss of minibatch
            if model.model_type == 'GRU':
                output, _ = model(x, h0)
                output = output[:, -1, :]
            else:
                output = model(x)
            
            loss = loss_fun(output, y)

            # perform backpropagation and gradient descent step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # keep track of current training quantities
            loss_value = loss.item()
            weight_updates += 1
            sample_count += x.shape[0]

            # calculate performance on training and testing data every 1000 weight updates
            if weight_updates % 1000 == 0:
                test_error, test_loss = test_loop(model, test_loader, loss_fun, device)
                training_history.test.weight_updates.append(weight_updates)
                training_history.test.error.append(test_error)
                training_history.test.loss.append(test_loss)

                train_error, train_loss = test_loop(model, train_loader, loss_fun, device)
                training_history.train.weight_updates.append(weight_updates)
                training_history.train.error.append(train_error)
                training_history.train.loss.append(train_loss)

            # give updates during training
            if batch == 0 or batch % (int(num_batches/3)) == 0:
                print(f"  training loss: {loss_value:>7f}  [{sample_count:>5d}/{size:>5d}]")

        # give update error on test data after each epoch
        test_error, test_loss = test_loop(model, test_loader, loss_fun, device)
        print(f" validation error {test_error*100:.2f} %")

    # return data about training history
    return training_history

def test_loop(model:torch.nn.Module, test_loader:DataLoader, loss_fun,
              device:torch.device) -> tuple[float]:
    """
    Function for testing PyTorch neural networks.
    Args:
        model: Model to be tested.
        test_loader: Testing data.
        loss_fun: Loss function for evaluating model.
        device: Device on which model should be trained.
    """
    # move model to specified device
    model.to(device)
    # enter evaluation mode
    model.eval()
    # save total number of samples in training dataset
    len_data = len(test_loader.dataset)

    # initialize error and loss values
    error_num = 0
    loss_value = 0
    with torch.no_grad():
        for x, y in test_loader:
            # move data to specified device
            x = x.to(device)
            y = y.float().to(device)
            # initialize hidden states if using recurrent network
            if model.model_type == 'GRU':
                h0 = torch.zeros((model.rnn.num_layers, x.shape[0], model.rnn.hidden_size), device=device)

            # Compute prediction and loss
            if model.model_type == 'GRU':
                output, _ = model(x, h0)
                output = output[:, -1, :]
            else:
                output = model(x)
            
            loss = loss_fun(output, y)
            prediction = (output > 0)

            error_num += (prediction != y).any(dim=1).sum().item()
            loss_value += loss.item()*y.shape[0]

    return error_num/len_data, loss_value/len_data


def calculate_confusion_matrices_torch(model:torch.nn.Module, test_loader:DataLoader, device:torch.device):
    """
    Calculate confusion matrices on test data.
    Args:
        model: Model to be tested.
        test_loader: Testing data.
        device: Device on which model should be trained.
    """
    # move model to specified device
    model.to(device)
    # enter evaluation mode
    model.eval()

    # determine number of classes
    num_classes = len(test_loader.dataset[0][1])
    # initialize confusion matrices
    confusion_matrices = np.zeros((num_classes, 2, 2))
    with torch.no_grad():
        for x, y in test_loader:
            # move data to specified device
            x = x.to(device)
            y = y.float().to(device)
            # initialize hidden states if using recurrent network
            if model.model_type == 'GRU':
                h0 = torch.zeros((model.rnn.num_layers, x.shape[0], model.rnn.hidden_size), device=device)

            # compute prediction
            if model.model_type == 'GRU':
                output, _ = model(x, h0)
                output = output[:, -1, :]
            else:
                output = model(x)
            prediction = (output > 0)

            # add confusion matrix for batch to total confusion matrix
            confusion_matrices += sklearn.metrics.multilabel_confusion_matrix(
                y.detach().cpu(), prediction.detach().cpu()
            )

    return confusion_matrices

def plot_confusion_matrices(confusion_matrices:np.ndarray, class_list:list[str],
                            num_cols:int=3, title:str=None):
    """
    Plots all confusion matrices in a subfigure.
    """
    # calculate number of subfigure rows
    num_rows = math.ceil(confusion_matrices.shape[0]/num_cols)
    # create subfigure plot
    fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=class_list)

    for i in range(confusion_matrices.shape[0]):
        col_idx = i%num_cols+1
        row_idx = i//num_cols+1

        fig.add_heatmap(
            z=confusion_matrices[i, ::-1, :].astype(int),
            col=col_idx,
            row=row_idx,
            x=['False', 'True'],
            y=['True', 'False'],
            coloraxis = "coloraxis",
            name=class_list[i],
            text=np.char.mod('%d', confusion_matrices[i, ::-1, :]),
            texttemplate="%{text}"
        )
        
        if row_idx == num_rows:
            fig.update_xaxes(title_text='Predicted', row=row_idx, col=col_idx)
        if col_idx == 1:
            fig.update_yaxes(title_text='Actual', row=row_idx, col=col_idx)

    fig.update_layout(title=title, width=750, height=600, coloraxis={'colorscale':'Inferno'})

    fig.show()

def resample_data(x:np.ndarray, y:np.ndarray, max_count:int) -> tuple[np.ndarray]:
    """
    Perform resampling on datset subsampling and supersampling under/overrepresented classes.
    Args:
        x: Input data.
        y: True labels.
        max_count: Number of samples that should approximately be reached by each class.
    """
    idx = []
    # loop over all classes
    for i in range(y.shape[1]):
        if y[:, i].sum() <= max_count:
            # if class is underrepresented, randomly choose samples with replacement
            idx.extend(np.random.choice(np.where(y[:, i] == 1)[0], max_count, replace=True))
        else:
            # if class is overrepresented, choose samples randomly without replacement
            # and include only samples that were not already chosen before
            idx.extend(set(np.random.choice(np.where(y[:, i] == 1)[0], max_count, replace=False)).difference(idx))
    return x[idx, :], y[idx, :]