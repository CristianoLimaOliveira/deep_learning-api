from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision
from pathlib import Path
from PIL import Image
from torch import nn
import pandas as pd
import numpy as np
import torch
import os
import cv2
from random import shuffle
from sklearn.metrics import classification_report
import base64


def prediction_normalize(image):
    normalize = T.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    transform = T.Compose([
        T.ToTensor(),
        T.Resize([256, 256]),
        T.CenterCrop(224),
        normalize])

    return transform(image)


def from_base64_to_image(image_str):
    img_bytes = base64.b64decode(image_str)
    img_np = np.frombuffer(img_bytes, dtype=np.uint8)
    image = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    return image

def get_categories():
    return ['not-hotdog', 'hotdog']

class HotDogDataset(Dataset):
    def __init__(self, image_paths, transform = None):
        super().__init__()
        self.paths = image_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        y = 0 if 'not-hotdog' in self.paths[index] else 1
        x = cv2.imread(self.paths[index])
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)

        if self.transform:
            x = self.transform(x)

        return x, y

def get_train_and_test_dataloader(batch_size = 64):
    image_paths_train = ['../hotdog/train/hotdog/' + path for path in os.listdir('../hotdog/train/hotdog')]
    image_paths_train.extend(['../hotdog/train/not-hotdog/' + path for path in os.listdir('../hotdog/train/not-hotdog')])
    shuffle(image_paths_train)

    image_paths_test = ['../hotdog/test/hotdog/' + path for path in os.listdir('../hotdog/test/hotdog')]
    image_paths_test.extend(['../hotdog/test/not-hotdog/' + path for path in os.listdir('../hotdog/test/not-hotdog')])
    shuffle(image_paths_test)

    normalize = T.Normalize(
        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    train_augs = T.Compose([
        T.ToTensor(),
        T.RandomHorizontalFlip(),
        T.RandomResizedCrop(224),
        normalize])

    test_augs = T.Compose([
        T.ToTensor(),
        T.Resize([256, 256]),
        T.CenterCrop(224),
        normalize])

    train_dataset_tensors = HotDogDataset(image_paths_train, train_augs)
    test_dataset_tensors = HotDogDataset(image_paths_test, test_augs)
    
    train_loader = DataLoader(train_dataset_tensors, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset_tensors, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def save_model(model, path, epoch, optimizer, loss_history, file_name):
    """Salva o modelo quando ocorre melhoria no conjunto de validação."""

    state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()

    torch.save({
        'epoch': epoch,
        'model_state_dict': state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss_history
    }, path / Path(file_name))

def modelAllocation(model, parallel, device_ids):
    if isinstance(device_ids, list):
        if (torch.cuda.device_count() > 1) and parallel:
            print('Running training on', torch.cuda.device_count(), 'GPU(s)')
            model = nn.DataParallel(model, device_ids=device_ids)

            return model.to(device_ids[0])
        else:
            return model.to(device_ids[0])
    else:
        raise Exception('Invalid type for: {}'.format(device_ids), ' List required.')

def load_model(model, path, device_ids):
    load = torch.load(path, map_location=torch.device('cpu'))
    epoch = load['epoch']
    model.load_state_dict(load['model_state_dict'])
    best_loss, train_loss, val_loss = load['loss']

    model = modelAllocation(model, True, device_ids)
    optimizer = torch.optim.SGD(model.parameters())
    optimizer.load_state_dict(load['optimizer_state_dict'])
    early_stopping = EarlyStopping(best_loss)

    training_summary = """
            TRAINING SUMMARY:
            Best loss: {}\n
            Last epoch: {}\n
            Learning Rate: {}\n
            Patience: {}\n
            """.format(best_loss, epoch, optimizer.param_groups[0]['lr'], early_stopping.patience)
    print(training_summary)

    return epoch, model, train_loss, val_loss, optimizer, early_stopping

class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-5, best_loss=None):
        """
        Parâmetros:
        - patience: Número de épocas sem melhoria antes de parar.
        - min_delta: Mínima mudança no valor monitorado para considerar como melhoria.
        - path: Caminho para salvar o melhor modelo.
        - best_loss: melhor resultado de redução da perda.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = best_loss
        self.early_stop = False

    def __call__(self, save, model, loss, epoch, optimizer):
        if self.best_loss is None:
            self.best_loss = loss[0]

        elif self.best_loss - loss[0] > self.min_delta:
            self.best_loss = loss[0]
            self.counter = 0

            save_model(model, save, epoch, optimizer, loss, 'bestmodel.pt')
        elif self.best_loss - loss[0] < self.min_delta:
            self.counter += 1
            print(f'Warning: Stop counter {self.counter} of {self.patience}')
            if self.counter >= self.patience:
                print('Warning: Early stopping')
                self.early_stop = True

def train(model, optimizer, train_loader, criterion, device):
    # Treinamento
    model.train()
    train_loss = 0.0
    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    return model, train_loss, optimizer

def validation(model, test_loader, criterion, device):
    # Validação
    model.eval()
    test_loss = 0.0
    with torch.inference_mode():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

    return test_loss

def test(model, dataloader, classes, device):
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    result = {
        'label': [],
        'prediction': []
    }
    # again no gradients needed
    with torch.no_grad():
        for images, labels in dataloader:
            labels = torch.tensor(labels)
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            result['label'].extend(labels.cpu())
            result['prediction'].extend(predictions.cpu())
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # print accuracy for each class
    total_correct = 0.0
    total_prediction = 0.0
    for classname, correct_count in correct_pred.items():
        total_correct += correct_count
        total_prediction += total_pred[classname]
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                        accuracy))
    print("Global acccuracy is {:.1f}".format(100 * total_correct/total_prediction))
    print(classification_report(result['label'], result['prediction'], target_names=get_categories()))

def start_training(device, num_epochs, model, train_loader, test_loader, optimizer, criterion, early_stopping, path_early_stopping='./'):
    train_losses = []
    test_losses = []
    
    for epoch in range(num_epochs):
        model, train_loss, optimizer = train(model, optimizer, train_loader, criterion, device)
        test_loss = validation(model, test_loader, criterion, device)

        # Média das perdas
        train_loss /= len(train_loader)
        test_loss /= len(test_loader)
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {test_loss:.4f}")

        # Early Stopping
        early_stopping(path_early_stopping, model, [test_loss, train_losses, test_losses], epoch, optimizer)

        if early_stopping.early_stop:
            print("Early stopping")
            losses = {
                "Train loss": train_losses,
                "Test loss": test_losses
            }
            break
        
        save_model(model=model, path=path_early_stopping, epoch=epoch, optimizer=optimizer, loss_history=[test_loss, train_losses, test_losses], file_name='backup.pt')
    
    return train_losses, test_losses
