import copy
import time
import torch
from torchvision import datasets


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False, device="cpu"):
    since = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['val', 'train']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} phase | Loss: {:.4f}, Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

    
def test_drift(model_ft, criterion, data_transforms, out_dists, folders, batch_size=16, device="cpu"):  
    accs = []
    drift_dataset = datasets.ImageFolder(folders["drift"], data_transforms['val'])
    dataloader_drift = torch.utils.data.DataLoader(drift_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # evaluate drift perfomrance
    model_ft.eval()   # Set model to evaluate mode
    avgmeters = {k: AverageMeter() for k in out_dists}

    # Iterate over data.
    for inputs, labels in dataloader_drift:
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(False):
            outputs = model_ft(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
        # statistics
        acc_bool = (preds==labels)
        for k in avgmeters.keys():
            cls = k.split("(")[0]
            mask = labels == drift_dataset.class_to_idx[cls]
            n_k = mask.detach().cpu().sum()
            if n_k < 1:
                continue
            avgmeters[k].update(acc_bool[mask].float().detach().cpu().mean(), n_k)
    
    accs = {k: avgmeters[k].avg.numpy() for k in avgmeters.keys()}
    print(accs)
    return accs

