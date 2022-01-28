from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models


def get_embedding(loader, model, n_samples=100, device="cpu"):
    activations = None
    fps = []
    for batch in tqdm(loader):
        image = batch["image"].to(device)
        fp = batch["fitzpatrick"]
        fps.append(fp)
        embedding = model(image).squeeze().detach()
        if activations is None:
            activations = embedding
        else:
            activations = torch.cat([activations, embedding], dim=0)
        if activations.shape[0] >= n_samples:
            return activations[:n_samples], fps
    return activations, fps


def get_trained_model(model_type, model_path):
    model_ckpt = torch.load(model_path, map_location="cpu")
    keys = list(model_ckpt.keys())

    # If the model is saved with DataParallel
    for key in keys:
        model_ckpt[key.replace("module.", "")] = model_ckpt.pop(key)

    n_classes = 114
    if model_type == "vgg":
        model_ft = models.vgg16(pretrained=True)
        for param in model_ft.parameters():
            param.requires_grad = False

        model_ft.classifier[6] = nn.Sequential(
            nn.Linear(4096, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, n_classes),
            nn.LogSoftmax(dim=1))

    elif model_type == "resnet":
        model_ft = models.resnet18(pretrained=True)
        for param in model_ft.parameters():
            param.requires_grad = False
        model_ft.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, n_classes),
            nn.LogSoftmax(dim=1))

    elif model_type == "squeezenet":
        model_ft = models.squeezenet1_0(pretrained=True)
        for param in model_ft.parameters():
            param.requires_grad = False

        model_ft.classifier[1] = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Conv2d(512, len(label_codes), kernel_size=1))

    else:
        raise ValueError(f"{model_type} does not exist for this analysis.")
    model_ft.load_state_dict(model_ckpt)
    return model_ft


def get_model_parts(model, model_name):
    if model_name == "squeezenet":
        model_bottom = SqueezenetBottom(model)
        model_top = SqueezenetTop(model)
    elif model_name == "resnet":
        model_bottom = ResNetBottom(model)
        model_top = ResNetTop(model)
    else:
        raise ValueError(f"{model_name} does not exist.")
    return model_bottom, model_top


class ResNetBottom(nn.Module):
    def __init__(self, original_model):
        super(ResNetBottom, self).__init__()
        self.features = nn.Sequential(*(list(original_model.children())[:-1]), nn.Flatten(1, 3))

    def forward(self, x):
        x = self.features(x)
        return x


class ResNetTop(nn.Module):
    def __init__(self, original_model):
        super(ResNetTop, self).__init__()
        self.features = nn.Sequential(*[(list(original_model.children())[-1])])

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.features(x)
        x = nn.Softmax(dim=-1)(x)
        return x
