import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import models


class ResNetBottom(nn.Module):
    def __init__(self, original_model):
        super(ResNetBottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x


class ResNetTop(nn.Module):
    def __init__(self, original_model):
        super(ResNetTop, self).__init__()
        self.features = nn.Sequential(*[list(original_model.children())[-1]])

    def forward(self, x):
        x = self.features(x)
        x = nn.Softmax(dim=-1)(x)
        return x


class SqueezenetBottom(nn.Module):
    def __init__(self, original_model):
        super(SqueezenetBottom, self).__init__()
        self.features = nn.Sequential(*list(list(original_model.children())[0].children())[:15], nn.Flatten())

    def forward(self, x):
        x = self.features(x)
        return x


class SqueezenetTop(nn.Module):
    def __init__(self, original_model):
        super(SqueezenetTop, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[1])

    def forward(self, x):
        x = x.view((-1, 512, 13, 13))
        x = self.features(x)
        x = x.view((-1, x.shape[1]))
        x = nn.Softmax(dim=-1)(x)
        return x


class VGGBottom(nn.Module):
    def __init__(self, original_model):
        super(VGGBottom, self).__init__()
        self.features = nn.Sequential(*list(list(original_model.children())[0].children())[:15], nn.Flatten())

    def forward(self, x):
        x = self.features(x)
        return x


class VGGTop(nn.Module):
    def __init__(self, original_model):
        super(VGGTop, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[1])

    def forward(self, x):
        x = x.view((-1, 512, 13, 13))
        x = self.features(x)
        x = x.view((-1, 5))
        x = nn.Softmax(dim=-1)(x)
        return x


class DensenetBottom(nn.Module):
    def __init__(self, original_model):
        super(DensenetBottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return x


class DensenetTop(nn.Module):
    def __init__(self, original_model):
        super(DensenetTop, self).__init__()
        self.features = nn.Sequential(*[list(original_model.children())[-1]])

    def forward(self, x):
        x = self.features(x)
        x = nn.Softmax(dim=-1)(x)
        return x

class AlexnetBottom(nn.Module):
    def __init__(self, original_model):
        super(AlexnetBottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class AlexnetTop(nn.Module):
    def __init__(self, original_model):
        super(AlexnetTop, self).__init__()
        self.features = nn.Sequential(*[list(original_model.children())[-1]])

    def forward(self, x):
        x = self.features(x)
        x = nn.Softmax(dim=-1)(x)
        return x


def get_model_parts(model, model_name):    
    if model_name == "squeezenet":
        model_bottom = SqueezenetBottom(model)
        model_top = SqueezenetTop(model)
    elif model_name == "resnet":
        model_bottom = ResNetBottom(model)
        model_top = ResNetTop(model)
    elif model_name == "densenet":
        model_bottom = DensenetBottom(model)
        model_top = DensenetTop(model)
    elif model_name == "alexnet":
        model_bottom = AlexnetBottom(model)
        model_top = AlexnetTop(model)
    else:
        raise ValueError(f"{model_name} does not exist.")
    return model_bottom, model_top


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
            

def initialize_model(num_classes=5, feature_extract=True, use_pretrained=True, model_name="squeezenet"):
    if model_name == "squeezenet":
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
    elif model_name == "resnet":
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.fc = nn.Linear(512, num_classes)
    elif model_name == "densenet":
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier = nn.Linear(1024, num_classes)
    elif model_name == "alexnet":
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[-1] = nn.Linear(4096, num_classes)
    else:
        raise ValueError(model_name)

    model_ft.num_classes = num_classes
    input_size = 224
    return model_ft, input_size
