import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import os
import json
import shutil
import random
import PIL
from easydict import EasyDict as edict
import sklearn
import sklearn.svm
import sklearn.model_selection
from sklearn.model_selection import train_test_split
import argparse
from tqdm import tqdm
import pickle
from model.classifier import Classifier


class DensenetCXRBottom(nn.Module):
    def __init__(self, model):
        super(DensenetCXRBottom, self).__init__()
        self.backbone = model.module.backbone
        self.attention_map = model.module.attention_map
        self.global_pool = model.module.global_pool
        self.bn_0 = model.module.bn_0

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.attention_map(feat)
        feat = self.global_pool(feat, [])
        return feat


class DensenetCXRTop(nn.Module):
    def __init__(self, model):
        super(DensenetCXRTop, self).__init__()
        self.fc_0 = model.module.fc_0

    def forward(self, x):
        out = self.fc_0(x)
        return out


def get_model_parts(args):
    with open(os.path.join(args.model_path, 'cfg.json')) as f:
        cfg = edict(json.load(f))
    model = Classifier(cfg)
    model = nn.DataParallel(model).to(args.device).eval()
    ckpt_path = os.path.join(args.model_path, 'best1.ckpt')
    ckpt = torch.load(ckpt_path, map_location=args.device)
    model.module.load_state_dict(ckpt['state_dict'])
    print("Model is loaded!!")

    model_bottom, model_top = DensenetCXRBottom(model), DensenetCXRTop(model)
    return model_bottom, model_top


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default="", metavar='MODEL_PATH',
                        type=str, help="Path to the trained models")
    parser.add_argument("--concept-dir", default="/path/dataset/broden_concepts/", type=str)
    parser.add_argument("--out-dir", default="/path/conceptual-explanations/cxr/", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--C", default=0.01, type=float, help="Regularization parameter for SVMs.")
    parser.add_argument("--ds-type", default="nih", type=str)

    return parser.parse_args()


def main():
    args = config()
    input_size = 224
    torch.manual_seed(args.seed)
    data_transforms = {
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    print("Initialized Datasets and Dataloaders")
    model_bottom, _ = get_model_parts(args)
    model_bottom = model_bottom.to(args.device)
    model_bottom.eval()

    transform = data_transforms['val']
    C = args.C
    concept_dict = {}
    # iterate through each folder
    for concept in os.listdir(args.concept_dir):
        print(concept)
        if concept in concept_dict:
            continue

        all_embeddings = []
        all_labels = []
        try:
            image_dataset = datasets.ImageFolder(os.path.join(args.concept_dir, concept), transform)
            dataloaders = torch.utils.data.DataLoader(image_dataset, batch_size=args.batch_size,
                                                      shuffle=True, num_workers=args.num_workers)

        except Exception as e:
            print(e, "folder is not an image folder.")
            continue

        # load all of the images, get the embeddings
        for inputs, labels in tqdm(dataloaders):
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            embeddings = model_bottom(inputs)
            all_embeddings.extend(embeddings.detach().cpu().numpy().reshape(embeddings.shape[0], embeddings.shape[1]))
            all_labels.extend(labels.detach().cpu().numpy())

        # train an svm on the pos, neg
        X_train, X_test, y_train, y_test = train_test_split(all_embeddings, all_labels)
        svm = sklearn.svm.SVC(kernel="linear", C=C, probability=True)
        svm.fit(X_train, y_train)
        train_acc = svm.score(X_train, y_train)
        test_acc = svm.score(X_test, y_test)
        X_train = [v.reshape(1, -1) for v in X_train]
        X_train = np.concatenate(X_train, axis=0)
        train_margin = ((np.dot(svm.coef_, X_train.T) + svm.intercept_) / np.linalg.norm(svm.coef_)).T
        margin_info = {"max": np.max(train_margin),
                       "min": np.min(train_margin),
                       "pos_mean":  np.nanmean(train_margin[train_margin > 0]),
                       "pos_std": np.nanstd(train_margin[train_margin > 0]),
                       "neg_mean": np.nanmean(train_margin[train_margin < 0]),
                       "neg_std": np.nanstd(train_margin[train_margin < 0]),
                       "q_90": np.quantile(train_margin, 0.9),
                       "q_10": np.quantile(train_margin, 0.1)
                       }
        print(train_acc, test_acc)
        # store svm coefs in dictionary
        concept_dict[concept] = (svm.coef_, train_acc, test_acc, svm.intercept_, margin_info)

    print(f"# concepts: {len(concept_dict)}")
    c_to_drop = []
    for c, (_, _, test_acc, _, _) in concept_dict.items():
        if test_acc < 0.7:
            c_to_drop.append(c)
    for c in c_to_drop:
        concept_dict.pop(c)
    print(f"# concepts after removing: {len(concept_dict)}")
    concept_dict_path = os.path.join(args.out_dir, f"concept_densenet_{len(concept_dict)}_{C}_{args.ds_type}.pkl")
    pickle.dump(concept_dict, open(concept_dict_path, 'wb'))


if __name__ == "__main__":
    main()
