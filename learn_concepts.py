import torch
import numpy as np
from torchvision import datasets, transforms
import os

import sklearn
import sklearn.svm
import sklearn.model_selection
from sklearn.model_selection import train_test_split
from model_utils import get_model_parts, initialize_model
import argparse
from tqdm import tqdm
import pickle


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept-dir", default="/path/dataset/broden_concepts/", type=str)
    parser.add_argument("--out-dir", default="/path/concept-banks/metadataset", type=str)
    parser.add_argument("--model-name", default="resnet", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--C", default=0.001, type=float, help="Regularization parameter for SVMs.")
    parser.add_argument("--num-samples", default=100, type=int, 
                        help="Number of positive/negative samples used to learn concepts.")
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
    model_ft, _ = initialize_model(use_pretrained=True, model_name=args.model_name)
    model_bottom, _ = get_model_parts(model_ft, model_name=args.model_name)
    model_bottom = model_bottom.to(args.device)
    model_bottom.eval()

    transform = data_transforms['val']
    concept_dict = {}
    # iterate through each folder
    for concept in os.listdir(args.concept_dir):
        print("Learning: ", concept)
        if concept in concept_dict:
            continue

        all_embeddings = []
        all_labels = []
        try:
            ## Sampling equally from positives and negatives.
            image_dataset = datasets.ImageFolder(os.path.join(args.concept_dir, concept), transform)
            samples = {k: [] for k in image_dataset.class_to_idx.keys()}
            idx_to_class = {v: k for k,v in image_dataset.class_to_idx.items()}
            for j, (sample, cls) in enumerate(image_dataset.samples):
                samples[idx_to_class[cls]].append(j)
            indices = []
            for cls, cls_idx in samples.items():
                indices.extend(np.random.choice(cls_idx, args.num_samples, replace=True))
            ds_subset = torch.utils.data.Subset(image_dataset, indices)
            dataloaders = torch.utils.data.DataLoader(ds_subset, batch_size=args.batch_size,
                                                      num_workers=args.num_workers,
                                                      shuffle=False)
        
        except Exception as e:
            print("Error: ", e)
            continue
        
        # load all of the images, get the embeddings
        for inputs, labels in tqdm(dataloaders):
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            embeddings = model_bottom(inputs)
            all_embeddings.extend(embeddings.detach().cpu().numpy().reshape(embeddings.shape[0], embeddings.shape[1]))
            all_labels.extend(labels.detach().cpu().numpy())

        # train an svm on the pos, neg
        X_train, X_test, y_train, y_test = train_test_split(all_embeddings, all_labels)
        svm = sklearn.svm.SVC(kernel="linear", C=args.C, probability=True)
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
    concept_dict_path = os.path.join(args.out_dir, f"concept_{args.model_name}_{len(concept_dict)}_{args.C}_{args.num_samples}.pkl")
    pickle.dump(concept_dict, open(concept_dict_path, 'wb'))


if __name__ == "__main__":
    main()
