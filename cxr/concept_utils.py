import torch.nn as nn
from tqdm import tqdm
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
from collections import defaultdict


class ConceptBank:
    def __init__(self, concept_dict, device):
        all_vectors, concept_names, all_intercepts = [], [], []
        all_margin_info = defaultdict(list)
        for k, (tensor, _, _, intercept, margin_info) in concept_dict.items():
            all_vectors.append(tensor)
            concept_names.append(k)
            all_intercepts.append(intercept.reshape(1, 1))
            for key, value in margin_info.items():
                if key != "train_margins":
                    all_margin_info[key].append(value.reshape(1, 1))
        for key, val_list in all_margin_info.items():
            margin_tensor = torch.tensor(np.concatenate(val_list, axis=0), requires_grad=False).float().to(device)
            all_margin_info[key] = margin_tensor

        self.concept_info = EasyDict()
        self.concept_info.margin_info = EasyDict(dict(all_margin_info))
        self.concept_info.bank = torch.tensor(np.concatenate(all_vectors, axis=0), requires_grad=False).float().to(
            device)
        self.concept_info.norms = torch.norm(self.concept_info.bank, p=2, dim=1, keepdim=True).detach()
        self.concept_info.intercepts = torch.tensor(np.concatenate(all_intercepts, axis=0),
                                                    requires_grad=False).float().to(device)
        self.concept_info.concept_names = concept_names
        print("Concept Bank is initialized.")

    def __getattr__(self, item):
        return self.concept_info[item]


class EasyDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


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


def get_embedding(loader, model, n_samples=200, device="cpu"):
    activations = None
    for image, path in tqdm(loader):
        image = image.to(device)
        embedding = model(image).squeeze().detach()
        if activations is None:
            activations = embedding
        else:
            activations = torch.cat([activations, embedding], dim=0)
        if activations.shape[0] >= n_samples:
            return activations[:n_samples]
    return activations


def learn_concept(activations, c_labels, args, C=0.001):
    # Learn concept vectors
    X_train, X_test, y_train, y_test = train_test_split(activations, c_labels, test_size=args.n_samples,
                                                        random_state=args.seed)
    svm = SVC(kernel="linear", C=C, probability=False)
    svm.fit(X_train, y_train)
    train_acc = svm.score(X_train, y_train)
    test_acc = svm.score(X_test, y_test)
    print(f"Accuracy - Training: {train_acc}, Test: {test_acc}")

    train_margin = ((np.dot(svm.coef_, X_train.T) + svm.intercept_) / np.linalg.norm(svm.coef_)).T

    margin_info = {"max": np.max(train_margin),
                   "min": np.min(train_margin),
                   "pos_mean": np.mean(train_margin[train_margin > 0]),
                   "pos_std": np.std(train_margin[train_margin > 0]),
                   "neg_mean": np.mean(train_margin[train_margin < 0]),
                   "neg_std": np.std(train_margin[train_margin < 0]),
                   "q_90": np.quantile(train_margin, 0.9),
                   "q_10": np.quantile(train_margin, 0.1)
                   }
    # print test accuracy
    print(train_acc, test_acc)
    return svm.coef_, train_acc, test_acc, svm.intercept_, margin_info


def get_concept_scores_mv_valid(tensor, labels, concept_bank, model_bottom, model_top,
                                alpha=1e-4, beta=1e-4, n_steps=100,
                                lr=1e-1, momentum=0.9, enforce_validity=True):
    max_margins = concept_bank.margin_info.max
    min_margins = concept_bank.margin_info.min
    concept_norms = concept_bank.norms
    concept_intercepts = concept_bank.intercepts
    concepts = concept_bank.bank
    concept_names = concept_bank.concept_names.copy()
    device = tensor.device
    embedding = model_bottom(tensor)
    model_shape = embedding.shape
    embedding = embedding.detach().flatten(1)
    
    criterion = nn.BCEWithLogitsLoss()
    W = nn.Parameter(torch.zeros(1, concepts.shape[0], device=device), requires_grad=True)

    # Normalize the concept vectors
    normalized_C = max_margins * concepts / concept_norms
    # print(concepts.shape, embedding.shape)
    # Compute the current distance of the sample to decision boundaries of SVMs
    margins = (torch.matmul(concepts, embedding.T) + concept_intercepts) / concept_norms
    # Computing constraints for the concepts scores
    W_clamp_max = (max_margins * concept_norms - concept_intercepts - torch.matmul(concepts, embedding.T))
    W_clamp_min = (min_margins * concept_norms - concept_intercepts - torch.matmul(concepts, embedding.T))

    W_clamp_max = (W_clamp_max / (max_margins * concept_norms)).T
    W_clamp_min = (W_clamp_min / (max_margins * concept_norms)).T
    
    if enforce_validity:
        W_clamp_max[(margins > torch.zeros_like(margins)).T] = 0.
        W_clamp_min[(margins < torch.zeros_like(margins)).T] = 0.
    zeros = torch.zeros_like(W_clamp_max)
    W_clamp_max = torch.where(W_clamp_max < zeros, zeros, W_clamp_max).detach().flatten(1)
    W_clamp_min = torch.where(W_clamp_min > zeros, zeros, W_clamp_min).detach().flatten(1)

    optimizer = optim.SGD([W], lr=lr, momentum=momentum)
    history = []
    for i in range(n_steps):
        optimizer.zero_grad()
        new_embedding = embedding + torch.matmul(W, normalized_C)
        new_out = model_top(new_embedding.view(*model_shape))
        l1_loss = torch.norm(W, dim=1, p=1)/W.shape[1]
        l2_loss = torch.norm(W, dim=1, p=2)/W.shape[1]
        ce_loss = criterion(new_out.view(1, 1), labels.view(1, 1))
        loss = ce_loss + l1_loss * alpha + l2_loss * beta
        history.append(f"{ce_loss.detach().item()}, L1:{l1_loss.detach().item()}, L2: {l2_loss.detach().item()}")
        loss.backward()
        optimizer.step()
        if enforce_validity:
            W_projected = torch.where(W < W_clamp_min, W_clamp_min, W).detach()
            W_projected = torch.where(W > W_clamp_max, W_clamp_max, W_projected)
            W.data = W_projected.detach()
            W.grad.zero_()

    final_emb = embedding + torch.matmul(W, normalized_C)
    W = W[0].detach().cpu().numpy().tolist()

    concept_scores = dict()
    for i, n in enumerate(concept_names):
        concept_scores[n] = W[i]
    concept_names = sorted(concept_names, key=concept_scores.get, reverse=True)

    new_out, orig_out = model_top(final_emb.view(*model_shape)), model_top(embedding.view(*model_shape))
    # Check if the counterfactual can flip the label
    
    new_pred = (torch.sigmoid(new_out.view(-1)) > 0.5).int()
    if (new_pred == labels):
        success = True
    else:
        success = False
    opt_result = {"success": success,
                  "concept_scores": concept_scores,
                  "concept_scores_list": concept_names,
                  "W": np.array(W)}
    opt_result = EasyDict(opt_result)

    return opt_result
