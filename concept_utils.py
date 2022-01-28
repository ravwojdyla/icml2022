from collections import defaultdict
import numpy as np
import torch
from torch._C import Value
import torch.nn as nn
import torch.optim as optim


class EasyDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


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
            margin_tensor = torch.tensor(np.concatenate(
                val_list, axis=0), requires_grad=False).float().to(device)
            all_margin_info[key] = margin_tensor

        self.concept_info = EasyDict()
        self.concept_info.margin_info = EasyDict(dict(all_margin_info))
        self.concept_info.bank = torch.tensor(np.concatenate(all_vectors, axis=0), requires_grad=False).float().to(
            device)
        self.concept_info.norms = torch.norm(
            self.concept_info.bank, p=2, dim=1, keepdim=True).detach()
        self.concept_info.intercepts = torch.tensor(np.concatenate(all_intercepts, axis=0),
                                                    requires_grad=False).float().to(device)
        self.concept_info.concept_names = concept_names
        print("Concept Bank is initialized.")

    def __getattr__(self, item):
        return self.concept_info[item]


def get_concept_scores_mv_valid(tensor, labels, concept_bank, model_bottom, model_top,
                                alpha=1e-4, beta=1e-4, n_steps=100,
                                lr=1e-1, momentum=0.9, enforce_validity=True,
                                kappa="mean"):
    """
    Full CCE.
    """
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

    criterion = nn.CrossEntropyLoss()
    W = nn.Parameter(torch.zeros(
        1, concepts.shape[0], device=device), requires_grad=True)

    # Normalize the concept vectors
    normalized_C = max_margins * concepts / concept_norms
    # Compute the current distance of the sample to decision boundaries of SVMs
    margins = (torch.matmul(concepts, embedding.T) +
               concept_intercepts) / concept_norms
    # Computing constraints for the concepts scores
    W_clamp_max = (max_margins * concept_norms -
                   concept_intercepts - torch.matmul(concepts, embedding.T))
    W_clamp_min = (min_margins * concept_norms -
                   concept_intercepts - torch.matmul(concepts, embedding.T))

    W_clamp_max = (W_clamp_max / (max_margins * concept_norms)).T
    W_clamp_min = (W_clamp_min / (max_margins * concept_norms)).T

    if enforce_validity:
        if kappa == "mean":
            W_clamp_max[(margins > concept_bank.margin_info.pos_mean).T] = 0.
            W_clamp_min[(margins < concept_bank.margin_info.neg_mean).T] = 0.
        elif kappa == "zero":
            W_clamp_max[(margins > torch.zeros_like(margins)).T] = 0.
            W_clamp_min[(margins < torch.zeros_like(margins)).T] = 0.
        else:
            raise ValueError(kappa)

    zeros = torch.zeros_like(W_clamp_max)
    W_clamp_max = torch.where(
        W_clamp_max < zeros, zeros, W_clamp_max).detach().flatten(1)
    W_clamp_min = torch.where(
        W_clamp_min > zeros, zeros, W_clamp_min).detach().flatten(1)

    optimizer = optim.SGD([W], lr=lr, momentum=momentum)
    history = []

    for i in range(n_steps):
        optimizer.zero_grad()
        new_embedding = embedding + torch.matmul(W, normalized_C)
        new_out = model_top(new_embedding.view(*model_shape))
        l1_loss = torch.norm(W, dim=1, p=1)/W.shape[1]
        l2_loss = torch.norm(W, dim=1, p=2)/W.shape[1]
        ce_loss = criterion(new_out, labels)
        loss = ce_loss + l1_loss * alpha + l2_loss * beta
        history.append(
            f"{ce_loss.detach().item()}, L1:{l1_loss.detach().item()}, L2: {l2_loss.detach().item()}")
        loss.backward()
        optimizer.step()
        if enforce_validity:
            W_projected = torch.where(W < W_clamp_min, W_clamp_min, W).detach()
            W_projected = torch.where(
                W > W_clamp_max, W_clamp_max, W_projected)
            W.data = W_projected.detach()
            W.grad.zero_()

    final_emb = embedding + torch.matmul(W, normalized_C)
    W = W[0].detach().cpu().numpy().tolist()

    concept_scores = dict()
    for i, n in enumerate(concept_names):
        concept_scores[n] = W[i]
    concept_names = sorted(concept_names, key=concept_scores.get, reverse=True)

    new_out = model_top(final_emb.view(*model_shape))
    # Check if the counterfactual can flip the label

    if (new_out.argmax(dim=1) == labels):
        success = True
    else:
        success = False
    opt_result = {"success": success,
                  "concept_scores": concept_scores,
                  "concept_scores_list": concept_names,
                  "W": np.array(W)}
    opt_result = EasyDict(opt_result)
    return opt_result


def batch_concept_scores_mv_valid(tensor, labels, concept_bank, model_bottom, model_top,
                                  alpha=1e-4, beta=1e-4, n_steps=100,
                                  lr=1e-1, momentum=0.9, enforce_validity=True,
                                  kappa="mean"):
    """
    Full Batch CCE.
    """
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

    criterion = nn.CrossEntropyLoss()
    W = nn.Parameter(torch.zeros(
        1, concepts.shape[0], device=device), requires_grad=True)

    # Normalize the concept vectors
    normalized_C = max_margins * concepts / concept_norms
    # Compute the current distance of the sample to decision boundaries of SVMs
    margins = (torch.matmul(concepts, embedding.T) +
               concept_intercepts) / concept_norms
    # Computing constraints for the concepts scores
    W_clamp_max = (max_margins * concept_norms -
                   concept_intercepts - torch.matmul(concepts, embedding.T))
    W_clamp_min = (min_margins * concept_norms -
                   concept_intercepts - torch.matmul(concepts, embedding.T))

    W_clamp_max = (W_clamp_max / (max_margins * concept_norms)).T
    W_clamp_min = (W_clamp_min / (max_margins * concept_norms)).T

    if enforce_validity:
        if kappa == "mean":
            W_clamp_max[(margins > concept_bank.margin_info.pos_mean).T] = 0.
            W_clamp_min[(margins < concept_bank.margin_info.neg_mean).T] = 0.
        elif kappa == "zero":
            W_clamp_max[(margins > torch.zeros_like(margins)).T] = 0.
            W_clamp_min[(margins < torch.zeros_like(margins)).T] = 0.
        else:
            raise ValueError(kappa)

    zeros = torch.zeros_like(W_clamp_max)
    W_clamp_max = torch.where(
        W_clamp_max < zeros, zeros, W_clamp_max).detach().flatten(1)
    W_clamp_min = torch.where(
        W_clamp_min > zeros, zeros, W_clamp_min).detach().flatten(1)

    W_clamp_max = torch.mean(W_clamp_max, dim=0, keepdim=True)
    W_clamp_min = torch.mean(W_clamp_min, dim=0, keepdim=True)

    optimizer = optim.SGD([W], lr=lr, momentum=momentum)
    history = []

    for i in range(n_steps):
        optimizer.zero_grad()
        new_embedding = embedding + torch.matmul(W, normalized_C)
        new_out = model_top(new_embedding.view(*model_shape))
        l1_loss = torch.norm(W, dim=1, p=1)/W.shape[1]
        l2_loss = torch.norm(W, dim=1, p=2)/W.shape[1]
        ce_loss = criterion(new_out, labels)
        loss = ce_loss + l1_loss * alpha + l2_loss * beta
        #history.append(
        #    f"{ce_loss.detach().item()}, L1:{l1_loss.detach().item()}, L2: {l2_loss.detach().item()}")
        loss.backward()
        optimizer.step()
        if enforce_validity:
            W_projected = torch.where(W < W_clamp_min, W_clamp_min, W).detach()
            W_projected = torch.where(
                W > W_clamp_max, W_clamp_max, W_projected)
            W.data = W_projected.detach()
            W.grad.zero_()

    final_emb = embedding + torch.matmul(W, normalized_C)
    W = W[0].detach().cpu().numpy().tolist()
    #print("\n".join(history))
    concept_scores = dict()
    for i, n in enumerate(concept_names):
        concept_scores[n] = W[i]
    concept_names = sorted(concept_names, key=concept_scores.get, reverse=True)

    new_out = model_top(final_emb.view(*model_shape))
    # Check if the counterfactual can flip the label

    success = (new_out.argmax(dim=1) == labels).float().mean().detach().cpu().numpy()
    opt_result = {"success": success,
                  "concept_scores": concept_scores,
                  "concept_scores_list": concept_names,
                  "W": np.array(W)}
    opt_result = EasyDict(opt_result)
    return opt_result


def get_concept_scores(tensor, labels, concept_bank, model_bottom, model_top, multiplier=100):
    """
    Univariate CCE.
    """
    correct_idx = labels.cpu().item()
    device = tensor.device
    concept_scores = {}
    embedding = model_bottom(tensor).detach()
    embedding = torch.flatten(embedding, 1)
    original_preds = model_top(embedding).detach().cpu().numpy().squeeze()
    max_margins = concept_bank.margin_info.max
    concept_norms = concept_bank.norms
    concepts = concept_bank.bank
    concept_names = concept_bank.concept_names.copy()

    normalized_C = max_margins * concepts / concept_norms

    for i, c_name in enumerate(concept_names):
        #coef = concepts[i:i+1]
        to_add = normalized_C[i:i+1].to(device).float()
        plus = embedding + to_add
        plus_preds = model_top(plus)
        plus_diff = plus_preds.squeeze(
        )[correct_idx] - original_preds.squeeze()[correct_idx]
        plus_diff = plus_diff.detach().cpu().numpy()
        concept_scores[c_name] = float(plus_diff)

    concept_names = sorted(
        concept_scores, key=concept_scores.get, reverse=True)

    opt_result = {"success": False,
                  "concept_scores": concept_scores,
                  "concept_scores_list": concept_names,
                  "W": None}
    opt_result = EasyDict(opt_result)
    return opt_result
