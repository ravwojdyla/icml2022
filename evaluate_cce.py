import torch
import torch
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
import os
import pickle
from tqdm import tqdm
from data_utils import get_drift_dataset
from model_utils import get_model_parts
import argparse
from concept_utils import get_concept_scores_mv_valid, ConceptBank
from concept_utils import get_concept_scores, EasyDict


data_root = "/path/dataset/metadataset/MetaDataset/subsets"
mean_pxs = np.array([0.485, 0.456, 0.406])
std_pxs = np.array([0.229, 0.224, 0.225])
experiments = ['dog(chair)',
                'cat(cabinet)',
               'dog(snow)',
               'dog(car)',
               'dog(horse)',
               'bird(water)',
               'dog(water)',
               'dog(fence)',
               'elephant(building)',
               'cat(keyboard)',
               'dog(sand)',
               'cat(computer)',
               'dog(bed)',
               'cat(bed)',
               'cat(book)',
               'dog(grass)',
               'cat(mirror)',
               'bird(sand)',
               'bear(chair)',
               'cat(grass)']


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="resnet", type=str)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--seed", default=4, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument(
        "--bank-path", default="/path/conceptual-explanations/banks/concept_resnet_170.pkl", type=str)
    parser.add_argument("--input-size", default=224, type=int)

    return parser.parse_args()


args = config()


data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(args.input_size),
    transforms.CenterCrop(args.input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean_pxs, std_pxs)
])

# Get the experiment folder and the control model
experiment_root = f'/path/outputs/conceptualexplanations/metadataset/{args.model_name}_mv_scale_50'
control_folder = os.path.join(experiment_root, f"control")

# Initialize the concept bank
all_concepts = pickle.load(open(args.bank_path, 'rb'))
concept_bank = ConceptBank(all_concepts, args.device)
all_concept_names = concept_bank.concept_names.copy()

eval_modes = ["mistakes"]

all_out = []
# Control model
control_folder = os.path.join(experiment_root, "control")
model_control = torch.load(
    open(os.path.join(control_folder, "result", "confounded-model.pt"), "rb"))
model_bottom_control, model_top_control = get_model_parts(
    model_control, args.model_name)
model_bottom_control.eval()
model_top_control.eval()
print("Control model is loaded!")

for experiment in experiments:
    print(f"Starting experiment: {experiment}")
    if len(experiment.split("-")) > 1:
        continue
    experiment_folder = os.path.join(experiment_root, experiment)
    try:
        with open(os.path.join(experiment_folder, "result", "concept_config.pkl"), "rb") as f:
            concept_config = pickle.load(f)
    except Exception as e:
        print(e, experiment)
        continue

    true_concept = None

    for cl in concept_config["in_distributions"]:
        if "(" in cl:
            class_name = cl.split("(")[0]
            true_concept = cl.split("(")[-1][:-1]
            if true_concept == 'branch':
                true_concept = 'tree'

    if true_concept is None:
        print(f"No concept in {experiment} exp.")
        continue

    if true_concept not in concept_bank.concept_names:
        print(f"{true_concept} not in bank!")
        continue

    dists_to_remove = [
        c for c in concept_config["in_distributions"] if "(" in c]
    print(f"Animal: {class_name}, True Concept: {true_concept}")

    # Get models
    model_ft = torch.load(
        open(os.path.join(experiment_folder, "result", "confounded-model.pt"), "rb"))
    # Model to test
    model_bottom, model_top = get_model_parts(model_ft, args.model_name)
    model_bottom.eval()
    model_top.eval()

    train_dataset = datasets.ImageFolder(os.path.join(
        experiment_folder, "train"), data_transforms)
    
    class_idx = train_dataset.class_to_idx[class_name]
    labels_orig = torch.tensor(class_idx).long().view(1).to(args.device)
    drift_ds = get_drift_dataset(experiment_folder, data_root, class_name, dists_to_remove,
                                 class_name, data_transforms, take_diff=True, seed=3,
                                 n_max_images=200)

    dataloaders_drift = torch.utils.data.DataLoader(
        drift_ds, batch_size=1, shuffle=False, num_workers=args.num_workers)
    
    methods = ["random", "control", "cce", "univariate"]
    score_dict = {method: {name: [] for name in concept_bank.concept_names} 
                    for method in methods} 
    success_arr = {method: [] for method in methods}

    for eval_mode in eval_modes:
        tqdm_iterator = tqdm(dataloaders_drift)
        for inputs in tqdm_iterator:
            inputs = inputs.to(args.device)
            labels = labels_orig.repeat(inputs.shape[0])
            label_mask = (labels == class_idx)
            if label_mask.float().sum() < 1:
                continue
            if eval_mode == "mistakes":
                preds = model_top(model_bottom(inputs)).argmax(dim=1)
                label_mask = (label_mask & (
                    preds.squeeze() != labels.squeeze()))
                if label_mask.float().sum() < 1:
                    continue
            results, scores = {}, {}
            
            np.random.shuffle(all_concept_names)
            results["random"] = EasyDict({
                "concept_scores_list": all_concept_names,
                "success": np.zeros(len(all_concept_names))
            })
            results["univariate"] = get_concept_scores(inputs, labels, concept_bank,
                                                        model_bottom, model_top)
            results["cce"] = get_concept_scores_mv_valid(inputs, labels,
                                                     concept_bank,
                                                     model_bottom, model_top,
                                                     alpha=1e-2, beta=1e-1, lr=1e-2,
                                                     enforce_validity=True, momentum=0.9)

            results["control"] = get_concept_scores_mv_valid(inputs, labels,
                                                             concept_bank,
                                                             model_bottom_control, model_top_control,
                                                             alpha=1e-2, beta=1e-1, lr=1e-2,
                                                             enforce_validity=True, momentum=0.9)
            
            for method in methods:
                for i, name in enumerate(results[method].concept_scores_list):
                    score_dict[method][name].append(i)
            
                scores[method] = np.array(score_dict[method][true_concept])
                success_arr[method].append(results[method].success)
            
            desc = [f"{m}: Mean: {np.mean(scores[m]):.2f}, Top5:{(scores[m] < 5).mean():.2f}" for m in methods]
            desc = " | ".join(desc)
            tqdm_iterator.set_description(desc)
            if scores["cce"].shape[0] >= 50:
                break

        print(f"Top5: {(np.array(score_dict['cce'][true_concept])<5).mean()}")
        mean_score_dict = {m: {} for m in methods}
        for m in methods:
            for k in score_dict[m].keys():
                mean_score_dict[m][k] = np.mean(score_dict[m][k])

        ordered = {m: sorted(concept_bank.concept_names, key=mean_score_dict[m].get) for m in methods}
        score_arrs = {m : np.array(score_dict[m][true_concept]) for m in methods}
        out_dict = {
            "experiment": experiment,
            "target": true_concept,
            "eval_mode": eval_mode,
            "sample_size": score_arrs["cce"].shape[0],
            "CCE-top10_concepts": ordered["cce"][:10],
            
        }
        for m in methods:
            method_update = {
            f"{m.upper()}-overall_rank": ordered[m].index(true_concept)+1,
            f"{m.upper()}-Top5": (score_arrs[m] < 5).mean(),
            f"{m.upper()}-Top3": (score_arrs[m] < 3).mean(),
            f"{m.upper()}-success": np.mean(success_arr[m]),
            f"{m.upper()}-Q90": np.quantile(score_arrs[m], q=0.9),
            f"{m.upper()}-Q75": np.quantile(score_arrs[m], q=0.75),
            f"{m.upper()}-Q25": np.quantile(score_arrs[m], q=0.25),
            f"{m.upper()}-Q10": np.quantile(score_arrs[m], q=0.10),
            f"{m.upper()}-Median": np.quantile(score_arrs[m], q=0.5)
            }
            out_dict.update(method_update)
        all_out.append(out_dict)
        print(all_out[-1])
        df_res = pd.DataFrame.from_records(all_out)
        df_res.to_csv(os.path.join(experiment_root, "table1_metadata.csv"))
