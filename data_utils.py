import os
import random
import glob
import shutil
import torch
import numpy as np
from skimage import io


def delete_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def prepare_drift_dataset(data_root, experiment_root, in_dist_folders, drift_dist_folders,
                          n_train=150, n_drift=50, seed=1, folders_only=False):
    if not isinstance(n_train, list):
        n_train = [n_train for i in range(len(in_dist_folders))]

    exp_name = []
    for name in in_dist_folders:
        if "(" in name:
            exp_name.append(name)

    if len(exp_name) == 0:
        exp_name = "control"
    else:
        exp_name = "-".join(exp_name)

    # Clear the folders and create the dataset for the experiments
    folders = {}
    folders["experiment"] = os.path.join(experiment_root, exp_name)
    folders["train"] = os.path.join(experiment_root, exp_name, "train")
    folders["val"] = os.path.join(experiment_root, exp_name, "val")
    folders["drift"] = os.path.join(experiment_root, exp_name, "drift")
    folders["result"] = os.path.join(experiment_root, exp_name, "result")

    if folders_only:
        return folders

    if os.path.exists(folders["train"]):
        delete_folder(folders["train"])
        delete_folder(folders["val"])
        delete_folder(folders["drift"])
        delete_folder(folders["result"])
    else:
        os.makedirs(folders["train"], exist_ok=False)
        os.makedirs(folders["val"], exist_ok=False)
        os.makedirs(folders["drift"], exist_ok=False)
        os.makedirs(folders["result"], exist_ok=False)

    # Generate the training and validation datasets
    all_indist_image_ids = []
    for j, folder in enumerate(in_dist_folders):
        n_train_folder = n_train[j]
        if "(" in folder:
            cls = folder.split("(")[0]
            image_path = os.path.join(data_root, cls, folder)
            print(image_path)
            images = glob.glob(os.path.join(image_path, "*.jpg"), recursive=False)
            image_ids = [i.split("/")[-1] for i in images]
            all_indist_image_ids.extend(image_ids)

            os.makedirs(os.path.join(folders["train"], cls), exist_ok=True)
            os.makedirs(os.path.join(folders["val"], cls), exist_ok=True)
            print(f"\t #images for {folder}: {len(images)}")
            print(f"\t{n_train_folder} will be used in training")
            if len(images) < n_train_folder:
                shutil.rmtree(folders["experiment"])
                raise ValueError(f"Not enough images for {folder}")

            for img in images[:n_train_folder]:
                shutil.copy(img, os.path.join(folders["train"], cls))
            for img in images[n_train_folder:n_train_folder + n_drift]:
                shutil.copy(img, os.path.join(folders["val"], cls))

        else:
            cls = folder
            image_path = os.path.join(data_root, cls)
            images = glob.glob(os.path.join(image_path, "**/*.jpg"), recursive=True)

            os.makedirs(os.path.join(folders["train"], cls), exist_ok=True)
            os.makedirs(os.path.join(folders["val"], cls), exist_ok=True)
            random.Random(seed).shuffle(images)
            print(f"\t #images for {cls}: {len(images)}")
            if len(images) < n_train_folder:
                shutil.rmtree(folders["experiment"])
                raise ValueError(f"Not enough images for {folder}")

            for img in images[:n_train_folder]:
                shutil.copy(img, os.path.join(folders["train"], cls))
            for img in images[n_train_folder:n_train_folder + n_drift]:
                shutil.copy(img, os.path.join(folders["val"], cls))

    for cls in [c.split("(")[0] for c in in_dist_folders]:
        os.makedirs(os.path.join(experiment_root, exp_name, "drift", cls), exist_ok=True)

    for folder in drift_dist_folders:
        print(f"Drift", end=" ")
        cls = folder.split("(")[0]
        image_path = os.path.join(data_root, cls, folder)
        drift_images = glob.glob(os.path.join(image_path, "*.jpg"), recursive=False)
        drift_image_ids = set([i.split("/")[-1] for i in drift_images])
        common = drift_image_ids.intersection(set(all_indist_image_ids))
        if len(common) > 0:
            print(f"{common} are removed from the drift dataset.")
            drift_image_ids = list(drift_image_ids.difference(common))
        drift_images = [os.path.join(image_path, im_id) for im_id in drift_image_ids]
        random.Random(seed).shuffle(drift_images)
        delete_folder(os.path.join(folders["drift"], cls))
        print(f"\t #images for {folder}: {len(drift_images)}")
        for img in drift_images[:n_drift]:
            shutil.copy(img, os.path.join(folders["drift"], cls))

    return folders


class DriftDataset():
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform
    
    def __len__(self):
        # Return the length of the dataset
        return len(self.images)
    
    def __getitem__(self, idx):
        # Return the observation based on an index. Ex. dataset[0] will return the first element from the dataset, in this case the image and the label.
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_path = self.images[idx]
        image = io.imread(img_path)
        if self.transform:
            image = self.transform(image)
        return image
    
def get_drift_dataset(experiment_folder, data_root, drift_dist, exclude_concepts, class_name, transforms, take_diff=True,
                     seed=1, n_max_images=50):
    
    drift_dist_args = drift_dist.split("(")
    print(drift_dist_args)
    
    if len(drift_dist_args) > 1:
        drift_folder = os.path.join(data_root, drift_dist_args[0], drift_dist)
        drift_ims = glob.glob(os.path.join(drift_folder, "*.jpg"), recursive=False)
    else:
        drift_folder = os.path.join(data_root, drift_dist_args[0])
        drift_ims = glob.glob(os.path.join(drift_folder, "**/*.jpg"), recursive=True)
    
    train_folder = os.path.join(experiment_folder, "train", class_name)
    train_ims = os.listdir(train_folder)
    common_ims = set([im_path for im_path in drift_ims if (im_path.split("/")[-1] in train_ims)])

    exclude_concept_ims = []
    print(f"Excluding images for : {exclude_concepts}")
    for c in exclude_concepts:
        if c == drift_dist:
            continue
        concept_folder = os.path.join(data_root, c.split("(")[0], c)
        concept_ims = os.listdir(concept_folder)
        exclude_concept_ims.extend(concept_ims)
        print(len(concept_ims))
    
    exclude_ims = set([im_path for im_path in drift_ims if im_path.split("/")[-1] in exclude_concept_ims])
    if take_diff:
        drift_ims = list(set(drift_ims).difference(common_ims))
    
    drift_ims = list(set(drift_ims).difference(exclude_ims))
    np.random.seed(seed)
    np.random.shuffle(drift_ims)
    drift_ims = drift_ims[:n_max_images]
    drift_ims = [path for path in drift_ims if len(io.imread(path).shape) > 2]
    drift_ds = DriftDataset(drift_ims, transforms)
    print(f"Final image count: {len(drift_ims[:n_max_images])}")
    return drift_ds


    
