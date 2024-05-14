import argparse
import clip
import torch
from torch.functional import F
from torchmetrics import Accuracy
import numpy as np
import os
import PIL
import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
import yaml, json
import imp_samp
import datetime
import clip

def create_eval_dir(run_params):
    eval_dir = os.path.join(run_params['save_dir'],run_params['backbone_name'], run_params['timestamp'])
    while os.path.exists(eval_dir):
        eval_dir += "_1"
    os.makedirs(eval_dir, exist_ok=True)
    return eval_dir

### load language
def get_sentence_descriptions(idx_to_classname, idx_to_descriptions,max_len):
    descriptions = {}
    if type(list(idx_to_descriptions.values())[0]) == dict:
        def build_descriptions(classname,category,description):
            return f"{classname} whose {category} is {description}"
        for idx in range(len(idx_to_classname)):
            descriptions[idx] = [build_descriptions(idx_to_classname[str(idx)], category, description) for category, description in idx_to_descriptions[str(idx)].items()]
        return descriptions
    if type(list(idx_to_descriptions.values())[0]) == str:
        def build_descriptions(classname,description):
            return f"{classname} {description}"
        def build_max_len_descriptions(classname,descriptions):
            def get_new_canidate(classname,description):
                return f"{classname} {description}"
            acc = []
            new_candidate = get_new_canidate(classname,descriptions[0])
            for i in range(len(descriptions)):
                if i != 0:
                    old_candidate = new_candidate
                    new_candidate = old_candidate +" "+descriptions[i]
                    try:
                        clip.tokenize(new_candidate)
                    except:
                        acc.append(old_candidate)
                        break
                    if i == len(descriptions)-1:
                        acc.append(new_candidate)
            return acc

        if not max_len:
            for idx in range(len(idx_to_classname)):
                descriptions[idx] = [build_descriptions(idx_to_classname[str(idx)],description[2:]) for description in idx_to_descriptions[str(idx)].split("\n")]
        else:
            for idx in range(len(idx_to_classname)):
                descriptions[idx] = build_max_len_descriptions(idx_to_classname[str(idx)],idx_to_descriptions[str(idx)].split("\n"))
        return descriptions

def get_raw_descriptions(idx_to_classname, idx_to_descriptions,max_len):
    descriptions = {}
    if type(list(idx_to_descriptions.values())[0]) == dict:
        for idx in range(len(idx_to_classname)):
            descriptions[idx] = [description for category, description in idx_to_descriptions[str(idx)].items()]
    if type(list(idx_to_descriptions.values())[0]) == str:
        def build_descriptions(description):
            return f"{description}"
        def build_max_len_descriptions(descriptions):
            def get_new_canidate(description):
                return f"{description}"
            acc = []
            new_candidate = get_new_canidate(descriptions[0])
            for i in range(len(descriptions)):
                if i != 0:
                    old_candidate = new_candidate
                    new_candidate = old_candidate +" "+descriptions[i]
                    try:
                        clip.tokenize(new_candidate)
                    except:
                        acc.append(old_candidate)
                        break
                    if i == len(descriptions)-1:
                        acc.append(new_candidate)
            return acc

        if not max_len:
            for idx in range(len(idx_to_classname)):
                descriptions[idx] = [build_descriptions(idx_to_classname[str(idx)],description[2:]) for description in idx_to_descriptions[str(idx)].split("\n")]
        else:
            for idx in range(len(idx_to_classname)):
                descriptions[idx] = build_max_len_descriptions(idx_to_descriptions[str(idx)].split("\n"))
    return descriptions

### sample visual###
def sample_important_crops_per_image(class_idx_to_sampled_img_paths, n_crops,imp_samp_params):
    class_idx_to_sampled_img_paths_to_sampled_crops = {}
    for idx, path_list in class_idx_to_sampled_img_paths.items():
        class_idx_to_sampled_img_paths_to_sampled_crops[idx] = {path: [] for path in path_list}
        for image_path in path_list:
            crop_list=[]
            patcher = imp_samp.Patcher(image_path=image_path, **imp_samp_params)
            for i in range(n_crops):
                crop = next(patcher)
                crop_list.append(crop)
            class_idx_to_sampled_img_paths_to_sampled_crops[idx][image_path] = crop_list
    return class_idx_to_sampled_img_paths_to_sampled_crops

def sample_images(run_params):
    path_to_ds = run_params['ds_dir']
    elems_in_dir = os.listdir(path_to_ds)
    elems_in_dir = [d for d in elems_in_dir if os.path.isdir(os.path.join(path_to_ds, d)) and not d.startswith("6")]
    elems_in_dir = sorted(elems_in_dir) #sort to ensure reproducibility and deterministic sampling
    class_idx_to_img_paths = {}
    for dir in elems_in_dir:
        class_idx_to_img_paths[dir] = [os.path.join(path_to_ds,dir,elem) for elem in os.listdir(os.path.join(path_to_ds, dir)) if elem.endswith(".png")]
    
    class_idx_to_sampled_img_paths = {}
    for key in class_idx_to_img_paths.keys():
        class_idx_to_sampled_img_paths[key] = random.sample(class_idx_to_img_paths[key], run_params['n_image_samples_per_class'])
    
    return class_idx_to_sampled_img_paths

def get_non_random_images(run_params):
    path_to_ds = run_params['ds_dir']
    elems_in_dir = os.listdir(path_to_ds)
    elems_in_dir = [d for d in elems_in_dir if os.path.isdir(os.path.join(path_to_ds, d)) and not d.startswith("6")]
    elems_in_dir = sorted(elems_in_dir) #sort to ensure reproducibility and deterministic sampling
    class_idx_to_img_paths = {}
    for dir in elems_in_dir:
        class_idx_to_img_paths[dir] = [os.path.join(path_to_ds,dir,elem) for elem in os.listdir(os.path.join(path_to_ds, dir)) if elem.endswith(".png")]
    
    class_idx_to_sampled_img_paths = {}
    for key in class_idx_to_img_paths.keys():
        class_idx_to_sampled_img_paths[key] = class_idx_to_img_paths[key][:run_params['n_image_samples_per_class']]
    
    return class_idx_to_sampled_img_paths

### transform to feature space ###
def get_normalized_image_encoding(img, backbone, preprocess,cuda_device):
    with torch.no_grad():
        image_input = preprocess(img).unsqueeze(0).to(cuda_device)
        image_features = backbone.encode_image(image_input)
        image_features = F.normalize(image_features, dim=-1) #is this the right dimensionality?
    return image_features

def get_class_idx_to_image_features(class_idx_to_sampled_img_paths, backbone, preprocess,cuda_device):
    class_idx_to_image_features = {}
    for idx, sampled_image_paths in tqdm.tqdm(class_idx_to_sampled_img_paths.items()):
        acc = []
        for path in sampled_image_paths:
            img = PIL.Image.open(path)
            img_features = get_normalized_image_encoding(img, backbone, preprocess,cuda_device)
            img_features = img_features.squeeze(0)
            acc.append(img_features)
        class_idx_to_image_features[idx] = torch.stack(acc)
    return class_idx_to_image_features

def get_class_idx_to_paths_to_crop_features(class_idx_to_paths_to_sampled_crops, backbone, preprocess,cuda_device):
    class_idx_to_paths_to_crop_features = {}
    for idx, path_to_crops in tqdm.tqdm(class_idx_to_paths_to_sampled_crops.items()):
        class_idx_to_paths_to_crop_features[idx] = {}
        for path, crops in path_to_crops.items():
            acc = []
            for crop in crops:
                crop_features = get_normalized_image_encoding(crop, backbone, preprocess,cuda_device)
                crop_features = crop_features.squeeze(0)
                acc.append(crop_features)
            class_idx_to_paths_to_crop_features[idx][path] = torch.stack(acc)
    return class_idx_to_paths_to_crop_features

def get_normalized_text_encoding(text, backbone, cuda_device):
    with torch.no_grad():
        text_input = clip.tokenize([text]).to(cuda_device)
        text_features = backbone.encode_text(text_input)
        text_features = F.normalize(text_features, dim=-1)
    return text_features

def get_class_idx_to_description_features(descriptions, backbone, cuda_device):
    class_idx_to_description_features = {}
    for idx, descriptions in tqdm.tqdm(descriptions.items()):
        acc = []
        for description in descriptions:
            description_features = get_normalized_text_encoding(description, backbone, cuda_device)
            description_features = description_features.squeeze(0)
            acc.append(description_features)
        class_idx_to_description_features[idx] = torch.stack(acc)
    return class_idx_to_description_features


def get_patch_to_lang_accuracy_majority_vote(class_idx_to_paths_to_crop_features, class_idx_to_description_features,info_str,run_params):
    save_dict = {}
    top_1_accuracy = Accuracy(task="multiclass",num_classes=len(class_idx_to_description_features.keys())).to(run_params['device'])
    n_descriptions_per_class = run_params['n_descriptions_per_class']

    img_path_to_crop_features = [v for k,v in class_idx_to_paths_to_crop_features.items()]
    crop_vector_list = [v for dictionary in img_path_to_crop_features for k,v in dictionary.items()]
    crop_vector_list = torch.cat(crop_vector_list)
    
    description_feature_vector_list = [v for k,v in class_idx_to_description_features.items()]
    description_feature_vector_list = torch.cat(description_feature_vector_list)
    
    #calculate dot product of every feature vector to every feature vector
    similarity_matrix = torch.mm(crop_vector_list,description_feature_vector_list.T)
    top_1_predictions = similarity_matrix.max(dim=1).indices // n_descriptions_per_class
    top_1_predictions_mj_vote = torch.mode(top_1_predictions.view(-1,run_params['imp_samp_params']['patches_per_image']),dim=1).values

    info_str = f'patch-based {info_str}'
    save_dict['type'] = info_str
    save_dict['top_1_predictions'] = str(top_1_predictions)
    
    img_labels = [[label for _ in range(run_params['n_image_samples_per_class'])] for label in list(class_idx_to_paths_to_crop_features.keys())]
    img_labels_flattened = [int(item) for sublist in img_labels for item in sublist]
    
    top_1_acc = top_1_accuracy(top_1_predictions_mj_vote,torch.tensor(img_labels_flattened,device=run_params['device']))
    save_dict['top_1_accuracy'] = round(float(top_1_acc)*100,2)
    save_dict['run_params'] = run_params

    with open(os.path.join(run_params['tmp_save_dir'],f'{info_str}_top_1_accuracy_mj_vote.yaml'), 'w') as f:
        yaml.dump(save_dict, f)

def get_patch_to_lang_accuracy(class_idx_to_paths_to_crop_features, class_idx_to_description_features,info_str,run_params):
    save_dict = {}
    top_1_accuracy = Accuracy(task="multiclass",num_classes=len(class_idx_to_description_features.keys())).to(run_params['device'])
    n_descriptions_per_class = run_params['n_descriptions_per_class']

    img_path_to_crop_features = [v for k,v in class_idx_to_paths_to_crop_features.items()]
    crop_vector_list = [v for dictionary in img_path_to_crop_features for k,v in dictionary.items()]
    crop_vector_list = torch.cat(crop_vector_list)
    
    description_feature_vector_list = [v for k,v in class_idx_to_description_features.items()]
    description_feature_vector_list = torch.cat(description_feature_vector_list)
    
    #calculate dot product of every feature vector to every feature vector
    similarity_matrix = torch.mm(crop_vector_list,description_feature_vector_list.T)
    top_1_predictions = similarity_matrix.max(dim=1).indices // n_descriptions_per_class

    info_str = f'patch-based {info_str}'
    save_dict['type'] = info_str
    save_dict['top_1_predictions'] = str(top_1_predictions)
    
    crop_labels = [[label for _ in range(run_params['n_image_samples_per_class']*run_params['imp_samp_params']['patches_per_image'])] for label in list(class_idx_to_description_features.keys())]
    crop_labels_flattened = [int(item) for sublist in crop_labels for item in sublist]
    
    top_1_acc = top_1_accuracy(top_1_predictions,torch.tensor(crop_labels_flattened,device=run_params['device']))
    save_dict['top_1_accuracy'] = round(float(top_1_acc)*100,2)
    save_dict['run_params'] = run_params

    with open(os.path.join(run_params['tmp_save_dir'],f'{info_str}_top_1_accuracy.yaml'), 'w') as f:
        yaml.dump(save_dict, f)


def get_img_to_lang_accuracy(class_idx_to_img_features, class_idx_to_description_features,info_str,run_params):
    save_dict = {}
    top_1_accuracy = Accuracy(task="multiclass",num_classes=len(class_idx_to_description_features.keys())).to(run_params['device'])
    n_descriptions_per_class = run_params['n_descriptions_per_class']

    image_feature_vector_list = [v for k,v in class_idx_to_img_features.items()]
    image_feature_vector_list = torch.cat(image_feature_vector_list)
    
    description_feature_vector_list = [v for k,v in class_idx_to_description_features.items()]
    description_feature_vector_list = torch.cat(description_feature_vector_list)
    
    #calculate dot product of every feature vector to every feature vector
    similarity_matrix = torch.mm(image_feature_vector_list,description_feature_vector_list.T)
    top_1_predictions = similarity_matrix.max(dim=1).indices // n_descriptions_per_class

    info_str = f'image-based {info_str}'
    save_dict['type'] = info_str
    save_dict['top_1_predictions'] = str(top_1_predictions)
    
    img_labels = [[label for _ in range(run_params['n_image_samples_per_class'])] for label in list(class_idx_to_img_features.keys())]
    img_labels_flattened = [int(item) for sublist in img_labels for item in sublist]
    
    top_1_acc = top_1_accuracy(top_1_predictions,torch.tensor(img_labels_flattened,device=run_params['device']))
    save_dict['top_1_accuracy'] = round(float(top_1_acc)*100,2)
    save_dict['run_params'] = run_params

    with open(os.path.join(run_params['tmp_save_dir'],f'{info_str}_top_1_accuracy.yaml'), 'w') as f:
        yaml.dump(save_dict, f)


def eval_microbiology(run_params):

    class_idx_to_sampled_img_paths = get_non_random_images(run_params)
    class_idx_to_sampled_img_paths_to_sampled_crops = sample_important_crops_per_image(class_idx_to_sampled_img_paths, run_params['imp_samp_params']['patches_per_image'], run_params['imp_samp_params'])

    backbone, preprocess = clip.load(run_params['backbone_name'], device=run_params['device'], jit=False)

    class_idx_to_image_features = get_class_idx_to_image_features(class_idx_to_sampled_img_paths, backbone, preprocess,run_params['device'])
    class_idx_to_paths_crop_features = get_class_idx_to_paths_to_crop_features(class_idx_to_sampled_img_paths_to_sampled_crops, backbone, preprocess,run_params['device'])
    
    with open(run_params['idx_to_classname_dir'], "r") as f:
        idx_to_classname = json.load(f)
    classnames = list(idx_to_classname.values())

    for description_path in run_params['idx_to_descriptions_dirs']:
        run_params['timestamp'] = datetime.datetime.now().strftime('%m-%d_%H-%M-%S')
        run_params['lang_path'] = description_path #for plotting later on

        with open(description_path, "r") as f:
            idx_to_descriptions = json.load(f)

        class_idx_to_sentence_descriptions = get_sentence_descriptions(idx_to_classname, idx_to_descriptions,run_params['max_len'])
        if type(list(class_idx_to_sentence_descriptions.values())[0]) == str:
            n_descriptions_per_class = len(list(class_idx_to_sentence_descriptions.values())[0].split("\n"))
        elif type(list(class_idx_to_sentence_descriptions.values())[0]) == list:
            n_descriptions_per_class = len(list(class_idx_to_sentence_descriptions.values())[0])
        run_params['n_descriptions_per_class'] = n_descriptions_per_class
        class_idx_to_raw_descriptions = get_raw_descriptions(idx_to_classname, idx_to_descriptions,run_params['max_len'])
        
        sentence_description_features = get_class_idx_to_description_features(class_idx_to_sentence_descriptions, backbone, run_params['device'])
        raw_description_features = get_class_idx_to_description_features(class_idx_to_raw_descriptions, backbone, run_params['device'])
        classnames_in_description_format = {i:[classname for _ in range(n_descriptions_per_class)] for i,classname in enumerate(classnames)}
        classname_features = get_class_idx_to_description_features(classnames_in_description_format, backbone, run_params['device'])

        run_params['tmp_save_dir'] = create_eval_dir(run_params)

        get_patch_to_lang_accuracy(class_idx_to_paths_crop_features, classname_features,"classname features",run_params)
        get_patch_to_lang_accuracy(class_idx_to_paths_crop_features, raw_description_features,"raw description features",run_params)
        get_patch_to_lang_accuracy(class_idx_to_paths_crop_features, sentence_description_features,"sentence description features",run_params)

        get_patch_to_lang_accuracy_majority_vote(class_idx_to_paths_crop_features, classname_features,"mj vote classname features",run_params)
        get_patch_to_lang_accuracy_majority_vote(class_idx_to_paths_crop_features, raw_description_features,"mj vote raw description features",run_params)
        get_patch_to_lang_accuracy_majority_vote(class_idx_to_paths_crop_features, sentence_description_features,"mj vote sentence description features",run_params)

        get_img_to_lang_accuracy(class_idx_to_image_features, classname_features,"classname features",run_params)
        get_img_to_lang_accuracy(class_idx_to_image_features, raw_description_features,"raw description features",run_params)
        get_img_to_lang_accuracy(class_idx_to_image_features, sentence_description_features,"sentence description features",run_params)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_idx', type=int, default=0)
    args = parser.parse_args()
    cfg_idx = args.cfg_idx
    run_params = yaml.load(open('cfgs/run/cfg_{}.yml'.format(cfg_idx), 'r'), Loader=yaml.FullLoader)

    backbone_names = run_params['backbone_names']
    
    for backbone in backbone_names:
        run_params['backbone_name'] = backbone
        eval_microbiology(run_params)



