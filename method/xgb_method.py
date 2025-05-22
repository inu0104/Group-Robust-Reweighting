import argparse
import random
import optuna
import torch
import torch.nn.functional as F

from collections import Counter
import numpy as np
import logging
import json
import sys

from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from utils.data_loader import load_data, dataloader_to_numpy


################## utils ##########################
def compute_sample_weights(group_labels, alpha, min_weight):
        group_counts = Counter(group_labels)
        group_weights = {
            g: max(min_weight, (1 / count) ** alpha)
            for g, count in group_counts.items()
        }
        return np.array([group_weights[g] for g in group_labels])


def compute_group_loss_weights(y_true, y_prob, group_labels, temp=1.0):
    eps = 1e-12
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    group_labels = np.array(group_labels)

    losses = - (y_true * np.log(y_prob + eps) + (1 - y_true) * np.log(1 - y_prob + eps))

    group_ids = np.unique(group_labels)
    group_losses = {g: losses[group_labels == g].mean() for g in group_ids}

    raw = np.array([group_losses[g] for g in group_ids])
    scaled = np.exp(raw / temp)
    group_weights = {g: scaled[i] for i, g in enumerate(group_ids)}

    total = sum(group_weights.values())
    group_weights = {g: w / total for g, w in group_weights.items()}

    return np.array([group_weights[g] for g in group_labels])


def compute_sample_weights_dro(y_true, y_prob, mode, min_weight=0.2):
    eps = 1e-12
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    losses = - (y_true * np.log(y_prob + eps) + (1 - y_true) * np.log(1 - y_prob + eps))

    if mode == 'linear':
        sample_weights = losses
    elif mode == 'minmax':
        loss_min = losses.min()
        loss_max = losses.max()
        if loss_max - loss_min < 1e-6:
            losses = np.ones_like(losses)
        else:
            losses = (losses - loss_min) / (loss_max - loss_min)
        sample_weights = min_weight + (1 - min_weight) * losses
    else:
        raise ValueError("Unsupported mode: choose from ['linear', 'minmax']")

    return sample_weights


################## Method1. ERM ##########################
def run_xgb_erm(train_loader, valid_loader, test_loader, train_params):
    print(f"🔥 Running xgb_erm Method")
    x_train, y_train, _ = dataloader_to_numpy(train_loader)
    x_valid, y_valid, _ = dataloader_to_numpy(valid_loader)
    
    model = XGBClassifier(**train_params)
    model.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],
        verbose=False
    )
    return model


################## Method2. GroupIndep ##########################
def run_xgb_GroupIndep(train_loader, valid_loader, test_loader, train_params):
    print(f"🔥 Running xgb_GroupIndep Method")
    print("\nGroup-Wise:")
    print(f"{'Group':>6} | {'Ratio (%)':>9} | {'AUC':>6} ")
    print("-" * 30)
    x_train, y_train, group_train = dataloader_to_numpy(train_loader)
    x_valid, y_valid, group_valid = dataloader_to_numpy(valid_loader)
    x_test, y_test, group_test = dataloader_to_numpy(test_loader)
    
    group_ids = np.unique(group_train)
    total = len(y_test)
    
    for g in group_ids:
        group_valid = np.array(group_valid)
        x_valid = np.array(x_valid)
        y_valid = np.array(y_valid)
        valid_idx = group_valid == g
        x_valid_g = x_valid[valid_idx]
        y_valid_g = y_valid[valid_idx]
        
        group_test = np.array(group_test)
        x_test = np.array(x_test)
        y_test = np.array(y_test)
        test_idx = group_test == g
        x_test_g = x_test[test_idx]
        y_test_g = y_test[test_idx]
        
        
        model = XGBClassifier(**train_params)
        model.fit(
            x_train, y_train,
            eval_set=[(x_valid_g, y_valid_g)],
            verbose=False
        )
        
        group_size = len(y_test_g)
        ratio = group_size / total * 100
        y_prob_g = model.predict_proba(x_test_g)[:, 1]
        
        try:
            auc_g = roc_auc_score(y_test_g, y_prob_g)
        except ValueError:
            auc_g = float('nan')

        print(f"{g:>6} | {ratio:9.2f} | {auc_g:6.4f}")
    sys.exit(0)    
    return 0


################## Method3. Inverse-size ##########################
def run_xgb_IS(train_loader, valid_loader, test_loader, train_params):
    
    x_train, y_train, group_train = dataloader_to_numpy(train_loader) 
    x_valid, y_valid, _ = dataloader_to_numpy(valid_loader)
    
    alpha = train_params.get("alpha", 1.2)
    min_weight = train_params.get("min_weight", 0.6)
    sample_weights = compute_sample_weights(group_train, alpha, min_weight)

    model = XGBClassifier(**train_params)
    model.fit(
            x_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(x_valid, y_valid)],
            verbose=False
        )
    return model


################## Method4. Group-wise ##########################
def run_xgb_GW(train_loader, valid_loader, test_loader, train_params):
    
    x_train, y_train, group_train = dataloader_to_numpy(train_loader) 
    x_valid, y_valid, group_valid = dataloader_to_numpy(valid_loader)
    
    sample_weights = np.ones(len(x_train))
    best_model = None
    best_worst_auc = -np.inf

    iterations = train_params.get("iterations", 10)
    lr = train_params.get("lr", 0.01)
    ne = train_params.get("n_estimators", 200)
    md = train_params.get("max_depth", 5)
    temp = train_params.get("temp", 0.9)
    
    for t in range(iterations):
        
        model = XGBClassifier(
            learning_rate=lr, 
            n_estimators=ne, 
            max_depth=md,
            eval_metric='logloss', 
            use_label_encoder=False, 
            verbosity=0
        )
        model.fit(x_train, y_train, sample_weight=sample_weights)

        y_prob = model.predict_proba(x_train)[:, 1]
        sample_weights = compute_group_loss_weights(y_true=y_train, y_prob=y_prob, group_labels=group_train, temp=temp)
        
        val_prob = model.predict_proba(x_valid)[:, 1]
        group_valid = np.array(group_valid)
        y_valid = np.array(y_valid)

        aucs = []
        for g in np.unique(group_valid):
            mask = group_valid == g
            try:
                auc = roc_auc_score(y_valid[mask], val_prob[mask])
            except:
                auc = float('nan')
            aucs.append(auc)

        worst_auc = np.nanmin(aucs)

        if worst_auc > best_worst_auc:
            best_model = model

    return best_model


################## Method5. Sample-wise ##########################
def run_xgb_SW(train_loader, valid_loader, test_loader, train_params):
    
    x_train, y_train, group_train = dataloader_to_numpy(train_loader) 
    x_valid, y_valid, group_valid = dataloader_to_numpy(valid_loader)
    
    sample_weights = np.ones(len(x_train))
    best_model = None
    best_worst_auc = -np.inf

    iterations = train_params.get("iterations", 10)
    lr = train_params.get("lr", 0.01)
    ne = train_params.get("n_estimators", 200)
    md = train_params.get("max_depth", 5)
    mode = train_params.get("mode", 'linear')
    min_weight = train_params.get("min_weight", 0.1)
    
    for t in range(iterations):
        
        model = XGBClassifier(
            learning_rate=lr, 
            n_estimators=ne, 
            max_depth=md,
            eval_metric='logloss', 
            use_label_encoder=False, 
            verbosity=0
        )
        model.fit(x_train, y_train, sample_weight=sample_weights)

        y_prob = model.predict_proba(x_train)[:, 1]
        sample_weights = compute_sample_weights_dro(y_train, y_prob, mode=mode, min_weight=min_weight)
        
        val_prob = model.predict_proba(x_valid)[:, 1]
        group_valid = np.array(group_valid)
        y_valid = np.array(y_valid)

        aucs = []
        for g in np.unique(group_valid):
            mask = group_valid == g
            try:
                auc = roc_auc_score(y_valid[mask], val_prob[mask])
            except:
                auc = float('nan')
            aucs.append(auc)

        worst_auc = np.nanmin(aucs)

        if worst_auc > best_worst_auc:
            best_model = model

    return best_model