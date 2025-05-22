from sklearn.metrics import roc_auc_score
import numpy as np
import torch
import torch.nn.functional as F

def evaluate_group_metrics(model, test_loader, device, model_class):
    if model_class != 'xgb':
        model.eval()
        model.to(device)

    all_logits = []
    all_labels = []
    all_groups = []


    with torch.no_grad():
        for batch in test_loader:
    
            x, y, g, *_ = batch

            if model_class == 'xgb':
                x_np = x.numpy() if isinstance(x, torch.Tensor) else x
                probs = model.predict_proba(x_np)[:, 1] 
            else:
                x = x.to(device)
                outputs = model(x)  
                probs = torch.sigmoid(outputs).flatten().cpu().numpy()
                
            all_logits.append(probs)
            all_labels.append(y.numpy())
            all_groups.append(g.numpy())

    y_prob = np.concatenate(all_logits)
    y_true = np.concatenate(all_labels)
    group_ids = np.concatenate(all_groups)
    total = len(y_true)

    auc = roc_auc_score(y_true, y_prob)
    print(f"Overall AUC:       {auc:.4f}")

    print("\nGroup-Wise:")
    print(f"{'Group':>6} | {'Ratio (%)':>9} | {'AUC':>6} ")
    print("-" * 30)

    for g in np.unique(group_ids):
        idx = group_ids == g
        group_size = np.sum(idx)
        ratio = group_size / total * 100

        y_true_g = y_true[idx]
        y_prob_g = y_prob[idx]

        try:
            auc_g = roc_auc_score(y_true_g, y_prob_g)
        except ValueError:
            auc_g = float('nan')

        print(f"{g:>6} | {ratio:9.2f} | {auc_g:6.4f}")