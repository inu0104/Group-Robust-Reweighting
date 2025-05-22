import copy
import torch
from utils.train import train_or_eval_model
import torch.nn.functional as F

def run_erm(model, train_loader, valid_loader, train_df, train_params, device):
    print(f"🔥 Running ERM Method on {device}...")
    
    is_tabnet = model.__class__.__name__.lower() == "tabnet"

    if is_tabnet:
       
        def tabnet_loss_fn(model, x, y, *args, **kwargs):
            output, M_loss = model(x, return_loss=True)  # output: (B, 1)
            loss = F.binary_cross_entropy_with_logits(output.squeeze(), y.float())
            lambda_sparse = train_params.get("lambda_sparse", 1e-3)
            return loss + lambda_sparse * M_loss
        
        loss_fn = tabnet_loss_fn
    else:
        loss_fn = None  

    model_erm = copy.deepcopy(model)
    model_erm = train_or_eval_model(
        model=model_erm,
        train_loader=train_loader,
        valid_loader=valid_loader,
        params=train_params, 
        device=device,
        mode="train",
        loss_fn=loss_fn,
    )
    
    return model_erm