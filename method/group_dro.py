import copy
import torch
import torch.nn.functional as F
from utils.train import train_or_eval_model

class GroupDRO:
    def __init__(self, criterion, n_groups, step_size_q, device='cpu'):
        self.criterion = criterion
        self.n_groups = n_groups
        self.step_size_q = step_size_q
        self.q = torch.ones(n_groups, device=device) / n_groups
        self.device = device

    def loss(self, model, x, y, group_idx):
        loss_ls = []

        for g in range(self.n_groups):
            selected = (group_idx == g)
            if selected.sum() > 0:
                x_g = x[selected]
                y_g = y[selected].float().view(-1)
                yhat_g = model(x_g).view(-1)
                loss = self.criterion(yhat_g, y_g)
                loss_ls.append(loss)
            else:
                loss_ls.append(torch.tensor(0.0, device=self.device, requires_grad=True))
       
        q_prime = self.q.clone()
        for g in range(self.n_groups):
            q_prime[g] *= torch.exp(self.step_size_q * loss_ls[g].detach())
        self.q = q_prime / q_prime.sum()

        return sum(self.q[g] * loss_ls[g] for g in range(self.n_groups))


def run_group_dro(model, train_loader, valid_loader, train_df, train_params, device):
    print(f"🔥 Running GroupDRO Method on {device}...")

    n_groups = train_df['group'].nunique()
    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    is_tabnet = model.__class__.__name__.lower().startswith("tabnet")
    lambda_sparse = train_params.get("lambda_sparse", 1e-3) if is_tabnet else 0.0
    step_size_q = train_params.get("group_dro_eta", 0.01)

    group_dro = GroupDRO(criterion, n_groups=n_groups, step_size_q=step_size_q, device=device)

    def loss_fn_gdro(model, x, y, group=None, sample_ids=None):
        if is_tabnet:
            _, M_loss = model(x, return_loss=True)
            loss = group_dro.loss(model, x, y, group) 
            return loss + lambda_sparse * M_loss
        else:
            return group_dro.loss(model, x, y, group)

    model_gdro = copy.deepcopy(model)
    model_gdro = train_or_eval_model(
        model=model_gdro,
        train_loader=train_loader,
        valid_loader=valid_loader,
        params=train_params,
        device=device,
        mode="train",
        loss_fn=loss_fn_gdro
    )

    return model_gdro