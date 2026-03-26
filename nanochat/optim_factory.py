"""
Decoupled optimizer factory for GPT models.
Extracted from GPT.setup_optimizer() to break the model -> optimizer coupling.
"""

from nanochat.common import get_dist_info, print0
from nanochat.optim import MuonAdamW, DistMuonAdamW


def make_optimizer(model, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5):
    model_dim = model.config.n_embd
    ddp, rank, local_rank, world_size = get_dist_info()

    # Separate out all parameters into groups
    matrix_params = list(model.transformer.h.parameters())
    value_embeds_params = list(model.value_embeds.parameters())
    embedding_params = list(model.transformer.wte.parameters())
    lm_head_params = list(model.lm_head.parameters())
    resid_params = [model.resid_lambdas]
    x0_params = [model.x0_lambdas]
    assert len(list(model.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params)

    # Scale the LR for the AdamW parameters by 1/sqrt(dmodel) (tuned for 768 dim model)
    dmodel_lr_scale = (model_dim / 768) ** -0.5
    print0(f"Scaling the LR for the AdamW parameters 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")

    # Build param_groups with all required fields explicit
    param_groups = [
        # AdamW groups (embeddings, lm_head, scalars)
        dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
        dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
        dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
        dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
        dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
    ]
    # Muon groups (matrix params, grouped by shape for stacking)
    for shape in sorted({p.shape for p in matrix_params}):
        group_params = [p for p in matrix_params if p.shape == shape]
        param_groups.append(dict(
            kind='muon', params=group_params, lr=matrix_lr,
            momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
        ))

    Factory = DistMuonAdamW if ddp else MuonAdamW
    optimizer = Factory(param_groups)
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
    return optimizer
