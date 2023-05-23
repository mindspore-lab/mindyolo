from mindyolo.models.registry import is_model, model_entrypoint

__all__ = ["create_loss"]


def create_loss(name: str, **kwargs):
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    if not is_model(name):
        raise RuntimeError(f"Unknown loss module {name}")

    create_fn = model_entrypoint(name)
    loss_fn = create_fn(**kwargs)

    return loss_fn
