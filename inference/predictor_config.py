"""Shared cardiac predictor settings for source fitting and fixed inference."""


def resolve_cardiac_predictor(primary_type="base", reference_type="mini", *, primary_cond_scale=None, reference_cond_scale=None) -> dict:
    return dict(primary_type=primary_type, reference_type=reference_type,
                primary_scale=1.0 if primary_cond_scale is None else primary_cond_scale,
                reference_scale=1.0 if reference_cond_scale is None else reference_cond_scale, timesteps=50,
                num_steps=5, sample_mode="one_hot", inference_stochastic=False,
                final_step_mode="majority")


def sample_cardiac(primary, reference, images, tuple_params, *, predictor=None):
    config = resolve_cardiac_predictor() if predictor is None else predictor
    return primary.sample_full_tac(
        channel_cond=images, reference_denoise_fn=reference.denoise_fn,
        primary_cond_scale=config["primary_scale"],
        reference_cond_scale=config["reference_scale"],
        num_steps=config["num_steps"], final_step_mode=config["final_step_mode"],
        **tuple_params,
    )
