from mill.models.transformers import TransformersModel

model = dict(
    type=TransformersModel,
    abbr="internvl2-8b",
    path="OpenGVLab/InternVL2-8B",
    modalities=["text", "image", "video"],
    dtype="bfloat16",
    max_context_length=8192,
    trust_remote_code=True,
    run_cfg=dict(num_gpus=1, batch_size=4),
)
