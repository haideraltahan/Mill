from mill.models.transformers import TransformersModel

model = dict(
    type=TransformersModel,
    abbr="qwen2.5-7b-instruct",
    path="Qwen/Qwen2.5-7B-Instruct",
    modalities=["text"],
    dtype="bfloat16",
    max_context_length=32768,
    run_cfg=dict(num_gpus=1, batch_size=16),
)
