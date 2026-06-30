from mill.models.transformers import TransformersModel

model = dict(
    type=TransformersModel,
    abbr="llama3-8b-instruct",
    path="meta-llama/Meta-Llama-3-8B-Instruct",
    modalities=["text"],
    dtype="bfloat16",
    max_context_length=8192,
    run_cfg=dict(num_gpus=1, batch_size=16),
)
