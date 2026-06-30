from mill.models.transformers import TransformersModel

model = dict(
    type=TransformersModel,
    abbr="qwen2-audio-7b-instruct",
    path="Qwen/Qwen2-Audio-7B-Instruct",
    modalities=["text", "audio"],
    dtype="bfloat16",
    max_context_length=8192,
    run_cfg=dict(num_gpus=1, batch_size=4),
)
