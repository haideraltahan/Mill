from mill.models.transformers import TransformersModel

model = dict(
    type=TransformersModel,
    abbr="qwen2.5-vl-7b",
    path="Qwen/Qwen2.5-VL-7B-Instruct",
    modalities=["text", "image", "video"],
    dtype="bfloat16",
    max_context_length=32768,
    run_cfg=dict(num_gpus=1, batch_size=4),
)
