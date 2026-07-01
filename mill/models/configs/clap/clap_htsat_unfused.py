from mill.models.hf_clip import HFContrastiveModel

model = dict(
    type=HFContrastiveModel,
    abbr="clap-htsat-unfused",
    path="laion/clap-htsat-unfused",
    modalities=["audio", "text"],
    dtype="float32",
    run_cfg=dict(num_gpus=1, batch_size=32),
)
