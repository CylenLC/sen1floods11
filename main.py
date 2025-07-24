import lightning.pytorch as pl
from terratorch.datamodules import Sen1Floods11NonGeoDataModule
from terratorch.tasks import SemanticSegmentationTask
from lightning.pytorch.callbacks import LearningRateMonitor, RichProgressBar
from lightning.pytorch import Trainer


def main():
    pl.seed_everything(0, workers=True)

    # 配置数据模块
    datamodule = Sen1Floods11NonGeoDataModule(
        batch_size=8,
        num_workers=4,
        data_root="sen1floods11",
        bands=[
            "BLUE",
            "GREEN",
            "RED",
            "NIR_NARROW",
            "SWIR_1",
            "SWIR_2",
        ]
    )

    model_args = dict(
        backbone="prithvi_eo_v2_300_tl",
        backbone_pretrained=True,
        num_classes=2,
        backbone_bands=[
            "BLUE",
            "GREEN",
            "RED",
            "NIR_NARROW",
            "SWIR_1",
            "SWIR_2",
        ],
        decoder="UNetDecoder",
        decoder_channels=[512, 256, 128, 64],
        head_channel_list=[256],
        head_dropout=0.1,
        necks=[{"name": "SelectIndices", "indices": [5, 11, 17, 23]},
               {"name": "ReshapeTokensToImage"},
               {"name": "LearnedInterpolateToPyramidal"},],
    )

    task = SemanticSegmentationTask(
        model_args,
        "EncoderDecoderFactory",
        loss="dice",
        lr=1e-4,
        ignore_index=-1,
        freeze_backbone=False,
        optimizer="AdamW",

        scheduler="ReduceLROnPlateau",

    )

    trainer = Trainer(
        accelerator="auto",
        max_epochs=100,
        strategy="auto",
        devices="auto",
        num_nodes=1,
        precision="16-mixed",
        logger=True,
        callbacks=[
            RichProgressBar(),
            LearningRateMonitor(logging_interval="epoch")],
        log_every_n_steps=5,
        enable_checkpointing=True,
        default_root_dir="output/prithvi/experiment"
    )

    trainer.fit(model=task, datamodule=datamodule)

if __name__ == '__main__':
    main()