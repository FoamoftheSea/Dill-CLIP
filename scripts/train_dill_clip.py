from argparse import ArgumentParser
from pathlib import Path
from typing import Mapping, List, Dict, Any

import numpy as np
import torch
import wandb
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    PvtV2Config,
    DeformableDetrImageProcessor,
    DillCLIPVisionConfig,
    DillCLIPVisionModelForRegression,
)
from transformers.data.data_collator import InputDataClass
from transformers.training_args import OptimizerNames
from transformers.utils import logging

logger = logging.get_logger(__name__)
logger.setLevel(logging.DEBUG)

image_processor = DeformableDetrImageProcessor.from_pretrained("sensetime/deformable-detr")


def compute_metrics(eval_pred) -> dict:

    with torch.no_grad():
        predictions, labels = eval_pred

        mse_metric = torch.nn.MSELoss()
        mae_metric = torch.nn.L1Loss()

        mse = mse_metric(predictions, labels)
        mae = mae_metric(predictions, labels)

        return {"MSE": mse, "MAE": mae}


def dill_clip_collator(features: List[InputDataClass]) -> Dict[str, Any]:

    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}

    for k, v in first.items():
        if k == "labels":
            batch[k] = [torch.tensor(f[k]) for f in features]
        elif k == "pixel_values":
            processed = image_processor([f[k] for f in features])
            batch["pixel_values"] = torch.tensor(processed.data["pixel_values"])
            batch["pixel_mask"] = torch.tensor(processed.data["pixel_mask"])

    return batch


class DillCLIPDataset(Dataset):

    def __init__(self, data_root: str, split: str = "train", num_workers: int = 1):
        self.split = split
        self.data_root = Path(data_root) / f"{split}2017"
        self.num_workers = num_workers
        self.frames = self._get_frames()


    def _get_frames(self):
        img_paths = list(self.data_root.glob("*.jpg"))
        target_paths = [
            self.data_root / "clip_soft_labels" / f"{img_path.stem}.npy" for img_path in img_paths
        ]

        return list(zip(img_paths, target_paths))

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx: int):
        img_path, target_path = self.frames[idx]
        img = Image.open(img_path)
        target = np.load(target_path)

        return {"pixel_values": img, "labels": target}


def main(args):
    config = DillCLIPVisionConfig(
        use_timm_backbone=False,
        backbone_config=PvtV2Config.from_pretrained("FoamoftheSea/pvt_v2_b4"),
        num_queries=257,
        max_position_embeddings=1024,
        encoder_layers=3,
        encoder_ffn_dim=1024,
        encoder_attention_heads=32,
        decoder_layers=3,
        decoder_ffn_dim=1024,
        decoder_attention_heads=32,
        encoder_layerdrop=0.0,
        is_encoder_decoder=True,
        activation_function="relu",
        d_model=1024,
        dropout=0.1,
        attention_dropout=0.0,
        activation_dropout=0.0,
        init_std=0.02,
        init_xavier_std=1.0,
        return_intermediate=True,
        auxiliary_loss=False,
        position_embedding_type="sine",
        dilation=False,
        num_feature_levels=4,
        encoder_n_points=4,
        decoder_n_points=4,
        two_stage=False,
        disable_custom_kernels=False,
    )

    model = DillCLIPVisionModelForRegression(config)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        save_total_limit=args.save_total_limit,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=1,
        load_best_model_at_end=True,
        dataloader_num_workers=args.workers,
        seed=args.seed,
        max_steps=args.max_steps,
        tf32=args.use_tf32,
        optim=OptimizerNames.ADAMW_8BIT if args.use_adam8bit else OptimizerNames.ADAMW_TORCH,
        dataloader_pin_memory=False if args.workers > 0 else True,
        # lr_scheduler_type=SchedulerType.COSINE,
        # push_to_hub=True,
        # hub_model_id=hub_model_id,
        # hub_strategy="end",
    )

    train_dataset = DillCLIPDataset(data_root=args.data_root, split="train")
    val_dataset = DillCLIPDataset(data_root=args.data_root, split="val")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=dill_clip_collator,
    )
    if args.eval_only:
        trainer.evaluate()
    else:
        trainer.train()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-o", "--output_dir", type=str, default="./segformer_output", help="Output dir to store results.")
    parser.add_argument("-d", "--data-root", type=str, default="E:/shift/", help="Path to SHIFT dataset.")
    parser.add_argument("-w", "--workers", type=int, default=0, help="Number of data loader workers.")
    parser.add_argument("-lr", "--learning-rate", type=float, default=0.00006, help="Initial learning rate for training.")
    parser.add_argument("-e", "--epochs", type=int, default=5, help="Number of epochs to run training.")
    parser.add_argument("-bs", "--batch-size", type=int, default=1, help="Train batch size.")
    parser.add_argument("-ebs", "--eval-batch-size", type=int, default=None, help="Eval batch size. Defaults to train batch size.")
    parser.add_argument("-gas", "--gradient-accumulation-steps", type=int, default=2, help="Number of gradient accumulation steps.")
    parser.add_argument("-gc", "--gradient-checkpointing", action="store_true", default=False, help="Use gradient checkpointing.")
    parser.add_argument("-es", "--eval-steps", type=int, default=5000, help="Number of steps between validation runs.")
    parser.add_argument("-ss", "--save-steps", type=int, default=None, help="Number of steps between checkpoints. Defaults to eval steps.")
    parser.add_argument("-ms", "--max-steps", type=int, default=-1, help="Set to limit the number of total training steps.")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed for training.")
    parser.add_argument("-tf32", "--use-tf32", action="store_true", default=False, help="Set to True if your setup supports TF32 dtype.")
    parser.add_argument("-bnb", "--use-adam8bit", action="store_true", default=False, help="Use ADAMW_8BIT optimizer (linux only).")
    parser.add_argument("-c", "--checkpoint", type=str, default=None, help="Path to checpoint to resume training.")
    parser.add_argument("-rwb", "--resume-wandb", type=str, default=None, help="ID of run to resume")
    parser.add_argument("-eval", "--eval-only", action="store_true", default=False, help="Only run evaluation step.")
    parser.add_argument("-stl", "--save-total-limit", type=int, default=None, help="Maximum number of checkpoints to store at once.")

    args = parser.parse_args()
    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size
    if args.save_steps is None:
        args.save_steps = args.eval_steps

    if args.use_tf32:
        logger.info("Using TF32 dtype.")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if args.resume_wandb:
        wandb.init(project="huggingface", resume="must", id=args.resume_wandb)

    main(args)
