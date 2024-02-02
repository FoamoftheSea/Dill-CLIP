from argparse import ArgumentParser
from pathlib import Path
from typing import Mapping, List, Dict, Any

import numpy as np
import torch
import wandb
from dill_clip.dataset.dill_clip_dataset import (
    DillCLIPTrainDataset,
    DillCLIPValDataset,
    DillCLIPLocalTrainDataset,
    DillCLIPLocalValDataset,
)
from transformers import (
    CLIPVisionModelWithProjection,
    Trainer,
    TrainingArguments,
    PvtV2Config,
    DeformableDetrImageProcessor,
    DillCLIPVisionConfig,
    DillCLIPVisionModelForRegression,
    CLIPImageProcessor,
)
from transformers.data.data_collator import InputDataClass
from transformers.training_args import OptimizerNames
from transformers.utils import logging
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

logger = logging.get_logger(__name__)
logger.setLevel(logging.DEBUG)

# image_processor = DeformableDetrImageProcessor.from_pretrained(
#     "sensetime/deformable-detr",
#     image_mean=OPENAI_CLIP_MEAN,
#     image_std=OPENAI_CLIP_STD
# )
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
clip_vision_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").to(device)
clip_vision_model.eval()
zeroshot_weights = torch.tensor(np.load("./scripts/zeroshot_weights.npy").T).to(device)
target_clip_layer = -2


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def calc_accuracy(pooled, target):
    with torch.no_grad():
        # predict
        image_features = clip_vision_model.visual_projection(pooled)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits = 100. * image_features @ zeroshot_weights

        # measure accuracy
        acc1, acc5 = accuracy(logits, target, topk=(1, 5))

    return acc1, acc5


def search_frames(path: Path, ext: str):
    return set(path.glob(f".{ext}"))


class DillCLIPEvalMetric:

    def __init__(self):
        self.reset_metric()

    def reset_metric(self):
        self.batch_mse = []
        self.batch_mae = []
        self.batch_top1_acc = []
        self.batch_top5_acc = []

    def update(self, pred, pooled, labels, target):
        mse_loss = torch.nn.MSELoss()
        mae_loss = torch.nn.L1Loss()
        labels = torch.stack(labels)
        self.batch_mse.append(mse_loss(pred, labels).detach().cpu().numpy())
        self.batch_mae.append(mae_loss(pred, labels).detach().cpu().numpy())

        top1, top5 = calc_accuracy(pooled, target)
        self.batch_top1_acc.append(top1)
        self.batch_top5_acc.append(top5)

    def calculate(self):
        total_mae = np.mean(self.batch_mae)
        output = {
            "mse": np.mean(self.batch_mse),
            "mae": total_mae,
            "top1_acc": np.mean(self.batch_top1_acc),
            "top5_acc": np.mean(self.batch_top5_acc),
            "eval_loss": total_mae,
        }
        self.reset_metric()
        return output


dill_metric = DillCLIPEvalMetric()


def compute_metrics(eval_pred, compute_result=True) -> dict:

    with torch.no_grad():
        outputs, labels = eval_pred
        target = outputs[1]
        pred = outputs[2]
        pooled = outputs[3]
        dill_metric.update(pred, pooled, labels, target)

    if compute_result:
        return dill_metric.calculate()


def dill_clip_online_collator(features: List[InputDataClass]) -> Dict[str, Any]:

    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}

    for k, v in first.items():
        if k == "pixel_values":
            processed = image_processor([f[k] for f in features], return_tensors="pt", padding=True)
            pixel_values = processed.data["pixel_values"]
            with torch.no_grad():
                outputs = clip_vision_model(pixel_values=pixel_values.to(device), output_hidden_states=True)
                batch["labels"] = [hs.cpu() for hs in outputs.hidden_states[target_clip_layer]]
            batch["pixel_values"] = pixel_values
            batch["pixel_mask"] = processed.data.get("pixel_mask", None)
        elif k == "targets":
            batch["targets"] = torch.tensor(np.array([f[k] for f in features]))

    return batch


def dill_clip_collator(features: List[InputDataClass]) -> Dict[str, Any]:

    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}

    for k, v in first.items():
        if k == "labels":
            batch[k] = [torch.tensor(f[k]) for f in features]
        elif k == "pixel_values":
            processed = image_processor([f[k] for f in features], return_tensors="pt")
            pixel_values = processed.data["pixel_values"]
            batch["pixel_values"] = pixel_values
            batch["pixel_mask"] = processed.data.get("pixel_mask", None)
        elif k == "targets":
            batch["targets"] = torch.tensor(np.array([f[k] for f in features]))

    return batch


def main(args):
    config = DillCLIPVisionConfig(
        use_timm_backbone=False,
        backbone_config=PvtV2Config(
            depths=[2, 2, 2, 2],
            hidden_sizes=[32, 64, 160, 256],
            # hidden_sizes=[64, 128, 320, 512],
            mlp_ratios=[8, 8, 4, 4],
            num_attention_heads=[1, 2, 5, 8],
            num_encoder_blocks=4,
            patch_sizes=[7, 3, 3, 3],
            sr_ratios=[8, 4, 2, 1],
            strides=[4, 2, 2, 2],
            drop_path_rate=0.1,
        ),
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
        disable_custom_kernels=True,
    )

    model = DillCLIPVisionModelForRegression(config)

    # train_dataset = DillCLIPTrainDataset()
    # val_dataset = DillCLIPValDataset(num_workers=args.workers)
    train_dataset = DillCLIPLocalTrainDataset()
    val_dataset = DillCLIPLocalValDataset(data_root="/mnt/d/ILSVRC/Data/CLS-LOC/")

    # optimizer = torch.optim.AdamW(
    #     params=model.parameters(),
    #     lr=args.learning_rate,
    # )

    params = [
        {"params": model.model.backbone.parameters(), "lr": args.learning_rate * 5},
        {"params": model.model.encoder.parameters()},
        {"params": model.model.decoder.parameters()},
        {"params": model.model.input_proj.parameters()},
        {"params": model.model.level_embed},
        {"params": model.model.query_position_embeddings.parameters()},
        {"params": model.model.reference_points.parameters()},
    ]
    if args.use_adam8bit:
        import bitsandbytes as bnb
        optimizer = bnb.optim.Adam8bit(
            params=params,
            lr=args.learning_rate,
        )
    else:
        optimizer = torch.optim.AdamW(
            params=params,
            lr=args.learning_rate,
        )

    lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer=optimizer, factor=1.0, total_iters=0)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer=optimizer,
    #     T_max=(len(train_dataset)//(args.gradient_accumulation_steps*args.batch_size))*args.epochs,
    # )
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
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
        bf16=args.use_bf16,
        bf16_full_eval=args.use_bf16,
        optim=OptimizerNames.ADAMW_8BIT if args.use_adam8bit else OptimizerNames.ADAMW_TORCH,
        dataloader_pin_memory=False if args.workers > 0 else True,
        batch_eval_metrics=True,
        log_outputs=True,
        torch_compile=True,
        torch_compile_backend="inductor",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=dill_clip_collator,
        optimizers=(optimizer, lr_scheduler),
    )
    if args.eval_only:
        trainer.evaluate()
    else:
        trainer.train()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-o", "--output_dir", type=str, default="./dill_clip_b0_output", help="Output dir to store results.")
    parser.add_argument("-d", "--data-root", nargs="*", type=str, default=["D:/", "E:/"], help="Folder containing dataset.")
    parser.add_argument("-w", "--workers", type=int, default=0, help="Number of data loader workers.")
    parser.add_argument("-lr", "--learning-rate", type=float, default=0.0002, help="Initial learning rate for training.")
    parser.add_argument("-e", "--epochs", type=int, default=1, help="Number of epochs to run training.")
    parser.add_argument("-bs", "--batch-size", type=int, default=1, help="Train batch size.")
    parser.add_argument("-ebs", "--eval-batch-size", type=int, default=None, help="Eval batch size. Defaults to train batch size.")
    parser.add_argument("-gas", "--gradient-accumulation-steps", type=int, default=32, help="Number of gradient accumulation steps.")
    parser.add_argument("-gc", "--gradient-checkpointing", action="store_true", default=False, help="Use gradient checkpointing.")
    parser.add_argument("-es", "--eval-steps", type=int, default=5000, help="Number of steps between validation runs.")
    parser.add_argument("-ss", "--save-steps", type=int, default=None, help="Number of steps between checkpoints. Defaults to eval steps.")
    parser.add_argument("-ms", "--max-steps", type=int, default=-1, help="Set to limit the number of total training steps.")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed for training.")
    parser.add_argument("-tf32", "--use-tf32", action="store_true", default=False, help="Set to True if your setup supports TF32 dtype.")
    parser.add_argument("-bf16", "--use-bf16", action="store_true", default=False, help="Set to True if your setup supports BF16 dtype.")
    parser.add_argument("-bnb", "--use-adam8bit", action="store_true", default=False, help="Use ADAMW_8BIT optimizer (linux only).")
    parser.add_argument("-c", "--checkpoint", type=str, default=None, help="Path to checkpoint to resume training.")
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
