import torch
from typing import Any, Tuple
from models.baseModel import BaseModel


class ViTBaseModel(BaseModel):
    """
    Base class for google/vit-base-patch16-224
    Handles:
        - parsing vision batches
        - HF processor for pixel normalization + resizing
        - extracting CLS embeddings from ViT backbone
        - continual-learning with task heads
        - optional PEFT and Nostalgia optimizer
    """

    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        lanczos_r: int = 16,
        use_peft: bool = False,
        peft_config=None,
        user_nostalgia: bool = False,
    ):
        super().__init__(
            model_name=model_name,
            lanczos_r=lanczos_r,
            use_peft=use_peft,
            peft_config=peft_config,
            user_nostalgia=user_nostalgia,
        )
    def _parse_batch(self, batch) -> Tuple[Any, torch.Tensor]:
        """
        Input batch may be:
            - (images, labels)
            - {"pixel_values": ..., "labels": ...}
            - {"image": PIL.Image, "label": int}
        Outputs:
            inputs: raw images (list, tensor, PIL, etc.)
            labels: tensor of shape (B,)
        """
        if isinstance(batch, dict):
            # Try common HF-style keys first
            if "pixel_values" in batch and "labels" in batch:
                return batch["pixel_values"], batch["labels"]

            # Custom dataset format with PIL images
            if "image" in batch and "label" in batch:
                return batch["image"], batch["label"]

            raise ValueError(f"Unknown dict batch format: {batch.keys()}")

        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            return batch[0], batch[1]

        raise ValueError(f"Unrecognized batch format: {type(batch)}")

    def _get_representations(self, inputs):
        """
        Inputs: a tensor, a list of PIL images, or raw numpy arrays.
        Processor handles:
            - resizing to 224x224
            - normalization
            - batching
            - device transfer
        Backbone outputs:
            - CLS token embedding: shape (B, hidden_dim)
        """
        processed = self.processor(
            images=inputs,
            return_tensors="pt"
        ).to(self.device)

        if self.peft_active:
            assert self.peft_model is not None, "PEFT model is None, while peft_active is True"
            outputs = self.peft_model(**processed)
        else:
            outputs = self.backbone(**processed)

        cls_repr = outputs.last_hidden_state[:, 0]
        return cls_repr

    def training_step(self, batch, batch_idx):
        assert self.current_head is not None, (
            "Current task head is not set. "
            "Call set_current_task_head(task_name) before training."
        )

        inputs, labels = self._parse_batch(batch)
        repr = self._get_representations(inputs)
        logits = self._forward_head(repr)

        loss = self.current_head.calculate_loss(logits, labels)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        assert self.current_head is not None, (
            "Current task head is not set. "
            "Call set_current_task_head(task_name) before training."
        )
        inputs, labels = self._parse_batch(batch)
        repr = self._get_representations(inputs)
        logits = self._forward_head(repr)
        loss = self.current_head.calculate_loss(logits, labels)

        # compute accuracy
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

        return loss

    # -------------------------------------------------------------
    # 5. Test Step
    # -------------------------------------------------------------
    def test_step(self, batch, batch_idx):

        assert self.current_head is not None, (
            "Current task head is not set. "
            "Call set_current_task_head(task_name) before training."
        )
        inputs, labels = self._parse_batch(batch)
        repr = self._get_representations(inputs)
        logits = self._forward_head(repr)
        loss = self.current_head.calculate_loss(logits, labels)

        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean()

        self.log("test_loss", loss, prog_bar=False)
        self.log("test_acc", acc, prog_bar=False)

        return loss
