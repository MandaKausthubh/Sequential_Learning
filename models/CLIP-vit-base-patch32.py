import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Tuple, Optional
from transformers import CLIPModel, AutoProcessor
from models.baseModel import BaseModel
from torch import Tensor


def contrastive_loss_from_embeddings(image_embeds: Tensor,
                                     text_embeds: Tensor,
                                     temperature: float = 0.07) -> Tensor:
    """
    Symmetric InfoNCE loss used for CLIP-style training.
    image_embeds: (B, D) L2-normalized or not (we will normalize)
    text_embeds : (B, D)
    Returns scalar loss (mean of image->text and text->image CE)
    """
    # normalize
    image_embeds = F.normalize(image_embeds, p=2, dim=-1)
    text_embeds = F.normalize(text_embeds, p=2, dim=-1)

    # logits: (B, B)
    logits_per_image = image_embeds @ text_embeds.t() / temperature
    logits_per_text = logits_per_image.t()

    targets = torch.arange(logits_per_image.size(0), device=logits_per_image.device)

    loss_i2t = F.cross_entropy(logits_per_image, targets)
    loss_t2i = F.cross_entropy(logits_per_text, targets)
    return 0.5 * (loss_i2t + loss_t2i)


class CLIPBaseModel(BaseModel):
    """
    CLIP model wrapper for contrastive training and classification heads.

    Two training modes (selected by current_head.task_name or a flag in head):
      - "contrastive": compute symmetric InfoNCE between image/text.
      - "classification": use a TaskHead (e.g., linear) that consumes either
            - text embedding, or
            - image embedding, or
            - concatenated multimodal embedding.
    """

    def __init__(self,
                 model_name: str = "openai/clip-vit-base-patch32",
                 lanczos_r: int = 16,
                 use_peft: bool = False,
                 peft_config=None,
                 user_nostalgia: bool = False,
                 temperature_init: float = 0.07):
        super().__init__(model_name=model_name,
                         lanczos_r=lanczos_r,
                         use_peft=use_peft,
                         peft_config=peft_config,
                         user_nostalgia=user_nostalgia)

        # Replace processor/backbone loaded by BaseModel with CLIP-specific ones
        # (BaseModel constructor already loaded AutoConfig/AutoModel/AutoProcessor for model_name,
        #  but for explicit clarity we reinitialize CLIPModel + AutoProcessor here.)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.clip = CLIPModel.from_pretrained(model_name)  # has vision_model + text_model

        # If using PEFT, you may wrap `self.clip` with get_peft_model externally.
        # We'll keep self.backbone for compatibility: point to clip for any code expecting backbone.
        self.backbone = self.clip

        # small learned temperature parameter (optional)
        self.temperature = nn.Parameter(torch.tensor(temperature_init))
        # for logging
        self.register_buffer("_dummy", torch.tensor(0.))  # example of buffer if needed

    # ----------------------------
    # Batch parsing
    # ----------------------------
    def _parse_batch(self, batch) -> Tuple[Dict[str, Any], Optional[torch.Tensor]]: #type: ignore
        """
        Accepts:
          - dict with "image" (PIL/tensor list), "text" (str list), optional "labels"
          - tuple/list (images, texts, labels optional)
          - HF-style already processed: "pixel_values", "input_ids", "attention_mask", "labels"
        Returns:
          inputs: dict with raw fields; will be processed in _get_representations
          labels: optional tensor (B,) or None
        """

        if isinstance(batch, dict):
            # Already processed by collate_fn? If so, it might contain "pixel_values" or "input_ids"
            if "pixel_values" in batch and "input_ids" in batch:
                labels = batch.get("labels", None)
                return batch, labels

            # Generic multimodal dataset item
            # expected keys: "image" (PIL or tensor or list), "text" (str or list), optionally "labels"
            imgs = batch.get("image") or batch.get("images")
            texts = batch.get("text") or batch.get("texts")
            labels = batch.get("label") or batch.get("labels", None)
            return {"image": imgs, "text": texts}, labels

        # tuple/list
        if isinstance(batch, (tuple, list)):
            if len(batch) == 2:
                images, texts = batch
                return {"image": images, "text": texts}, None
            if len(batch) == 3:
                images, texts, labels = batch
                return {"image": images, "text": texts}, labels

        raise ValueError(f"Unsupported batch format: {type(batch)}")

    # ----------------------------
    # Representations extractor
    # ----------------------------
    def _get_representations(self, inputs: Dict[str, Any]): #type: ignore
        """
        Inputs expected to have keys "image" and "text".
        We call the HF processor to tokenize and prepare tensors, move to device, then call CLIP.
        Returns a dict: {'image_embeds': (B, D), 'text_embeds': (B, D)}
        """
        images = inputs.get("image", None)
        texts = inputs.get("text", None)

        # Use the HF processor to prepare input tensors (it supports batches of PIL/tensors/strings)
        processed = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        # move to device
        processed = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in processed.items()}

        # Call CLIP (if PEFT active we would use self.peft_model)
        if self.peft_active and self.peft_model is not None:
            outputs = self.peft_model(**processed)  # expects same args as CLIPModel
        else:
            outputs = self.clip(**processed)

        # CLIPModel returns image_embeds and text_embeds (before/after projection depending on version).
        # We'll prefer `image_embeds` and `text_embeds` if present; else compute from pooled outputs.
        image_embeds = getattr(outputs, "image_embeds", None)
        text_embeds = getattr(outputs, "text_embeds", None)

        # Some HF versions return last_hidden_state + pooled outputs instead; fallback:
        if image_embeds is None:
            # take vision_model output and project if necessary
            image_embeds = outputs.vision_model_output.pooler_output if hasattr(outputs, "vision_model_output") else outputs.vision_model_output

        if text_embeds is None:
            # take text pooled
            text_embeds = outputs.text_model_output.pooler_output if hasattr(outputs, "text_model_output") else outputs.text_model_output

        # Ensure shapes are (B, D)
        return {"image_embeds": image_embeds, "text_embeds": text_embeds}

    # ----------------------------
    # Forward through head(s) and training step
    # ----------------------------
    def training_step(self, batch, batch_idx):
        # unify parsing as base class expects
        assert self.current_head is not None, "Set a current task head before training."

        inputs, labels = self._parse_batch(batch)
        reprs = self._get_representations(inputs)
        image_embeds = reprs["image_embeds"]
        text_embeds = reprs["text_embeds"]

        # default head name "contrastive" -> perform contrastive pretraining
        if getattr(self.current_head, "task_name", None) == "contrastive":
            # symmetric InfoNCE
            loss = contrastive_loss_from_embeddings(image_embeds, text_embeds, temperature=self.temperature.item())
            # log temperature too
            self.log("temperature", self.temperature.item(), prog_bar=False)
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            return loss

        # If classification head: decide which embedding to use or concat
        # For example, head can expect "multimodal" attribute to choose concat
        head_type = getattr(self.current_head, "head_type", "text")  # "text" | "image" | "multimodal"
        if head_type == "image":
            features = image_embeds
        elif head_type == "text":
            features = text_embeds
        else:
            # multimodal: concatenate along feature dim
            features = torch.cat([image_embeds, text_embeds], dim=-1)

        logits = self._forward_head(features)
        # if labels is None, we attempt to get labels from batch
        if labels is None:
            _, labels = self._parse_batch(batch)
        loss = self.current_head.calculate_loss(logits, labels)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    # ----------------------------
    # Validation/test steps (contrastive by default)
    # ----------------------------
    def validation_step(self, batch, batch_idx):

        assert self.current_head is not None, "Set a current task head before validation."

        inputs, labels = self._parse_batch(batch)
        reprs = self._get_representations(inputs)
        image_embeds = reprs["image_embeds"]
        text_embeds = reprs["text_embeds"]

        if getattr(self.current_head, "task_name", None) == "contrastive":
            loss = contrastive_loss_from_embeddings(image_embeds, text_embeds, temperature=self.temperature.item())

            # compute linear retrieval accuracy (image->text)
            with torch.no_grad():
                im = F.normalize(image_embeds, dim=-1)
                tx = F.normalize(text_embeds, dim=-1)
                sims = im @ tx.t()
                preds = sims.argmax(dim=1)
                acc = (preds == torch.arange(sims.size(0), device=sims.device)).float().mean()

            self.log("val_loss", loss, prog_bar=True)
            self.log("val_acc_retrieval", acc, prog_bar=True)
            return loss

        # classification-style
        head_type = getattr(self.current_head, "head_type", "text")
        if head_type == "image":
            features = image_embeds
        elif head_type == "text":
            features = text_embeds
        else:
            features = torch.cat([image_embeds, text_embeds], dim=-1)

        logits = self._forward_head(features)
        loss = self.current_head.calculate_loss(logits, labels)

        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)  # reuse logic
