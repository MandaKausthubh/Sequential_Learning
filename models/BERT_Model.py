import torch
from typing import Any, Dict, Tuple
from models.baseModel import BaseModel   # import your BaseModel


class BERTBaseModel(BaseModel):
    """
    BERT-based continual-learning model.
    Supports:
        - text classification / regression / QA heads
        - PEFT (LoRA) on BERT
        - Nostalgia optimizer on backbone parameters only
        - multi-task continual learning
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
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

    # -------------------------------------------------------------
    # 1. Parse NLP Batch
    # -------------------------------------------------------------
    def _parse_batch(self, batch) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]: #type: ignore
        """
        Expected batch formats:
            - (text_list, labels)
            - {"input_ids":..., "attention_mask":..., "labels":...}
            - {"text":..., "labels":...}
        Output:
            inputs: dict of model-ready tensors
            labels: tensor (B,)
        """

        # HF-style batch
        if isinstance(batch, dict):
            # Already numericalized (tokenized) → simply return
            if "input_ids" in batch and "attention_mask" in batch:
                labels = batch.get("labels")
                inputs = {
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch["attention_mask"]
                }
                # Optional token_type_ids
                if "token_type_ids" in batch:
                    inputs["token_type_ids"] = batch["token_type_ids"]
                if labels is None:
                    raise ValueError("Labels missing in batch dict")
                return inputs, labels

            # Raw text → need to tokenize
            if "text" in batch and "labels" in batch:
                texts = batch["text"]
                labels = batch["labels"]

                encoded = self.processor(
                    texts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                )
                return encoded.to(self.device), labels.to(self.device)

        # (texts, labels) tuple
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            texts, labels = batch
            encoded = self.processor(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            return encoded.to(self.device), labels.to(self.device)

        raise ValueError(f"Unrecognized NLP batch format: {batch}")

    # -------------------------------------------------------------
    # 2. Extract BERT Representations
    # -------------------------------------------------------------
    def _get_representations(self, inputs: Dict[str, torch.Tensor]):
        """
        BERT outputs:
            outputs.last_hidden_state   → (B, L, hidden)
            outputs.pooler_output       → (B, hidden)
        CLS representation = last_hidden_state[:, 0]
        """
        if self.peft_active:
            assert self.peft_model is not None, "PEFT model is None, while peft_active is True"
            outputs = self.peft_model(**inputs)
        else:
            outputs = self.backbone(**inputs)

        # CLS embedding for classification-style tasks
        cls_repr = outputs.last_hidden_state[:, 0]    # (B, hidden_size)

        # If future tasks need token-level or full sequence outputs,
        # return a dict. But for now, simple:
        return cls_repr

    # -------------------------------------------------------------
    # 3. Training Step (inherits from BaseModel)
    # -------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        assert self.current_head is not None, (
            "Call set_current_task_head(task_name) before training!"
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

    # -------------------------------------------------------------
    # 4. Validation Step
    # -------------------------------------------------------------
    def validation_step(self, batch, batch_idx):
        assert self.current_head is not None, (
            "Call set_current_task_head(task_name) before validation!"
        )
        inputs, labels = self._parse_batch(batch)
        repr = self._get_representations(inputs)
        logits = self._forward_head(repr)
        loss = self.current_head.calculate_loss(logits, labels)

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
            "Call set_current_task_head(task_name) before validation!"
        )
        inputs, labels = self._parse_batch(batch)
        repr = self._get_representations(inputs)
        logits = self._forward_head(repr)
        loss = self.current_head.calculate_loss(logits, labels)

        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean()

        self.log("test_loss", loss)
        self.log("test_acc", acc)

        return loss
