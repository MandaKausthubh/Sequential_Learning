from abc import abstractmethod

from torch import nn, Tensor
from lightning import LightningModule

from peft import PeftConfig, PeftMixedModel, PeftModel, get_peft_model
import torch
from transformers import AutoModel, AutoConfig, AutoProcessor, PreTrainedModel
from typing import Dict, Optional, Tuple, Union, cast

from utils.nostalgia import NostalgiaOptimizer


class TaskHead(nn.Module):
    def __init__(self, task_name: str, input_dim: int, output_dim: int,
                 loss_function = nn.CrossEntropyLoss()):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.loss_function = loss_function
        self.task_name = task_name

    def forward(self, data):
        return self.linear(data)

    def calculate_loss(self, y, target) -> Tensor:
        return self.loss_function(y, target)


class BaseModel(LightningModule):
    def __init__(
        self,
        model_name:str,
        lanczos_r:int = 16,
        use_peft:bool = False,
        peft_config:Optional[PeftConfig] = None,
        user_nostalgia:bool = False,
    ):
        super().__init__()
        self.model_name = model_name
        self.lanczos_rank = lanczos_r
        self.use_peft = use_peft
        self.peft_config = peft_config

        if self.use_peft:
            self.automatic_optimization = False

        self.model_config = AutoConfig.from_pretrained(self.model_name)
        self.backbone:PreTrainedModel = AutoModel.from_pretrained(self.model_name, config=self.model_config)
        self.processor = AutoProcessor.from_pretrained(self.model_name)

        # determine hidden size
        if hasattr(self.backbone.config, "hidden_size"):
            self.representation_dim = self.backbone.config.hidden_size
        elif hasattr(self.backbone, "hidden_size"):
            self.representation_dim = cast(int, self.backbone.hidden_size)
        else:
            raise AttributeError("Cannot determine classification header")

        self.task_dict:Dict[str, TaskHead] = {}
        self.current_head:Optional[TaskHead]= None
        self.current_task:Optional[str] = None

        self.use_nostalgia:bool = user_nostalgia
        self.peft_active:bool = False
        self.peft_model:Optional[Union[PeftModel, PeftMixedModel]] = None
        self._apply_peft()

        # ★ Nostalgia storage for Q and scaling
        self.nostalgia_Q: Optional[torch.Tensor] = None
        self.nostalgia_scaling: Optional[torch.Tensor] = None


    # ------------------ PEFT ------------------
    def _apply_peft(self):
        if not self.use_peft:
            print("BaseModel.use_peft is False. Using backbone without PEFT.")
            return

        if self.peft_config is None:
            raise AttributeError("peft_config cannot be None when use_peft=True")

        self.peft_model = get_peft_model(self.backbone, self.peft_config)
        self.peft_active = True

    def _merge_and_unload(self):
        if not self.use_peft:
            raise AttributeError("PEFT not configured: Cannot merge and unload")

        assert self.peft_model is not None, "PEFT model is None"
        self.backbone = cast(PreTrainedModel, self.peft_model.merge_and_unload()) #type: ignore
        self.peft_active = False


    # ---------------- Tasks -------------------
    def add_task(self, task_name:str, output_dim:int):
        self.task_dict[task_name] = TaskHead(task_name, self.representation_dim, output_dim)

    def set_current_task_head(self, task_name):
        if task_name not in self.task_dict:
            raise ValueError("Task not found")
        self.current_task = task_name
        self.current_head = self.task_dict[task_name]
        return self.current_head


    # ------------- Nostalgia switches ---------------
    def switch_on_nostalgia(self):  self.use_nostalgia = True
    def switch_off_nostalgia(self): self.use_nostalgia = False

    # ============= OPTIMIZER INTEGRATION ==============
    def configure_optimizers(self): # type: ignore
        """
        Correct optimizer setup for:
          - PEFT or full backbone fine-tuning
          - Task-specific heads
          - Nostalgia gradient projection (backbone only)
        """

        if self.use_peft:
            assert self.peft_model is not None
            backbone_params = [p for p in self.peft_model.parameters() if p.requires_grad]
        else:
            backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]

        head_params = []
        if self.current_head is not None:
            head_params = [p for p in self.current_head.parameters() if p.requires_grad]

        base_opt = torch.optim.AdamW(backbone_params + head_params, lr=1e-4, weight_decay=1e-2)

        if not self.use_nostalgia:
            return {"optimizer": base_opt}


        nostalgia_opt = NostalgiaOptimizer(
            base_optimizer=base_opt,
            params=backbone_params,
            device=self.device,
            dtype=torch.float32,
        )

        if self.nostalgia_Q is not None:
            nostalgia_opt.set_Q(
                self.nostalgia_Q,
                scaling=self.nostalgia_scaling,
            )
        return {"optimizer": nostalgia_opt}


    # Helper: choose which params to optimize
    def _get_trainable_params(self):
        # Backbone or PEFT parameters
        if self.use_peft:
            assert self.peft_model is not None
            backbone_params = [p for p in self.peft_model.parameters() if p.requires_grad]
        else:
            backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]

        # Current task head parameters
        head_params = []
        if self.current_head is not None:
            head_params = [p for p in self.current_head.parameters() if p.requires_grad]

        return backbone_params + head_params


    # ---------------- ABSTRACT INTERFACES ----------------
    def _forward_head(self, repr):
        assert self.current_head is not None
        return self.current_head(repr)

    @abstractmethod
    def _get_representations(self, inputs): pass

    @abstractmethod
    def _parse_batch(self, batch) -> Tuple[Tensor, Tensor]: pass

    def training_step(self, batch, batch_idx):
        """
        One training iteration:
        1. Extract inputs and labels
        2. Compute backbone representations
        3. Forward through the current task-specific head
        4. Compute loss
        5. Log and return loss
        """
        assert self.current_head is not None, (
            "Current task head not set. "
            "Call `set_current_task_head(task_name)` before training."
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

    @abstractmethod
    def validation_step(self, batch, batch_idx) -> Tensor: pass

    @abstractmethod
    def test_step(self, batch, batch_idx) -> Tensor: pass
