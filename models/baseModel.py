from abc import abstractmethod
from typing_extensions import override

import torch
from torch import nn, optim, utils, Tensor
import pytorch_lightning as pl
from lightning import LightningModule

from peft import PeftConfig, PeftMixedModel, PeftModel, get_peft_model
from transformers import AutoModel, AutoConfig, AutoProcessor, PreTrainedModel

from typing import Any, Dict, Optional, Union, cast


class TaskHead(nn.Module):
    def __init__(
        self,
        task_name:str,
        input_dim:int,
        output_dim:int,
        loss_function = nn.CrossEntropyLoss,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.linear = nn.Linear(input_dim, output_dim)
        self.loss_function = loss_function
        self.task_name = task_name

    def forward(self, data):
        return self.linear(data)

    def calculate_loss(self, y, target):
        return self.loss_function(y, target)





class BaseModel(LightningModule):
    def __init__(
        self,
        model_name:str,
        lanczos_r:int,
        use_peft:bool = False,
        peft_config:Optional[PeftConfig] = None,
    ):
        self.model_name = model_name
        self.lanczos_rank = lanczos_r
        self.use_peft = use_peft
        self.peft_config = peft_config

        self.model_config = AutoConfig.from_pretrained(self.model_name)
        self.backbone:PreTrainedModel = AutoModel.from_pretrained(self.model_name, config=self.model_config)
        self.processor = AutoProcessor.from_pretrained(self.model_name)

        self.representation_dim:int = -1
        if hasattr(self.backbone.config, "hidden_size"):
            self.representation_dim = self.backbone.config.hidden_size
        elif hasattr(self.backbone, "hidden_size"):
            self.representation_dim = cast(int, self.backbone.hidden_size)
        else:
            raise AttributeError("Cannot determine classification header")

        self.task_dict:Dict[str, TaskHead] = {}
        self.current_head:Optional[TaskHead]= None
        self.current_task:Optional[str] = None

        self.use_nostalgia = False

        self.peft_active = False
        self.peft_model:Optional[Union[PeftModel, PeftMixedModel]] = None
        self._apply_peft()




    def _apply_peft(self):
        if not self.use_peft:
            print("BaseModel.use_peft is set to False. Not configuring with PEFT")
        else:
            if self.peft_config is None:
                raise AttributeError("PEFT Configureation can't be None")
            self.peft_model = get_peft_model(self.backbone, self.peft_config)
            self.peft_active = True

    def _merge_and_unload(self):
        if not self.use_peft:
            raise AttributeError("PEFT not configured: Cannot merge and unload")

        assert self.peft_model is not None, "PEFT model is None"
        self.backbone = cast(PreTrainedModel, self.peft_model.merge_and_unload())# type: ignore
        self.peft_active = False

    def add_task(self, task_name:str, output_dim:int):
        self.task_dict[task_name] = TaskHead(task_name, self.representation_dim, output_dim)

    def set_current_task_head(self, task_name):
        if task_name not in self.task_dict.keys():
            raise ValueError("Task not found in task list")
        self.current_task = task_name
        self.current_head = self.task_dict[task_name]

    def _get_current_task_head(self):
        return self.current_task


    # ======= Configuring Nostalgia ===========

    def switch_on_nostalgia(self) -> None:
        self.use_nostalgia = True

    def switch_off_nostalgia(self) -> None:
        self.use_nostalgia = False

    def configure_optimizers(self):
        return super().configure_optimizers()

    def _nostalgic_optimizer(self):
        pass

    def _non_nostalgic_optimizer(self):
        pass


    # ======= To Be implemented =========

    def _forward_head(self, repr):
        assert self.current_head is not None, "No current head is choosen"
        return self.current_head(repr)

    @abstractmethod
    def _get_representations(self, x):
        pass

    @abstractmethod
    def training_step(self, batch, batch_idx, *args: Any, **kwargs: Any):
        pass

    @abstractmethod
    def test_step(self, *args: Any, **kwargs: Any):
        pass

    @abstractmethod
    def validation_step(self, *args: Any, **kwargs: Any):
        pass
