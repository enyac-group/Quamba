# Import necessary modules
import time
import logging
import torch
import transformers

from lm_eval.api.model import LM
from lm_eval.models.huggingface import HFLM
from lm_eval.api.registry import register_model

from lm_eval.utils import make_table

logger = logging.getLogger(__name__)

@register_model("mamba")
class MambaEvalWrapper(HFLM):

    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    def __init__(self, model, tokenizer, max_length=2048, batch_size=None, device="cuda"):
        super().__init__()
        LM.__init__(self)
        self._model = model
        self.tokenizer = tokenizer
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.vocab_size = self.tokenizer.vocab_size
        self._batch_size = int(batch_size) if batch_size is not None else 64
        self._max_length = max_length
        self._device = torch.device(device)
    @property
    def batch_size(self):
        return self._batch_size

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        raise NotImplementedError()


def eval_mamba_zero_shot(model, tokenizer, model_type, batch_size=1, max_length=2048, task_list=["lambada_openai"], fewshot=0, limit=None):
    import lm_eval
    import os
    # Workaround for the following error
    # huggingface/tokenizers: The current process just got forked, 
    # after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    if model_type == "jamba":
        lm_obj = HFLM(pretrained=model, tokenizer=tokenizer, add_bos_token=True)
    elif model_type == "mamba":
        lm_obj = MambaEvalWrapper(model=model, tokenizer=tokenizer, max_length=max_length, batch_size=batch_size)
    else:
        raise ValueError(f"Unsupported model type: {model_type}, only support 'mamba' and 'jamba'")
    # indexes all tasks from the `lm_eval/tasks` subdirectory.
    # Alternatively, you can set `TaskManager(include_path="path/to/my/custom/task/configs")`
    # to include a set of tasks in a separate directory.
    task_manager = lm_eval.tasks.TaskManager()

    # Setting `task_manager` to the one above is optional and should generally be done
    # if you want to include tasks from paths other than ones in `lm_eval/tasks`.
    # `simple_evaluate` will instantiate its own task_manager is the it is set to None here.
    results = lm_eval.simple_evaluate( # call simple_evaluate
        model=lm_obj,
        model_args= "add_bos_token=True" if model_type == "jamba" else "",
        tasks=task_list,
        num_fewshot=fewshot,
        task_manager=task_manager,
        log_samples=False,
        limit=limit
    ) 

    res = make_table(results)
    logger.info(f"{fewshot}-shot evaluation results: \n{res}")
    
    return results