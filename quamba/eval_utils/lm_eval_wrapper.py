# Import necessary modules
import os
import time
import logging
import torch
import transformers

import lm_eval
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
        self.is_hf = False
        # self.is_hf = is_hf or (True if pretrained.endswith("hf") else False)
        self._device = torch.device(device)
    
    @property
    def batch_size(self):
        return self._batch_size

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        remove_arg = (
            ["attention_mask"] if self.is_hf else ["do_sample", "attention_mask"]
        )
        for key in remove_arg:
            if key in generation_kwargs:
                generation_kwargs.pop(key)

        # mamba's custom GenerationMixin currently does not support
        # passing stopping criteria.
        # for the time being, we simply generate to max length,
        # then truncate (equivalent result)
        # -- this should be revisited to speed up generation
        # stopping_criteria = stop_sequences_criteria(
        #     self.tokenizer, stop, 1, context.shape[0]
        # )

        if not self.is_hf:
            return self.model.generate(
                input_ids=context,
                max_length=max_length,
                # stopping_criteria=stopping_criteria,
                # pad_token_id=self.tokenizer.pad_token_id,
                # use_cache=True,
                **generation_kwargs,
            )
        else:
            stopping_criteria = lm_eval.models.utils.stop_sequences_criteria(
                self.tokenizer,
                stop,
                context.shape[1],
                context.shape[0],
            )

            generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
            do_sample = generation_kwargs.get("do_sample", None)

            # The temperature has to be a strictly positive float -- if it is 0.0, use greedy decoding strategies
            if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
                generation_kwargs["do_sample"] = do_sample = False
            if do_sample is False and generation_kwargs.get("temperature") == 0.0:
                generation_kwargs.pop("temperature")

            return self.model.generate(
                input_ids=context,
                max_length=max_length,
                stopping_criteria=stopping_criteria,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
                **generation_kwargs,
            )


def eval_mamba_few_shot(model, tokenizer, model_type, batch_size=1, max_length=2048, task_list=["lambada_openai"], fewshot=0, limit=None):
    # Workaround for the following error
    # huggingface/tokenizers: The current process just got forked, 
    # after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    if model_type == "mamba" or model_type == "mamba2" or model_type == "quamba" or model_type == "quamba2":
        lm_obj = MambaEvalWrapper(model=model, tokenizer=tokenizer, max_length=max_length, batch_size=batch_size)
    else:
        raise ValueError(f"Unsupported model type: {model_type}, only support 'mamba', 'mamba2', 'quamba' and 'quamba2'")
    # indexes all tasks from the `lm_eval/tasks` subdirectory.
    # Alternatively, you can set `TaskManager(include_path="path/to/my/custom/task/configs")`
    # to include a set of tasks in a separate directory.
    task_manager = lm_eval.tasks.TaskManager()

    # Setting `task_manager` to the one above is optional and should generally be done
    # if you want to include tasks from paths other than ones in `lm_eval/tasks`.
    # `simple_evaluate` will instantiate its own task_manager is the it is set to None here.
    results = lm_eval.simple_evaluate( # call simple_evaluate
        model=lm_obj,
        model_args= "",
        tasks=task_list,
        num_fewshot=fewshot,
        task_manager=task_manager,
        log_samples=False,
        limit=limit
    ) 

    res = make_table(results)
    logger.info(f"{fewshot}-shot evaluation results: \n{res}")
    
    return results


def eval_mamba_generation(model, tokenizer, model_type, batch_size=1, max_length=2048, task_list=["lambada_openai"], fewshot=0, limit=None):
    # Workaround for the following error
    # huggingface/tokenizers: The current process just got forked, 
    # after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    if model_type == "mamba" or model_type == "mamba2" or model_type == "quamba" or model_type == "quamba2":
        lm_obj = MambaEvalWrapper(model=model, tokenizer=tokenizer, max_length=max_length, batch_size=batch_size)
    else:
        raise ValueError(f"Unsupported model type: {model_type}, only support 'mamba', 'mamba2', 'quamba' and 'quamba2'")
    # indexes all tasks from the `lm_eval/tasks` subdirectory.
    # Alternatively, you can set `TaskManager(include_path="path/to/my/custom/task/configs")`
    # to include a set of tasks in a separate directory.
    task_manager = lm_eval.tasks.TaskManager()

    # Setting `task_manager` to the one above is optional and should generally be done
    # if you want to include tasks from paths other than ones in `lm_eval/tasks`.
    # `simple_evaluate` will instantiate its own task_manager is the it is set to None here.
    results = lm_eval.simple_evaluate( # call simple_evaluate
        model=lm_obj,
        model_args= "",
        tasks=task_list,
        num_fewshot=fewshot,
        task_manager=task_manager,
        log_samples=False,
        limit=limit
    ) 

    res = make_table(results)
    logger.info(f"generation evaluation results: \n{res}")
    
    return results