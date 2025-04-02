import logging
import torch
import torch.nn as nn
from tqdm import tqdm
from datasets import load_dataset

logger = logging.getLogger(__name__)

# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids


# Adapt from Atom
# https://github.com/efeslab/Atom/blob/main/model/datautils.py#L148
def get_eval_loaders(name, tokenizer, seqlen=2048):
    if "wikitext2" in name:
        testdata = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split="test",
        )
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
        return testenc
    if "ptb" in name:
        valdata = load_dataset(
            "ptb_text_only",
            "penn_treebank",
            split="validation",
        )
        testenc = tokenizer("\n\n".join(valdata["sentence"]), return_tensors="pt")
        return testenc
    if "c4" in name:
        valdata = load_dataset(
            "allenai/c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            revision="607bd4c8450a42878aa9ddc051a65a055450ef87",
            split="validation",
        )
        testenc = tokenizer("\n\n".join(valdata["text"]), return_tensors="pt")
        valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
        valenc = valenc.input_ids[:, :(256 * seqlen)]
        valenc = TokenizerWrapper(valenc)
        return valenc
    if "pile" in name:
        valdata = load_dataset(
            "monology/pile-uncopyrighted",
            data_files="val.jsonl.zst",
            split="train",
        )
        valdata.shuffle(seed=42)
        tokenizer.pad_token = tokenizer.eos_token
        valenc = tokenizer(' '.join(valdata[:1100]["text"]), return_tensors="pt")
        valenc = valenc.input_ids[:, :(256 * seqlen)]
        valenc = TokenizerWrapper(valenc)
        return valenc 
    raise NotImplementedError

@torch.no_grad()
def evaluate_ppl(
    model,
    tokenizer,
    model_name,
    batch_size=1,
    device="cuda",
    dataset = 'wikitext2'
):
    """
    model: model name
    limit: number of test samples for debug, set to -1 is no limit
    tasks: str tasks are split by ,
    num_fewshot: Number of examples in few-shot context
    eval_ppl: str datasets are split by , such as 'wikitext2,ptb,c4'
    """
    results = {}
    logger.info(f"Evaluating pereplexity on {dataset} dataset")
    testloader = get_eval_loaders(dataset, tokenizer)
   #torch.save(testloader, cache_testloader)
    model.eval()            
    ppl, _ = my_eval_ppl(model, testloader, bs=batch_size, device=device)
    logger.info(f"pereplexity on {dataset}: {ppl.item()}")
    results[dataset] = ppl.item()
    return results


# Function to evaluate perplexity (ppl)
@torch.no_grad()
def my_eval_ppl(model, testenc, bs=1, device=None, seq_len=2048, quick_eval_nsamples=-1):
    model.seqlen = seq_len
    
    # Get input IDs
    testenc = testenc.input_ids

    # Calculate number of samples
    total_nsamples = testenc.numel() // model.seqlen
    if quick_eval_nsamples == -1:
        nsamples = total_nsamples
    else:
        nsamples = min(quick_eval_nsamples, total_nsamples)
    
    # List to store negative log likelihoods
    nlls = []
    loss_lst = []
    logger.info(f"dataset size: {testenc.numel()}, model seqlen: {model.seqlen}, nsamples: {nsamples}")

    # Loop through each batch
    for i in tqdm(range(0,nsamples,bs)):
        # if i % 50 == 0:
        #     print(f"sample {i}")

        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)

        # Forward pass through the model
        lm_logits = model(inputs).logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seqlen * (j-i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)
        # loss_lst.append(loss.float())
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return ppl, torch.stack(nlls).sum() / (nsamples * model.seqlen)