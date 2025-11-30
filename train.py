import json
from collections import defaultdict
import gc
from datetime import timedelta
from tqdm import tqdm
from functools import partial
import argparse
from typing import Callable, List, Tuple, Dict, Any, Iterator

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from transformers import (
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    GPT2LMHeadModel,
    GPT2Config,
)
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import ProjectConfiguration, set_seed

from data_utils import (
    EvalDataset,
    GSDataset,
    collate_fn_eval,
    collate_fn_gs,
)


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["NCCL_TIMEOUT"] = "28800" # 4hr for evaluation time variance across gpus
os.environ["NCCL_TIMEOUT_MS"] = "28800000"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["NCCL_BLOCKING_WAIT"] = "1"
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"


def get_individual_loss(lm_logits: torch.Tensor, label_ids: torch.Tensor) -> torch.Tensor:
    # move labels to correct device to enable model parallelism
    labels = label_ids.to(lm_logits.device)

    # shift so that tokens < n predict n
    losses = []
    for logs, labs in zip(lm_logits, labels):
        shift_logits = logs[:-1, :].contiguous()
        shift_labels = labs[1:].contiguous()
        # flatten the tokens
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss[labs[1:] != -100].mean()
        losses.append(loss)

    return torch.stack(losses)


def compute_macrof1_or_accuracy(predictions, groundtruths, is_classification) -> float:
    # accuracy measurement to use the same evaluation setup as MetaICL
    accs = []
    precisions = defaultdict(list)
    recalls = defaultdict(list)
    for prediction, groundtruth in zip(predictions, groundtruths):
        prediction = prediction.strip()
        groundtruth = groundtruth.strip()
        is_correct = prediction==groundtruth
        accs.append(is_correct)
        if is_classification:
            recalls[groundtruth].append(is_correct)
            precisions[prediction].append(is_correct)

    if not is_classification:
        return float(np.mean(accs))

    f1s = []
    for key in recalls:
        precision = np.mean(precisions[key]) if key in precisions else 1.0
        recall = np.mean(recalls[key])
        if precision+recall==0:
            f1s.append(0)
        else:
            f1s.append(2*precision*recall / (precision+recall))

    return float(np.mean(f1s))


def chunks(lst: List[Any], n: int) -> Iterator[List[Any]]:
    # iterator for batched items from list
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


@torch.no_grad()
def inference_time_optimization(
    model: nn.Module,
    dataset: EvalDataset,
    accelerator: Accelerator,
    batch_size: int,
    collate_fn: Callable,
    log_every: int,
    zero_shot: bool,
    epochs: int,
    ct_batch_size: int,
    lr: float,
    token_dropout: float,
) -> Tuple[float, float, float, float, float, float, float, float, List]:

    model.eval()

    embed_tokens = model.transformer.wte
    model.generation_config.pad_token_id = dataset.tokenizer.pad_token_id

    # outputs
    output_list = []

    for task in dataset.tasks:
        print(f"processing {task}...")
        model.eval()

        # HACK: get demonstration ids and indices
        demon_input_ids = None
        demon_start_idxs = None
        for data in dataset.data:
            if data['task'] == task:
                demon_input_ids = data["demon_input_ids"].unsqueeze(0).to(accelerator.device)
                demon_start_idxs = data["demon_start_idxs"]
                break

        # initialize kv
        with accelerator.autocast():
            past_key_values = model(input_ids=demon_input_ids, output_hidden_states=True).past_key_values
        torch.cuda.empty_cache()
        gc.collect()

        # use ctkv to refine kv
        if epochs > 0:
            with accelerator.no_sync(model):
                past_key_values = tuple(
                    (layer_k.to(torch.bfloat16), layer_v.to(torch.bfloat16))
                    for layer_k, layer_v in past_key_values
                )
                model, past_key_values = context_tuning(
                    demonstration_pairs=dataset.task_to_demonstrations[task],
                    eval_dataset=dataset,
                    accelerator=accelerator,
                    model=model,
                    demon_start_idxs=demon_start_idxs,
                    past_key_values=past_key_values,
                    demon_input_ids_len=demon_input_ids.shape[1],
                    epochs=epochs,
                    lr=lr,
                    batch_size=ct_batch_size,
                    token_dropout=token_dropout,
                )
                torch.cuda.empty_cache()
                gc.collect()

        process_data_idxs = [i for i, d in enumerate(dataset.data) if d['task'] == task]
        data_idxs = [idxs for idxs in chunks(process_data_idxs, batch_size)]
        progress_bar = tqdm(
            range(len(data_idxs)),
            desc=f"evaluating on {task}...",
            disable=not accelerator.is_local_main_process,
        )
        for eval_step, batch_idxs in enumerate(data_idxs):
            batch_data = [dataset[i] for i in batch_idxs]
            bs = len(batch_data)
            batch = collate_fn(batch_data)

            # get tensors
            task = batch['task']
            test_idx = batch['test_idx']
            option = batch['option']
            correct_option = batch['correct_option']
            gen_input_ids = batch["gen_input_ids"].to(accelerator.device)
            gen_attention_mask = batch["gen_attention_mask"].to(accelerator.device)
            gen_label_ids = batch["gen_label_ids"].to(accelerator.device)

            with accelerator.autocast():
                # expand past key values
                batch_past_key_values = [
                    (
                        layer_k.detach().clone().expand(bs, *layer_k.shape[1:]),
                        layer_v.detach().clone().expand(bs, *layer_v.shape[1:]),
                    )
                    for layer_k, layer_v in past_key_values
                ]
                batch_past_key_values_attention_mask = torch.ones(
                    (bs, batch_past_key_values[0][0].shape[2]),
                    device=accelerator.device,
                    dtype=torch.int64
                )

                gen_inputs_embeds = embed_tokens(gen_input_ids)
                gen_attention_mask = torch.cat([batch_past_key_values_attention_mask, gen_attention_mask], dim=1)

                # build position ids (does NOT depend on dropout)
                attention_mask_after_kv = gen_attention_mask[:, batch_past_key_values[0][0].shape[2]:]
                position_ids = []
                position_start = demon_input_ids.shape[1]
                for mask_after_kv in attention_mask_after_kv:
                    sequence_position_ids = torch.zeros(gen_inputs_embeds.shape[1], device=accelerator.device, dtype=torch.int64)
                    n_new_positions = mask_after_kv.sum()
                    new_positions = torch.tensor(range(position_start, position_start + n_new_positions), device=accelerator.device, dtype=torch.int64)
                    sequence_position_ids[:n_new_positions] = new_positions
                    position_ids.append(sequence_position_ids)
                position_ids = torch.stack(position_ids)

                model_out = model(
                    inputs_embeds=gen_inputs_embeds,
                    attention_mask=gen_attention_mask if not zero_shot else attention_mask_after_kv,
                    past_key_values=batch_past_key_values if not zero_shot else None,
                    position_ids=position_ids if not zero_shot else None,
                )
                losses = get_individual_loss(lm_logits=model_out.logits.half(), label_ids=gen_label_ids)

            for x0, x1, x2, x3, x4 in zip(losses, task, test_idx, option, correct_option):
                output_list.append((x0.item(), x1, x2, x3, x4))
            if (eval_step + 1) % log_every == 0:
                progress_bar.update(log_every)

            torch.cuda.empty_cache()
            gc.collect()

    # determine which tasks are classification (for macro-f1)
    task_to_is_clf = {}
    for task in dataset.tasks:
        meta_data_path = os.path.join('config/tasks', f'{task}.json')
        task_meta_data = json.load(open(meta_data_path, 'r'))
        task_to_is_clf[task] = task_meta_data['task_type'] == "classification"

    # metrics
    task_to_score = {}
    for task in dataset.tasks:
        task_outs = [x for x in output_list if x[1] == task]
        if len(task_outs) == 0:
            continue

        preds, gts = [], []
        test_idxs = set(x[2] for x in task_outs)
        for test_i in test_idxs:
            task_test_outs = [x for x in task_outs if x[2] == test_i]
            correct_option = task_test_outs[0][4]
            # choose option with lowest loss
            lowest_loss = float('inf')
            chosen_option = None
            for x in task_test_outs:
                if x[0] < lowest_loss:
                    lowest_loss = x[0]
                    chosen_option = x[3]
            # record
            preds.append(chosen_option)
            gts.append(correct_option)

        task_to_score[task] = compute_macrof1_or_accuracy(preds, gts, task_to_is_clf[task])

    # display scores
    sorted_tasks = sorted(task_to_score.keys())
    for task in sorted_tasks:
        print(f"{task} clf {task_to_is_clf[task]} has a score {task_to_score[task]}")
    score = sum(v for v in task_to_score.values()) / len(task_to_score)
    print(f'average score: {score}')


@torch.enable_grad()
def context_tuning(
    demonstration_pairs: List[Dict],
    eval_dataset: EvalDataset,
    accelerator: Accelerator,
    model: nn.Module,
    # inputs
    demon_start_idxs: List[int],
    past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor]],
    demon_input_ids_len: int,
    # config
    epochs: int,
    lr: float,
    batch_size: int,
    token_dropout: float,
):
    # may not be necessary, just to be safe
    past_key_values = tuple(
        (layer_k.detach().clone(), layer_v.detach().clone())
        for layer_k, layer_v in past_key_values
    )

    # get program parameters
    program_params = []

    # full tuning of initialized KV
    for layer_k, layer_v in past_key_values:
        program_params.append(layer_k)
        program_params.append(layer_v)

    # dataset
    gs_dataset = GSDataset(
        demonstration_pairs={i: p for i, p in enumerate(demonstration_pairs)},
        tokenizer=eval_dataset.tokenizer,
        pad_side='right',
        past_kv_len=demon_input_ids_len,
    )

    # dataloader
    batch_size = min(batch_size, len(gs_dataset))
    gs_collate_fn = partial(collate_fn_gs, dataset=gs_dataset)
    gs_loader = DataLoader(
        gs_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=gs_collate_fn,
        drop_last=False, # full batch
        num_workers=0,
    )

    # set requires grad
    for p in program_params:
        p.requires_grad = True

    # optimizer
    optimizer_grouped_params = [{"params": program_params, "lr": lr}]
    all_params = program_params
    optim = torch.optim.AdamW(optimizer_grouped_params, weight_decay=0.0)
    optim.zero_grad()

    # lr scheduler
    scheduler = get_cosine_schedule_with_warmup(optim, num_warmup_steps=0, num_training_steps=epochs)

    # prepare some stuff
    model.train()

    module = model
    embed_tokens = module.transformer.wte

    # train!
    for _ in range(epochs):
        for batch in gs_loader:
            pair_input_ids = batch["input_ids"].to(accelerator.device)
            pair_attention_mask = batch["attention_mask"].to(accelerator.device)
            pair_label_ids = batch["label_ids"].to(accelerator.device)
            pair_example_idx = batch["example_idx"]
            device, dtype = pair_input_ids.device, pair_input_ids.dtype
            bs = pair_input_ids.shape[0]

            batch_past_key_values_attention_mask = torch.ones((bs, past_key_values[0][0].shape[2]), device=accelerator.device, dtype=torch.int64)
            batch_past_key_values = tuple(
                (layer_k.expand(bs, -1, -1, -1), layer_v.expand(bs, -1, -1, -1)) # expand here because no modifications
                for layer_k, layer_v in past_key_values
            )

            # leave one out
            for batch_i, idx in enumerate(pair_example_idx):
                start = demon_start_idxs[idx]
                end = demon_start_idxs[idx + 1] if idx < len(demon_start_idxs) - 1 else demon_input_ids_len
                batch_past_key_values_attention_mask[batch_i, start:end] = 0

            # token dropout
            if token_dropout != 0.0:
                drop_mask = (torch.rand_like(batch_past_key_values_attention_mask, dtype=torch.float) > token_dropout).float()
                batch_past_key_values_attention_mask = (batch_past_key_values_attention_mask * drop_mask).long()

            with accelerator.autocast():
                position_ids = torch.zeros((bs, pair_input_ids.shape[1]), device=device, dtype=torch.int64)
                new_lens = pair_attention_mask.sum(dim=1)
                for task_position_ids, new_len in zip(position_ids, new_lens):
                    new_positions = torch.tensor(range(demon_input_ids_len, demon_input_ids_len + new_len), device=device, dtype=dtype)
                    task_position_ids[:new_len] = new_positions

                pair_inputs_embeds = embed_tokens(pair_input_ids)
                pair_attention_mask = torch.cat([batch_past_key_values_attention_mask, pair_attention_mask], dim=1)
                model_kwargs = {
                    "inputs_embeds": pair_inputs_embeds,
                    "attention_mask": pair_attention_mask,
                    "labels": pair_label_ids,
                    "use_cache": True,
                    "past_key_values": batch_past_key_values,
                    "position_ids": position_ids,
                }

                # get loss
                model_out = model(**model_kwargs)
                loss = model_out.loss * bs / batch_size

            accelerator.backward(loss)

        accelerator.clip_grad_norm_(all_params, 1.0)
        optim.step()
        scheduler.step()
        optim.zero_grad()

    model.eval()

    # may not be necessary, just to be safe
    past_key_values = tuple(
        (layer_k.detach().clone(), layer_v.detach().clone())
        for layer_k, layer_v in past_key_values
    )

    return model, past_key_values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config_file", type=str, default="./hr_to_lr.json")
    parser.add_argument("--data_dir", type=str, default="./metaicl-data/data")
    parser.add_argument("--num_demonstrations", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument('--eval_split', type=str, default='87') # main table averages over 4 other splits: 13, 21, 42, 100
    parser.add_argument('--eval_ratio', type=float, default=1.0)
    parser.add_argument('--zero_shot', action='store_true')
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ct_batch_size", type=int, default=16) # ct uses less memory, full batch
    parser.add_argument("--token_dropout", type=float, default=0.05)
    args = parser.parse_args()

    # Setup accelerator
    project_config = ProjectConfiguration(project_dir=args.experiment_name)
    init_process_process_kwargs = InitProcessGroupKwargs()
    init_process_process_kwargs.timeout = timedelta(seconds=28800)
    accelerator = Accelerator(
        project_config=project_config,
        kwargs_handlers=[init_process_process_kwargs],
    )
    set_seed(args.seed + accelerator.process_index)
    torch.backends.cuda.matmul.allow_tf32 = True

    # log args
    print("#### BEGIN ALL ARGUMENTS ####")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("#### END ALL ARGUMENTS ####\n")

    # Load tokenizers
    tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2-large', cache_dir='./.cache')
    tokenizer.pad_token = tokenizer.eos_token

    # load model, deterministic to lower inference variance
    config = GPT2Config.from_pretrained(
        'openai-community/gpt2-large',
        attn_pdrop=0.0,
        embd_pdrop=0.0,
        resid_pdrop=0.0,
        summary_first_dropout=0.0,
        _attn_implementation='sdpa',
    )
    model = GPT2LMHeadModel.from_pretrained(
        'openai-community/gpt2-large',
        config=config,
        torch_dtype=torch.bfloat16,
        cache_dir="./.cache",
    )

    # Prepare with accelerator
    model = accelerator.prepare(model)

    # dont train model
    for p in model.parameters():
        p.requires_grad = False

    # Build evaluation dataset
    dataset = EvalDataset(
        data_dir=args.data_dir,
        config_file=args.config_file,
        seed=args.seed,
        eval_split=args.eval_split,
        tokenizer=tokenizer,
        pad_side='right',
        eval_ratio=args.eval_ratio,
        num_demonstrations=args.num_demonstrations,
    )
    collate_fn = partial(collate_fn_eval, dataset=dataset)

    # Eval Datasets
    inference_time_optimization(
        model=model,
        dataset=dataset,
        accelerator=accelerator,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        log_every=args.log_every,
        zero_shot=args.zero_shot,
        epochs=args.epochs,
        lr=args.lr,
        ct_batch_size=args.ct_batch_size,
        token_dropout=args.token_dropout,
    )
    accelerator.end_training()


if __name__ == "__main__":
    main()
