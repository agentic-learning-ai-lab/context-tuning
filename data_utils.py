import math
import os
import json
from collections import defaultdict
from typing import Dict, List, List, Optional, Union, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from accelerate.logging import get_logger
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers import GPT2TokenizerFast


logger = get_logger(__name__, log_level="INFO")


def pad_sequence_with_side(sequences: List[torch.Tensor], padding_value: int, side: str) -> torch.Tensor:
    if side == 'right':
        return pad_sequence(sequences, batch_first=True, padding_value=padding_value)
    else:
        reversed_sequences = [seq.flip(0) for seq in sequences]
        padded_reversed = pad_sequence(reversed_sequences, batch_first=True, padding_value=padding_value)
        return padded_reversed.flip(1)


def tokenize(
        text: str,
        tokenizer: Union[PreTrainedTokenizerFast, GPT2TokenizerFast]
    ) -> torch.Tensor:

    tokenizer_out = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    assert tokenizer_out['input_ids'].shape == tokenizer_out['attention_mask'].shape # type: ignore
    assert tokenizer_out['input_ids'].dim() == 2 and tokenizer_out['input_ids'].shape[0] == 1 # type: ignore
    assert tokenizer_out['attention_mask'].numel() == tokenizer_out['attention_mask'].sum() # type: ignore

    input_ids = tokenizer_out['input_ids'][0] # type: ignore
    if not isinstance(tokenizer, GPT2TokenizerFast):
        assert input_ids[0].item() == tokenizer.bos_token_id # type: ignore
        input_ids = input_ids[1:]

    return input_ids


def parse_pairs(
    pairs: List[Dict],
    tokenizer: Union[PreTrainedTokenizerFast, GPT2TokenizerFast],
    delimiter_token_id: torch.Tensor,
    is_train: bool,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                    Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[List]]]:

    # compute input_ids, attention_mask, label_ids for each pair
    input_ids_of_each_pair = []
    label_ids_of_each_pair = []
    final_output_start_idx = -1

    demon_input_ids = []
    demon_start_idxs = [0]

    for pair_i, pair in enumerate(pairs):
        input_input_ids = tokenize(pair['input'], tokenizer)
        output_input_ids = tokenize(pair['output'], tokenizer)

        if len(input_input_ids) > 256 - len(output_input_ids):
            input_input_ids = input_input_ids[: 256 - len(output_input_ids)]

        input_input_ids = torch.cat([input_input_ids, delimiter_token_id])
        if pair_i != 0:
            input_input_ids = torch.cat([delimiter_token_id, input_input_ids])

        if pair_i == len(pairs) - 1:
            final_output_start_idx = sum(x.shape[0] for x in input_ids_of_each_pair) + input_input_ids.shape[0]

        input_ids = torch.cat([input_input_ids, output_input_ids])
        if pair_i < len(pairs) - 1:
            label_ids = torch.full(input_ids.shape, -100, dtype=input_ids.dtype)
        else:
            label_ids = torch.cat([
                torch.full(input_input_ids.shape, -100, dtype=input_ids.dtype),
                output_input_ids,
            ])

        input_ids_of_each_pair.append(input_ids)
        label_ids_of_each_pair.append(label_ids)
        demon_input_ids.append(input_input_ids)
        demon_input_ids.append(output_input_ids)

        if pair_i < len(pairs) - 2:
            demon_start_idxs.append(demon_start_idxs[-1] + len(input_ids))

    # concat
    input_ids = torch.cat(input_ids_of_each_pair)
    attention_mask = torch.full(input_ids.shape, 1, dtype=input_ids.dtype)
    label_ids = torch.cat(label_ids_of_each_pair)
    assert input_ids.shape == attention_mask.shape == label_ids.shape

    assert final_output_start_idx > -1
    if len(input_ids) > 1024:
        return None

    if not is_train:
        gen_input_ids = torch.cat(demon_input_ids[-2:])
        gen_attention_mask = torch.full(gen_input_ids.shape, 1, dtype=torch.int64)
        gen_label_ids = torch.full(demon_input_ids[-2].shape, -100, dtype=input_ids.dtype)
        gen_label_ids = torch.cat([gen_label_ids, demon_input_ids[-1]])
        demon_input_ids = torch.cat(demon_input_ids[:-2])
        demon_attention_mask = torch.full(demon_input_ids.shape, 1, dtype=torch.int64)
    else:
        gen_input_ids, gen_attention_mask, gen_label_ids, demon_input_ids, demon_attention_mask, demon_start_idxs = None, None, None, None, None, None

    return input_ids, attention_mask, label_ids, \
        gen_input_ids, gen_attention_mask, gen_label_ids, demon_input_ids, demon_attention_mask, demon_start_idxs


def collate_data(
    batch: List[Dict],
    tokenizer: Union[PreTrainedTokenizerFast, GPT2TokenizerFast],
    pad_side: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:

    batch_size = len(batch)
    all_input_ids = [x['input_ids'] for x in batch]
    all_attention_mask = [x['attention_mask'] for x in batch]
    all_label_ids = [x['label_ids'] for x in batch]
    input_ids_lens = [len(x) for x in all_input_ids]

    # collate
    assert isinstance(tokenizer.pad_token_id, int)
    all_input_ids = pad_sequence_with_side(all_input_ids, padding_value=tokenizer.pad_token_id, side=pad_side)
    all_attention_mask = pad_sequence_with_side(all_attention_mask, padding_value=0, side=pad_side)
    all_label_ids = pad_sequence_with_side(all_label_ids, padding_value=-100, side=pad_side)
    assert all_input_ids.shape == all_attention_mask.shape == all_label_ids.shape

    assert len(all_input_ids) == batch_size
    assert all_input_ids.shape == all_attention_mask.shape == all_label_ids.shape
    return all_input_ids, all_attention_mask, all_label_ids, input_ids_lens


class EvalDataset:
    def __init__(
        self,
        data_dir: str,
        config_file: str,
        seed: int,
        eval_split: str,
        tokenizer: Union[PreTrainedTokenizerFast, GPT2TokenizerFast],
        pad_side: str,
        eval_ratio: float,
        num_demonstrations: int,
    ):
        self.tokenizer = tokenizer
        self.pad_side = pad_side
        self.seed = seed
        self.num_demonstrations = num_demonstrations

        # separate input and output by newline
        self.delimiter_token_id = tokenize(" ", tokenizer)

        # rng
        rng = np.random.RandomState(seed)

        # get tasks
        self.tasks = json.load(open(config_file))['test']
        assert len(self.tasks) == len(set(self.tasks))

        # load data
        task_to_demonstrations = defaultdict(list)
        task_to_test_pairs = defaultdict(list)
        for task in self.tasks:
            task_data_dir = os.path.join(data_dir, task)

            # train demonstration pairs
            train_file = os.path.join(task_data_dir, f"{task}_16_{eval_split}_train.jsonl")
            for l in open(train_file, 'r').readlines():
                example = json.loads(l)
                task_to_demonstrations[task].append(example)
            assert len(task_to_demonstrations[task]) == 16

            # subset demonstrations if needed
            task_to_demonstrations[task] = task_to_demonstrations[task][:num_demonstrations]

            # subset test pairs
            test_file = os.path.join(task_data_dir, f"{task}_16_{eval_split}_test.jsonl")
            lines = open(test_file, 'r').readlines()
            rng.shuffle(lines)
            num_chosen = math.ceil(len(lines) * eval_ratio)
            num_chosen = max(num_chosen, num_demonstrations - 16 + 1) # at least have one test
            for l in lines[:num_chosen]:
                example = json.loads(l)
                task_to_test_pairs[task].append(example)

            # take from test if needed
            if num_demonstrations > 16:
                task_to_demonstrations[task] += task_to_test_pairs[task][:num_demonstrations - 16]
                task_to_test_pairs[task] = task_to_test_pairs[task][num_demonstrations - 16:]
                assert len(task_to_demonstrations[task]) == num_demonstrations, len(task_to_demonstrations[task])

        # filter out tasks with empty options
        assert set(task_to_demonstrations.keys()) == set(task_to_test_pairs.keys()) == set(self.tasks)

        # process and filter down
        self.data = []
        unfiltered_total_test, filtered_total_test, unfiltered_total_sample = 0, 0, 0

        for task_i, (task, test_pairs) in enumerate(task_to_test_pairs.items()):
            logger.info(f'{task_i+1}/{len(task_to_test_pairs)}')

            test_added = 0
            patience = 0 # some tasks just contains sequences that are too long
            for test_idx, test_pair in enumerate(test_pairs):
                assert len(test_pair['options']) > 1
                assert test_pair['output'] in test_pair['options']
                correct_option = test_pair['output']

                # get outputs for each option
                outs = []
                for option in test_pair['options']:
                    test_pair['output'] = option
                    outs.append(self.format_and_filter(task_to_demonstrations[task], test_pair, test_idx, correct_option))

                # add to data, accumulate filter and unfilter stats
                unfiltered_total_test += 1
                unfiltered_total_sample += len(test_pair['options'])
                if None not in outs: # all tests fit in sequence length
                    self.data += outs
                    filtered_total_test += 1
                    test_added += 1
                    patience = 0
                else:
                    patience += 1

                if patience == 100:
                    break

        # some tasks may be completely filtered due to max sequence length
        self.tasks = sorted(set(data['task'] for data in self.data))
        self.task_to_demonstrations = {t: task_to_demonstrations[t] for t in self.tasks}
        assert set(self.tasks) == set(self.task_to_demonstrations.keys())


    def format_and_filter(self, demonstrations: List[Dict], test_pair: Dict, test_idx: int, correct_option: str) -> Optional[Dict]:
        # make sure they are all the same task with the same non-empty options
        task = test_pair['task']
        assert all(e['task'] == task for e in demonstrations) # test and demonstration pair have same task (dont need same option)
        assert correct_option in test_pair['options']

        out = parse_pairs(
            pairs=demonstrations + [test_pair],
            tokenizer=self.tokenizer,
            delimiter_token_id=self.delimiter_token_id,
            is_train=False,
        )
        if out == None:
            return None
        input_ids, attention_mask, label_ids, gen_input_ids, gen_attention_mask, gen_label_ids, demon_input_ids, demon_attention_mask, demon_start_idxs = out

        return {
            "task": task,
            "test_idx": test_idx,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label_ids": label_ids,
            "option": test_pair['output'],
            "correct_option": correct_option,
            "demon_input_ids": demon_input_ids,
            "demon_attention_mask": demon_attention_mask,
            "gen_input_ids": gen_input_ids,
            "gen_label_ids": gen_label_ids,
            "gen_attention_mask": gen_attention_mask,
            "demonstrations_pairs": demonstrations,
            "demon_start_idxs": demon_start_idxs,
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn_eval(batch: List[Dict], dataset: EvalDataset) -> Dict:
    batch_size = len(batch)

    all_input_ids, all_attention_mask, all_label_ids, input_ids_lens = collate_data(
        batch=batch,
        tokenizer=dataset.tokenizer,
        pad_side=dataset.pad_side,
    )
    assert len(all_input_ids) == batch_size

    # eval only
    task = [x['task'] for x in batch]
    test_idx = [x['test_idx'] for x in batch]
    option = [x['option'] for x in batch]
    correct_option = [x['correct_option'] for x in batch]
    demon_input_ids = [x['demon_input_ids'] for x in batch]
    demon_attention_mask = [x['demon_attention_mask'] for x in batch]
    gen_input_ids = [x['gen_input_ids'] for x in batch]
    gen_attention_mask = [x['gen_attention_mask'] for x in batch]
    gen_label_ids = [x['gen_label_ids'] for x in batch]
    demon_start_idxs = [x['demon_start_idxs'] for x in batch]

    # collate
    assert isinstance(dataset.tokenizer.pad_token_id, int)
    demon_input_ids = pad_sequence_with_side(demon_input_ids, padding_value=dataset.tokenizer.pad_token_id, side=dataset.pad_side)
    demon_attention_mask = pad_sequence_with_side(demon_attention_mask, padding_value=0, side=dataset.pad_side)
    gen_input_ids = pad_sequence_with_side(gen_input_ids, padding_value=dataset.tokenizer.pad_token_id, side=dataset.pad_side)
    gen_attention_mask = pad_sequence_with_side(gen_attention_mask, padding_value=0, side=dataset.pad_side)
    gen_label_ids = pad_sequence_with_side(gen_label_ids, padding_value=-100, side=dataset.pad_side)
    assert all_input_ids.shape == all_attention_mask.shape == all_label_ids.shape

    return {
        'task': task,
        'test_idx': test_idx,
        'input_ids': all_input_ids,
        'attention_mask': all_attention_mask,
        'label_ids': all_label_ids,
        'input_ids_lens': input_ids_lens,
        "option": option,
        "correct_option": correct_option,
        "demon_input_ids": demon_input_ids,
        "demon_attention_mask": demon_attention_mask,
        "gen_input_ids": gen_input_ids,
        "gen_attention_mask": gen_attention_mask,
        "gen_label_ids": gen_label_ids,
        "demon_start_idxs": demon_start_idxs,
    }


class GSDataset(Dataset):
    def __init__(
        self,
        demonstration_pairs: Dict[int, Dict],
        tokenizer: Union[PreTrainedTokenizerFast, GPT2TokenizerFast],
        pad_side: str,
        past_kv_len: int,
    ):
        self.demonstration_pairs = demonstration_pairs
        self.tokenizer = tokenizer
        self.pad_side = pad_side
        self.past_kv_len = past_kv_len

        # separate input and output by newline
        self.delimiter_token_id = tokenize(" ", tokenizer)

        # format data (only use demonstration pairs)
        parsed_examples = [self.format(i, example) for i, example in demonstration_pairs.items()]
        self.parsed_examples = [e for e in parsed_examples if e is not None]

    def __len__(self):
        return len(self.parsed_examples)

    def __getitem__(self, idx):
        return self.parsed_examples[idx]

    def format(self, example_idx: int, pair: Dict) -> Optional[Dict]:
        # tokenize
        input_input_ids = tokenize(pair['input'], self.tokenizer)
        output_input_ids = tokenize(pair['output'], self.tokenizer)

        # truncate each pair like in metaICL, not account for newlines tho
        if len(input_input_ids) > 256 - len(output_input_ids):
            input_input_ids = input_input_ids[: 256 - len(output_input_ids)]

        # append delimiter to input
        input_input_ids = torch.cat([input_input_ids, self.delimiter_token_id])
        input_input_ids = torch.cat([self.delimiter_token_id, input_input_ids])

        # create input_ids and label_ids
        input_ids = torch.cat([input_input_ids, output_input_ids])
        attention_mask = torch.full(input_ids.shape, 1, dtype=torch.int64)

        label_ids = torch.full(input_input_ids.shape, -100, dtype=input_ids.dtype)
        label_ids = torch.cat([label_ids, output_input_ids])

        overflow = len(input_ids) - (1024 - self.past_kv_len)
        if overflow > 0:
            return None

        return {
            "example_idx": example_idx,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label_ids": label_ids,
        }


def collate_fn_gs(batch: List[Dict], dataset: GSDataset) -> Dict:
    input_ids = [x["input_ids"] for x in batch]
    attention_mask = [x["attention_mask"] for x in batch]
    label_ids = [x["label_ids"] for x in batch]

    assert isinstance(dataset.tokenizer.pad_token_id, int)
    input_ids = pad_sequence_with_side(input_ids, padding_value=dataset.tokenizer.pad_token_id, side=dataset.pad_side)
    attention_mask = pad_sequence_with_side(attention_mask, padding_value=0, side=dataset.pad_side)
    label_ids = pad_sequence_with_side(label_ids, padding_value=-100, side=dataset.pad_side)

    batch_dict = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "label_ids": label_ids,
        "example_idx": [x['example_idx'] for x in batch],
    }
    return batch_dict
