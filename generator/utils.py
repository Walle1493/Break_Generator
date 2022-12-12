import json
import linecache
import os
from collections import Counter, OrderedDict
from multiprocessing import Pool, cpu_count     # https://docs.python.org/3/library/multiprocessing.html
from logging import getLogger
from pathlib import Path
from itertools import chain
from typing import Callable, Dict, Iterable, List

import torch
from torch.utils.data import Dataset

from transformers import BartTokenizer, GPT2Tokenizer, T5Tokenizer
import pdb

logger = getLogger(__name__)

class Seq2SeqDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        max_source_length,
        max_target_length,
        type_path="train",
        n_obs=None,
        prefix="",
    ):
        super().__init__()
        self.type_path = type_path
        self.src_file = Path(data_dir).joinpath(type_path + ".source")
        self.tgt_file = Path(data_dir).joinpath(type_path + ".target")
        self.src_lens = self.get_char_lens(self.src_file)
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        assert min(self.src_lens) > 0, f"found empty line in {self.src_file}"
        self.tokenizer = tokenizer
        self.prefix = prefix
        if n_obs is not None:
            self.src_lens = self.src_lens[:n_obs]

    def __len__(self):
        return len(self.src_lens)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        index = index + 1  # linecache starts at 1
        if isinstance(self.tokenizer, T5Tokenizer):
            source_line = "Paraphrase: " + linecache.getline(str(self.src_file), index).rstrip("\n")
        else:
            source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"

        # add #1-#6 to tokenizer
        # if isinstance(self.tokenizer, BartTokenizer):
        #     index_tokens = ["#1", "#2", "#3", "#4", "#5", "#6"]
        #     self.tokenizer.add_tokens(index_tokens)

        if isinstance(self.tokenizer, BartTokenizer):
            # source_inputs = self.tokenizer(source_line+' '+self.tokenizer.mask_token, add_prefix_space=True)
            source_inputs = self.tokenizer(source_line, add_prefix_space=True)
        elif isinstance(self.tokenizer, T5Tokenizer):
            # source_inputs = self.tokenizer(source_line+' '+self.tokenizer.additional_special_tokens[0])
            source_inputs = self.tokenizer(source_line)
        elif isinstance(self.tokenizer, GPT2Tokenizer):
            source_inputs = self.tokenizer(source_line, add_prefix_space=True)
        else:
            raise ValueError("Tokenizer not recognized")
        
        if isinstance(self.tokenizer, T5Tokenizer):
            # target_inputs = self.tokenizer(source_line+' '+tgt_line)
            target_inputs = self.tokenizer(self.tokenizer.pad_token + ' ' + tgt_line)
            # target_inputs = self.tokenizer(tgt_line)
        else:
            # target_inputs = self.tokenizer(source_line+' '+tgt_line, add_prefix_space=True)
            target_inputs = self.tokenizer(self.tokenizer.eos_token + ' ' + tgt_line, add_prefix_space=True)
            # target_inputs = self.tokenizer(tgt_line, add_prefix_space=True)

        source_ids = source_inputs["input_ids"]
        target_ids = target_inputs["input_ids"]
        src_mask = source_inputs["attention_mask"]
        qid = self.type_path + str(index)
        return {
            "qid": qid,
            "input_ids": source_ids,
            "attention_mask": src_mask,
            "decoder_input_ids": target_ids,
        }

    @staticmethod
    def get_char_lens(data_file):
        return [len(x) for x in Path(data_file).open().readlines()]

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        pad_token_id = self.tokenizer.pad_token_id
        max_src_len = max([len(x['input_ids']) for x in batch])
        max_tgt_len = max([len(x['decoder_input_ids']) for x in batch])
        input_ids = []
        masks = []
        target_ids = []
        for x in batch:
            input_ids.append(x['input_ids']+[pad_token_id]*(max_src_len-len(x['input_ids'])))
            masks.append(x['attention_mask']+[0]*(max_src_len-len(x['attention_mask'])))
            target_ids.append(x['decoder_input_ids']+[pad_token_id]*(max_tgt_len-len(x['decoder_input_ids'])))
        source_ids = torch.tensor(input_ids, dtype=torch.long)
        source_mask = torch.tensor(masks, dtype=torch.long)
        y = torch.tensor(target_ids, dtype=torch.long)
        qids = [x['qid'] for x in batch]
        batch = {
            "qids": qids,
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "decoder_input_ids": y,
        }
        return batch


class ProtoQA_Dataset(Dataset):
    
    def __init__(self, 
        dataset_path, model_type='gpt2',
        max_src_len=None, max_tgt_len=None, 
        tokenizer=None, 
        dataset_type='train', evaluate=False,
        experiment='', seed=42
        ):

        super().__init__()
        self.evaluate = evaluate
        self.raw_examples = []
        self.labels = []
        self.examples = []
        self.qids = []
        self.scores = []

        self.tokenizer = tokenizer
        self.dataset_path = dataset_path
        self.dataset_type = dataset_type
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.experiment = experiment
        self.model_type = model_type
        self.seed = seed

        if evaluate:
            self.load_data()
        else:
            self.load_data()
            self.convert_tokens_to_ids()

                # logging.info(f"dumping features to {file_name}")
                # with open(cached_feature_file, 'wb') as handle:
                    # pickle.dump(self.raw_examples, handle)
                    # pickle.dump(self.examples, handle)

    def load_data(self):
        if not os.path.isfile(self.dataset_path):
            if not self.evaluate:
                # no need label
                target = os.path.join(self.dataset_path, f"{self.dataset_type}.scraped.jsonl")
            else:
                target = os.path.join(self.dataset_path, f"test.questions.jsonl")
        else:
            target = self.dataset_path
        assert os.path.isfile(target)
        
        f = open(target, 'r', encoding='utf-8')
        for line in f.readlines():
            line = line.strip()
            line_dict = json.loads(line)

            example_id = line_dict['metadata']['id']
            question = line_dict['question']['original']

            question = self.transform_question(question)
            if 'keyword_wkdt' in self.experiment:
                keyword = line_dict['keyword']
                description = []
                for word in keyword:
                    if 'dpr' in self.experiment:
                        desc = line_dict['description_dict'][word]
                        if desc != 'MISMATCH':
                            des = f"{word}: {desc}"
                            des = des + '.' if des[-1] != '.' else des
                            description.append(des)
                    else:
                        wkdt = self.wkdt.match_description(word)
                        if wkdt['matched'] != "MISMATCH":
                            des = f"{word}: {wkdt['desc_list'][0]}"
                            des = des + '.' if des[-1] != '.' else des
                            description.append(des)
            elif 'knowledge' in self.experiment:
                try:
                    knowledge = line_dict['knowledge']
                except:
                    import pdb; pdb.set_trace()
                    print(1)

            if self.evaluate:
                temp = [example_id, question]
                if 'keyword_wkdt' in self.experiment:
                    if 'multi_3' in self.experiment:
                        temp.append('; '.join(description[:3] if description else ''))
                    elif 'multi_4' in self.experiment:
                        temp.append('; '.join(description[:4] if description else ''))
                    elif 'multi' in self.experiment:
                        temp.append('; '.join(description[:2] if description else ''))
                    elif 'tp2' in self.experiment and len(description) >=2:
                        temp.append(description[1] if description else '')
                    else:
                        temp.append(description[0] if description else '')
                elif 'knowledge' in self.experiment:
                    knowledge = line_dict['knowledge']
                    temp.append(knowledge)
                self.raw_examples.append(temp)
            else:
                answers = list(line_dict['answers']['raw'].keys())
                answers = self.transform_answer_list(answers)
                
                if 'rm_bad' in self.experiment:
                    answers = [answer for answer in answers if answer != 'send us your answers!']
                                    
                answers = [' '+answer+'.' for answer in answers]    # add space
                
                for answer in answers:
                    temp = [question, answer]
                    if 'keyword_wkdt' in self.experiment:
                        if 'multi_3' in self.experiment:
                            temp.append('; '.join(description[:3] if description else ''))
                        elif 'multi_4' in self.experiment:
                            temp.append('; '.join(description[:4] if description else ''))
                        elif 'multi' in self.experiment:
                            temp.append('; '.join(description[:2] if description else ''))
                        else:
                            temp.append(description[0] if description else '')
                    elif 'knowledge' in self.experiment:
                        temp.append(knowledge)
                    self.qids.append(example_id)
                    self.raw_examples.append(temp)
        logger.info(f"{self.dataset_type} dataset loaded")

    class Convert:
        # tokenize and get the max len   for multiprocessing
        def __init__(self, 
            tokenizer, experiment, 
            evaluate, model_type='gpt2',
            max_src_len=50, max_tgt_len=50
            # max_seq_len=None      # TODO
            ):
            self.tokenizer = tokenizer
            self.evaluate = evaluate
            self.experiment = experiment
            self.model_type = model_type
            self.max_src_len = max_src_len   # for bart encoder
            self.max_tgt_len = max_tgt_len   # for gpt2 & bart decoder

        def __call__(self, raw_example):
            if self.model_type.lower() == 'bart':
                try:
                    return self._bart(raw_example)
                except:
                    logger.warning(raw_example)
                    exit()
            elif self.model_type.lower() == 't5':
                try:
                    return self._t5(raw_example)
                except:
                    logger.warning(raw_example)
                    exit()
            elif self.model_type == 'gpt2':
                return self._gpt2(raw_example)

        def _t5(self, raw_example):
            
            if 'keyword_wkdt' in self.experiment or \
                'knowledge' in self.experiment:
                question, answer, description = raw_example
                source_inputs = self.tokenizer(description, question + ' ' + self.tokenizer.    additional_special_tokens[0], max_length=self.max_src_len, padding='max_length', truncation='only_first')
                target_inputs = self.tokenizer.encode(description, question + answer,  max_length=self.max_tgt_len, padding='max_length', truncation='only_first')
            else:
                question, answer = raw_example
                source_inputs = self.tokenizer(question + " " + self.tokenizer.additional_special_tokens[0], max_length=self.max_src_len, padding='max_length', truncation=True)
                target_inputs = self.tokenizer.encode(question + answer, 
                    max_length=self.max_tgt_len, padding='max_length', truncation=True)

            feature_dict = source_inputs
            feature_dict['decoder_input_ids'] = target_inputs
            return feature_dict
        
        def _bart(self, raw_example):
            
            if 'keyword_wkdt' in self.experiment or \
                'knowledge' in self.experiment:
                question, answer, description = raw_example
                source_inputs = self.tokenizer(description, question + self.tokenizer.mask_token, 
                add_prefix_space=True, max_length=self.max_src_len, padding='max_length', truncation='only_first')

                target_inputs = self.tokenizer.encode(description, question + answer, add_prefix_space=True, max_length=self.max_tgt_len, padding='max_length', truncation='only_first')
            else:
                question, answer = raw_example
                source_inputs = self.tokenizer(question + self.tokenizer.mask_token, 
                    add_prefix_space=True, max_length=self.max_src_len, padding='max_length', truncation=True)
                target_inputs = self.tokenizer.encode(question + answer, 
                    add_prefix_space=True, max_length=self.max_tgt_len, padding='max_length', truncation=True)

            feature_dict = source_inputs
            feature_dict['decoder_input_ids'] = target_inputs
            return feature_dict

        def _gpt2(self, raw_example):
            if 'keyword_wkdt' in self.experiment or \
                'knowledge' in self.experiment:
                question, answer, description = raw_example
                tokenized_source = self.tokenizer.encode(description+self.tokenizer.eos_token, question, max_length=self.max_src_len, truncation='only_first')
                tokenized_answer = self.tokenizer.encode(answer+self.tokenizer.eos_token)
                return (tokenized_source, tokenized_answer)
            else:
                question, answer = raw_example
                tokenized_question = self.tokenizer.encode(question)
                tokenized_answer = self.tokenizer.encode(answer+self.tokenizer.eos_token)
                return (tokenized_question, tokenized_answer)
    
    def convert_tokens_to_ids(self):
        '''
        make input and label
        tokenized_examples: 
        -gpt2 [question, answer] 
        -bart [
            source: <s>question <mask></s>
            target: <s>question answer.</s>
            ]
        '''
        
        logger.info(f"tokenizing {self.dataset_type} examples")
        logger.info(f'data format {self.raw_examples[0]}')
        
        # import pdb; pdb.set_trace()
        
        # now = time.time()
        if not self.tokenizer.mask_token:
            self.tokenizer.mask_token = '<extra_id_0>'
        with Pool(processes=min(8, cpu_count())) as pool:
            tokenized_examples = pool.map(
                self.Convert(self.tokenizer, self.experiment,
                self.evaluate, self.model_type, self.max_src_len, self.max_tgt_len),
                self.raw_examples)
        # logger.info(f"start {min(8, cpu_count())} processes, cost {time.time() - now}")

        if not self.max_src_len and self.model_type == 'gpt2':
            self.max_src_len = max([len(example[0]) + len(example[1]) for example in tokenized_examples])
            logger.info(f"GPT2 max_seq_len {self.max_src_len}")

        if self.evaluate:
            for example in tokenized_examples:
                q = example
                self.examples.append(q)
        else:
            if self.model_type == 'gpt2':
                # add eos token or tuncation
                for example in tokenized_examples:
                    q, a = example
                    total_len = len(q) + len(a)
                    if total_len < self.max_src_len:
                        for _ in range(total_len, self.max_src_len):
                            a.append(self.tokenizer.eos_token_id)
                    elif total_len > self.max_src_len:
                        longer = total_len - self.max_src_len
                        q = q[longer:]      # cut head
                    qa = list(chain(q, a))
                    self.examples.append(qa)
                    # make label
                    label = qa[:]
                    label[:len(q)] = [-100] * len(q)
                    end = False
                    for index in range(len(q), len(label)):
                        if label[index] == self.tokenizer.eos_token_id:
                            if not end:
                                end = True
                            else:
                                label[index] = -100
                    self.labels.append(label)
            elif self.model_type.lower() in ['bart', 't5']:
                for feature_dict in tokenized_examples:
                    source_input_ids = feature_dict['input_ids']
                    label = feature_dict['decoder_input_ids'][1:]
                    decoder_input_ids = feature_dict['decoder_input_ids'][:-1]# no need to input the last token 
                    feature_dict['decoder_input_ids'] = decoder_input_ids
                    self.examples.append(feature_dict)
                    # mask question part
                    q_part = True
                    for index, token in enumerate(label):
                        if q_part:
                            if token == source_input_ids[index+1]:
                                label[index] = -100
                            else:
                                q_part = False
                        elif token == self.tokenizer.pad_token_id:
                            label[index] = -100
                        elif token == self.tokenizer.eos_token_id and 'rank' in self.experiment:
                            label[index] = -100
                    self.labels.append(label)

    @staticmethod
    def transform_answer_list(answer_list, scores=None):
        
        for index in range(len(answer_list)):
            answer = answer_list[index]
            answer = answer.replace('\"', '')
            if '/' in answer:
                temp_list = answer.split('/')
                answer = temp_list.pop(0)
                if scores:
                    score = scores[index]
                    scores.extend([score] * len(temp_list))
                answer_list.extend(temp_list)
                
            answer_list[index] = answer
        if scores:
            return answer_list, scores
        else:
            return answer_list

    @staticmethod
    def transform_question(origin):
        '''
        > after having kids name something that happens that interrupts a couples alone time at night

        > after having kids one thing that happens that interrupts a couples alone time at night is

        '''
        question = origin.lower()
        question = question.replace('.', '')
        question = question.replace(':', '')
        question = question.replace('?', '')
        question = question.replace('someone', 'one person')
        question = question.replace('someplace', 'one place')
        transform_dict = {
            "name something": "one thing",
            'tell me something': 'one thing',
            'name a ': 'one ',
            "name an ": "one ",
            "name": "",
            # "name ": "",
            # "name another ": "another ",
            "SW tell me a ": "one ",
            "SW tell me an ": "one ",
            "SW what": "one",
            "SW give me a ": "one ",
            "SW tell me ": "",
            "which": "one",
            "what": "one",
            "how can you tell": "one way to tell",
        }
        order = ['name something', 'tell me something', 'name a ', 'name an ', 'name',
            'SW tell me a ', 'SW tell me an ', 'SW what', 'SW give me a ', 'SW tell me ',
            'which', 'what', 'how can you tell']
        transform = OrderedDict.fromkeys(order)
        transform.update(transform_dict)

        for pattern, trans in transform.items():
            if pattern.startswith('SW') and pattern[3:] in question:
                question = question.replace(pattern[3:], trans)
                question = question.strip() + ' is'
                break
            elif pattern in question:
                question = question.replace(pattern, trans)
                question = question.strip() + ' is'
                break
        else:
            question = 'Q: ' + question +'? A: '

        question = question[0].upper() + question[1:]

        return question

    def collate_fn(self, batch):
        # TODO  add GPT2
        if self.model_type == 'gpt2':
            pad_token_id = self.tokenizer.eos_token_id
            max_inputids_len = 0
            for example in batch:
                input_ids = example['input_ids']
                for index, token in enumerate(input_ids):
                    if token == pad_token_id:
                        max_inputids_len = max(max_inputids_len, index+1)
                        break

            input_ids = []
            labels = []

            for example in batch:
                input_ids.append(example['input_ids'][:max_inputids_len])
                labels.append(example['labels'][:max_dec_inputids_len])
            
            input_ids = torch.stack(input_ids)
            labels = torch.stack(labels)
            qids = [x['qid'] for x in batch]
            
            batch = {
                "qids": qids,
                "input_ids": input_ids,
                "labels": labels
            }
            return batch
        else:
            pad_token_id = self.tokenizer.pad_token_id
            max_inputids_len = 0
            max_dec_inputids_len = 0
            # import pdb; pdb.set_trace()
            for example in batch:
                input_ids = example['input_ids']
                decoder_input_ids = example['decoder_input_ids']
                for index, token in enumerate(input_ids):
                    if token == pad_token_id:
                        max_inputids_len = max(max_inputids_len, index)
                        break
                else:
                    max_inputids_len = max(max_inputids_len, len(input_ids)+1)
                for index, token in enumerate(decoder_input_ids):
                    if token == pad_token_id:
                        max_dec_inputids_len = max(max_dec_inputids_len, index)
                        break
                else:
                    max_dec_inputids_len = max(max_dec_inputids_len, len(decoder_input_ids)+1)

            input_ids = []
            masks = []
            target_ids = []
            labels = []
            for example in batch:
                input_ids.append(example['input_ids'][:max_inputids_len])
                masks.append(example['attention_mask'][:max_inputids_len])
                target_ids.append(example["decoder_input_ids"][:max_dec_inputids_len])
                labels.append(example['labels'][:max_dec_inputids_len])
            
            input_ids = torch.stack(input_ids)
            masks = torch.stack(masks)
            target_ids = torch.stack(target_ids)
            labels = torch.stack(labels)
            qids = [x['qids'] for x in batch]
            
            batch = {
                "qids": qids,
                "input_ids": input_ids,
                "attention_mask": masks,
                "decoder_input_ids": target_ids,
                "labels": labels
            }
            return batch

    def __len__(self):
        if self.evaluate:
            return len(self.raw_examples)
        else:
            return len(self.examples)

    def __getitem__(self, idx):
        if self.evaluate:
            return self.raw_examples[idx]
        else:
            if self.model_type.lower() in ['bart', 't5']:
                feature_dict= {}
                for key, value in self.examples[idx].items():
                    feature_dict[key] = torch.tensor(value)
                feature_dict['labels'] = torch.tensor(self.labels[idx])
                feature_dict['qids'] = self.qids[idx]
                return feature_dict
            elif self.model_type == 'gpt2':
                return {
                    'input_ids': torch.tensor(self.examples[idx]),
                    'labels': torch.tensor(self.labels[idx]),
                    'qids': self.qids[idx]
                }


class TriggerDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        max_source_length=512,
        max_target_length=512,
        type_path="train",
        n_obs=None,
        prefix="",
        model_type=None,
    ):
        super().__init__()
        self.type_path = type_path
        self.src_file = Path(data_dir).joinpath(type_path + ".source")
        self.tgt_file = Path(data_dir).joinpath(type_path + ".target")
        self.src_lens = self.get_char_lens(self.src_file)
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        assert min(self.src_lens) > 0, f"found empty line in {self.src_file}"
        self.tokenizer = tokenizer
        self.prefix = prefix
        if n_obs is not None:
            self.src_lens = self.src_lens[:n_obs]
        self.model_type = model_type
        if not hasattr(tokenizer, 'trigger_token'):
            raise ValueError(
                'Tokenizer missing special trigger tokens in vocab.'
                'Use `utils.add_special_tokens` to add them.'
            )

    def __len__(self):
        return len(self.src_lens)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"

        if isinstance(self.tokenizer, BartTokenizer):
            source_inputs = self.tokenizer(source_line+' '+self.tokenizer.mask_token, add_prefix_space=True)
        elif isinstance(self.tokenizer, T5Tokenizer):
            source_inputs = self.tokenizer(source_line+' '+self.tokenizer.additional_special_tokens[0])
        elif isinstance(self.tokenizer, GPT2Tokenizer):
            source_inputs = self.tokenizer(source_line, add_prefix_space=True)
        else:
            raise ValueError("Tokenizer not recognized")
        target_inputs = self.tokenizer(source_line+' '+tgt_line, add_prefix_space=True)

        source_ids = source_inputs["input_ids"]
        target_ids = target_inputs["input_ids"]
        src_mask = source_inputs["attention_mask"]
        qid = self.type_path+str(index)
        return {
            "qid": qid,
            "input_ids": source_ids,
            "attention_mask": src_mask,
            "decoder_input_ids": target_ids,
        }

    @staticmethod
    def get_char_lens(data_file):
        return [len(x) for x in Path(data_file).open().readlines()]

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        pad_token_id = self.tokenizer.pad_token_id
        max_src_len = max([len(x['input_ids']) for x in batch])
        max_tgt_len = max([len(x['decoder_input_ids']) for x in batch])
        input_ids = []
        masks = []
        target_ids = []
        for x in batch:
            input_ids.append(x['input_ids']+[pad_token_id]*(max_src_len-len(x['input_ids'])))
            masks.append(x['attention_mask']+[0]*(max_src_len-len(x['attention_mask'])))
            target_ids.append(x['decoder_input_ids']+[pad_token_id]*(max_tgt_len-len(x['decoder_input_ids'])))
        source_ids = torch.tensor(input_ids, dtype=torch.long)
        source_mask = torch.tensor(masks, dtype=torch.long)
        y = torch.tensor(target_ids, dtype=torch.long)
        qids = [x['qid'] for x in batch]

        if self.model_type == 'bart':
            decoder_input_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone()
            for i in range(len(lm_labels)):
                for j in range(len(lm_labels[i])):
                    if lm_labels[i][j] == source_ids[i][j+1]:
                        lm_labels[i][j] = -100
                    else:
                        break
            lm_labels[lm_labels == pad_token_id] = -100
        elif self.model_type == 'T5':
            decoder_input_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone()
            for i in range(len(lm_labels)):
                for j in range(len(lm_labels[i])):
                    if lm_labels[i][j] == source_ids[i][j+1]:
                        lm_labels[i][j] = -100
                    else:
                        break
            lm_labels[lm_labels == pad_token_id] = -100
        elif self.model_type == 'gpt2':
            decoder_input_ids = y[:, :-1].contiguous()
            lm_labels = y.clone()[:, 1:].clone()
            for i in range(len(lm_labels)):
                for j in range(1, len(source_ids[i])):
                    if lm_labels[i][j-1] == source_ids[i][j]:
                        lm_labels[i][j-1] = -100
                    else:
                        break
            lm_labels[lm_labels == pad_token_id] = -100

        decoder_trigger_mask = decoder_input_ids.eq(self.tokenizer.trigger_token_id)
        trigger_mask = source_ids.eq(self.tokenizer.trigger_token_id)

        batch = {
            "qids": qids,
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "decoder_input_ids": decoder_input_ids,
            "labels": lm_labels,
            "trigger_mask": trigger_mask, 
            "decoder_trigger_mask": decoder_trigger_mask,
        }
        return batch


class CommonGenDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        max_source_length=512,
        max_target_length=512,
        type_path="train",
        n_obs=None,
        prefix="",
    ):
        super().__init__()
        self.type_path = type_path
        self.src_file = Path(data_dir).joinpath(type_path + ".source")
        self.tgt_file = Path(data_dir).joinpath(type_path + ".target")
        self.src_lens = self.get_char_lens(self.src_file)
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        assert min(self.src_lens) > 0, f"found empty line in {self.src_file}"
        self.tokenizer = tokenizer
        self.prefix = prefix
        if n_obs is not None:
            self.src_lens = self.src_lens[:n_obs]

    def __len__(self):
        return len(self.src_lens)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = self.prefix + linecache.getline(str(self.tgt_file), index).rstrip("\n")
        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"
       
        source_inputs = self.tokenizer(source_line, add_prefix_space=True)
        target_inputs = self.tokenizer(tgt_line, add_prefix_space=True)

        source_ids = source_inputs["input_ids"]
        target_ids = target_inputs["input_ids"]
        src_mask = source_inputs["attention_mask"]
        qid = self.type_path+str(index)
        return {
            "qid": qid,
            "input_ids": source_ids,
            "attention_mask": src_mask,
            "decoder_input_ids": target_ids,
        }

    @staticmethod
    def get_char_lens(data_file):
        return [len(x) for x in Path(data_file).open().readlines()]

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        pad_token_id = self.tokenizer.pad_token_id
        max_src_len = max([len(x['input_ids']) for x in batch])
        max_tgt_len = max([len(x['decoder_input_ids']) for x in batch])
        input_ids = []
        masks = []
        target_ids = []
        for x in batch:
            input_ids.append(x['input_ids']+[pad_token_id]*(max_src_len-len(x['input_ids'])))
            masks.append(x['attention_mask']+[0]*(max_src_len-len(x['attention_mask'])))
            target_ids.append(x['decoder_input_ids']+[pad_token_id]*(max_tgt_len-len(x['decoder_input_ids'])))
        source_ids = torch.tensor(input_ids, dtype=torch.long)
        source_mask = torch.tensor(masks, dtype=torch.long)
        y = torch.tensor(target_ids, dtype=torch.long)
        qids = [x['qid'] for x in batch]
        if self.prefix == '':
            batch = {
                "qids": qids,
                "input_ids": source_ids,
                "attention_mask": source_mask,
                "decoder_input_ids": y,
            }
            return batch
        else:
            decoder_input_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone()
            lm_labels[lm_labels == pad_token_id] = -100
            lm_labels[lm_labels == self.tokenizer.trigger_token_id] = -100
            decoder_trigger_mask = decoder_input_ids.eq(self.tokenizer.trigger_token_id)
            trigger_mask = source_ids.eq(self.tokenizer.trigger_token_id)

            batch = {
                "qids": qids,
                "input_ids": source_ids,
                "attention_mask": source_mask,
                "decoder_input_ids": decoder_input_ids,
                "labels": lm_labels,
                "trigger_mask": trigger_mask, 
                "decoder_trigger_mask": decoder_trigger_mask,
            }
            return batch
