import abc
import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Type, Union, cast

import datasets
import torch
from olmo_eval import ICLMultiChoiceTaskDataset, Tokenizer, list_tasks
from olmo_eval.tasks import LABEL_TO_TASK_MAP
from olmo_eval.util import load_hf_dataset

log = logging.getLogger(__name__)


class ICLMultiChoiceTaskDatasetUpdated(metaclass=abc.ABCMeta):
    """Only supports zero-shot for now."""

    metric_type: str

    def __init__(
        self,
        tokenizer: Tokenizer,
        dataset_path: str,
        dataset_name: Union[str, Sequence[str], None] = None,
        filter_tag: Optional[str] = None,
        model_ctx_len: int = 2048,
        fixed_ctx_len: bool = False,
        split="validation",
        metric_type=None,  # Override default metric type
        prompts: Optional[List[Optional[str]]] = None,  # List of prompt variants to use,
        expert_mask_exclude=None,  # swj
        subset_size=None,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.model_ctx_len = model_ctx_len
        self.fixed_ctx_len = fixed_ctx_len
        self.prompts = prompts or [None]
        self.current_prompt: Optional[str] = None
        if metric_type is not None:
            self.metric_type = metric_type
        self.log_instances = 0  # Set to > 0 to log the first few instances as a sanity check
        self.subset_size = subset_size
        self.samples: List[Dict[str, Any]] = []
        dataset_names: Sequence[Optional[str]]
        if isinstance(dataset_name, str) or dataset_name is None:
            dataset_names = [dataset_name]
        else:
            dataset_names = dataset_name

        dataset_list = []
        for ds_name in dataset_names:
            ############################################################################
            # load_hf_dataset("fancyzhx/ag_news","test")
            # bp()
            try:
                dataset = load_hf_dataset(self.dataset_path, ds_name, split)
            except NotADirectoryError:
                from datasets import load_dataset

                try:
                    # bp()
                    dataset = load_dataset(self.dataset_path, ds_name, split=split)
                except:  # noqa:E722
                    dataset = load_dataset(self.dataset_path, split=split)

            if filter_tag is not None:
                # filter the dataset by tag
                dataset = dataset.filter(lambda x: x["primary_tag"] == filter_tag)

            # choose only a subset of the dataset if subset_size is not None and the subset size is less than the dataset size

            # TODO: refactor this!
            if self.subset_size is not None and self.subset_size < len(dataset):  # type: ignore
                dataset = dataset.select(range(self.subset_size))  # type: ignore
                # bp()
            dataset_list.append(dataset)
        self.dataset = datasets.concatenate_datasets(dataset_list)
        self.expert_mask_exclude = expert_mask_exclude

        ############################################################################

        # prep examples
        self.prep_examples()
        self._max_sequence_length: Optional[int] = None

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)

    def prep_examples(self):
        """Append doc_ids to each example so that they are processed together in the metric"""
        doc_id = 0
        for doc in self.dataset:
            for prompt in self.prompts:
                self.current_prompt = prompt
                # from EAI harness
                # how this all works:
                #          CTX      CONT
                # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
                # gpt2    \               \
                # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
                # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice

                continuations = self.doc_to_continuations(doc)
                label_id = self.doc_to_label(doc)
                doc_text = self.doc_to_text(doc)
                ctx = self.token_encode(doc_text)
                dc = self.token_encode(self.doc_to_domain_conditional(doc))
                if self.log_instances > 0:
                    self.log_instances -= 1
                    ds_name = self.dataset_name
                    if isinstance(ds_name, list):
                        ds_name = ds_name[0]
                    log.info(
                        f"Sample doc from ({self.dataset_path}, {ds_name}, {self.current_prompt}):"
                        + f"\ndoc_text: {doc_text}\ncontinuations: {continuations}"
                    )

                for cont_id, continuation_str in enumerate(continuations):
                    cont_str_len = len(continuation_str) - 1  # continuation contain leading blank
                    cont_byte_len = len(continuation_str[1:].encode("utf-8"))
                    continuation = self.token_encode(continuation_str)

                    # query, remove last token from continuation, truncate from left is longer than model ctx length
                    query = ctx + continuation[:-1]
                    query = query[-self.model_ctx_len :]
                    # this will be different from len(ctx) when truncated by model_ctx_len
                    actual_ctx_len = len(query) - len(continuation) + 1

                    # get domain conditional query
                    # we don't expect this to be longer than self.model_ctx_len and it won't make sense to truncate from left
                    dc_query = dc + continuation[:-1]

                    # swj
                    if self.expert_mask_exclude is not None:
                        # swj hard code 64
                        expert_mask_exclude_tensor = torch.ones(224)  # swj: hard code
                        for data_type in self.expert_mask_exclude.keys():
                            start_id, end_id = self.expert_mask_exclude[data_type]
                            expert_mask_exclude_tensor[start_id:end_id] = 0
                    else:
                        expert_mask_exclude_tensor = None

                    # form a sample
                    self.samples.append(
                        {
                            "doc_id": doc_id,
                            "cont_id": cont_id,
                            "ctx": ctx,
                            "continuation": continuation,
                            "ctx_len": actual_ctx_len,
                            "dc_len": len(dc),
                            "cont_len": len(
                                continuation
                            ),  # even if query has last token removed, LM will output same cont len
                            "cont_str_len": cont_str_len,
                            "cont_byte_len": cont_byte_len,
                            "query": query,  # remove last token from continuation
                            "dc_query": dc_query,
                            "label_id": label_id,
                            "expert_mask": expert_mask_exclude_tensor,  # swj
                        }
                    )

                doc_id += 1

    def pad_tokens_until_max(self, tokens, max_len=2048):
        """truncate from left if len(tokens) > model_ctx_len, max_len is not considered then
        queries are already truncated at max length of model_ctx_len
        this acts as additional check for all types of sequences in the batch
        """
        if len(tokens) > self.model_ctx_len:
            return tokens[-self.model_ctx_len :]
        else:
            # pad to max_len, but check again if this padding exceeded self.model_ctx_len
            # this time truncate from right side of the sequence because additional padding caused len(tokens) > self.model_ctx_len
            tokens = tokens + [self.tokenizer.pad_token_id] * (max_len - len(tokens))

            if len(tokens) > self.model_ctx_len:
                tokens = tokens[: self.model_ctx_len]

            return tokens

    @property
    def max_sequence_length(self) -> int:
        if self._max_sequence_length is None:
            max_seq_len = 0
            for sample in self.samples:
                if len(sample["query"]) > max_seq_len:
                    max_seq_len = len(sample["query"])
            # Pad to multiple of 128 for efficiency.
            # TODO (epwalsh): make that configurable
            max_seq_len = 128 * math.ceil(max_seq_len / 128)
            self._max_sequence_length = max_seq_len
        return self._max_sequence_length

    def collate_fn(self, data):
        # pad to max length
        # 'ctx', 'continuation', 'query' can all have variable length
        max_ctx_len = 0
        max_cont_len = 0
        max_query_len = 0
        max_dc_query_len = 0

        for sample in data:
            if len(sample["ctx"]) > max_ctx_len:
                max_ctx_len = len(sample["ctx"])

            if len(sample["continuation"]) > max_cont_len:
                max_cont_len = len(sample["continuation"])

            if len(sample["query"]) > max_query_len:
                max_query_len = len(sample["query"])

            if len(sample["dc_query"]) > max_dc_query_len:
                max_dc_query_len = len(sample["dc_query"])

        # Pad to multiple of 128 for efficiency.
        # TODO (epwalsh): make that configurable
        max_query_len = 128 * math.ceil(max_query_len / 128)
        assert max_query_len <= self.max_sequence_length, (
            f"{max_query_len=}",
            f"{self.max_sequence_length=}",
        )

        doc_ids = []
        cont_ids = []
        ctxs = []
        continuations = []
        ctx_lens = []
        dc_lens = []
        cont_lens = []
        cont_str_lens = []
        cont_byte_lens = []
        queries = []
        dc_queries = []
        label_ids = []
        expert_mask = []

        # pad according to max_lengths
        for sample in data:
            doc_ids.append(sample["doc_id"])
            cont_ids.append(sample["cont_id"])

            ctxs.append(
                torch.LongTensor(self.pad_tokens_until_max(sample["ctx"], max_len=max_ctx_len))
            )
            continuations.append(
                torch.LongTensor(
                    self.pad_tokens_until_max(sample["continuation"], max_len=max_cont_len)
                )
            )

            ctx_lens.append(sample["ctx_len"])
            dc_lens.append(sample["dc_len"])
            cont_lens.append(sample["cont_len"])
            cont_str_lens.append(sample["cont_str_len"])
            cont_byte_lens.append(sample["cont_byte_len"])

            queries.append(
                torch.LongTensor(
                    self.pad_tokens_until_max(
                        sample["query"],
                        max_len=self.model_ctx_len if self.fixed_ctx_len else max_query_len,
                    )
                )
            )
            dc_queries.append(
                torch.LongTensor(
                    self.pad_tokens_until_max(sample["dc_query"], max_len=max_dc_query_len)
                )
            )

            label_ids.append(sample["label_id"])
            if "expert_mask" in sample and sample["expert_mask"] is not None:
                expert_mask.append(sample["expert_mask"])  # swj

        if len(expert_mask) > 0:
            expert_mask = torch.stack(expert_mask)
        else:
            expert_mask = None  # type: ignore
        batch = {
            "doc_id": torch.LongTensor(doc_ids),
            "cont_id": torch.LongTensor(cont_ids),
            "ctx": torch.stack(ctxs),
            "continuation": torch.stack(continuations),
            "ctx_len": torch.LongTensor(ctx_lens),
            "dc_len": torch.LongTensor(dc_lens),
            "cont_len": torch.LongTensor(
                cont_lens
            ),  # since query has last token removed from continuation
            "cont_str_len": torch.LongTensor(cont_str_lens),
            "cont_byte_len": torch.LongTensor(cont_byte_lens),
            "input_ids": torch.stack(queries),
            "dc_input_ids": torch.stack(dc_queries),
            "label_id": torch.LongTensor(label_ids),
            "expert_mask": expert_mask,  # swj
        }

        return batch

    def token_encode(self, string: str) -> List[int]:
        return self.tokenizer.encode(string, add_special_tokens=False)

    def token_decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)

    @abc.abstractmethod
    def doc_to_text(self, doc) -> str:
        """Match EAI eval harness
        returns a single context string
        """
        raise NotImplementedError

    @abc.abstractmethod
    def doc_to_continuations(self, doc) -> List[str]:
        """Match EAI eval harness
        returns a list of continuations
        """
        raise NotImplementedError

    @abc.abstractmethod
    def doc_to_label(self, doc) -> int:
        """Match EAI eval harness
        returns continuation id which corresponds to true label
        """
        raise NotImplementedError

    def doc_to_domain_conditional(self, doc) -> str:
        """Provide string for domain conditional normalization
        by default its blank string, continuation normalized by prob conditioned on a blank
        """
        del doc
        return " "


class XSum(ICLMultiChoiceTaskDatasetUpdated):
    metric_type = "ce_loss"

    def __init__(
        self,
        tokenizer,
        dataset_path="EdinburghNLP/xsum",
        dataset_name=None,
        split="test",
        expert_mask_exclude=None,
        subset_size=300,
        **kwargs,
    ):
        super().__init__(
            tokenizer,
            dataset_path,
            dataset_name,
            split=split,
            expert_mask_exclude=expert_mask_exclude,
            subset_size=subset_size,
            **kwargs,
        )

    def doc_to_text(self, doc):
        return "Document: " + doc["document"] + "\nSummary:"

    def doc_to_continuations(self, doc):
        return [" " + doc["summary"]]

    def doc_to_label(self, doc):
        return 0

    def doc_to_domain_conditional(self, doc):
        del doc
        return "Summary:"


class WildBench(ICLMultiChoiceTaskDatasetUpdated):
    metric_type = "ce_loss"
    _usecases = [
        "Creative Writing",
        "Data Analysis",
        "Brainstorming",
        "Editing",
        "Math",
        "Information seeking",
        "Advice seeking",
        "Reasoning",
        "Coding & Debugging",
        "Role playing",
        "Planning",
    ]

    def __init__(
        self,
        tokenizer,
        dataset_path="allenai/WildBench",
        dataset_name="v2",
        split="test",
        filter_tag=None,
        expert_mask_exclude=None,
        subset_size=None,
        **kwargs,
    ):
        super().__init__(
            tokenizer,
            dataset_path,
            dataset_name,
            split=split,
            filter_tag=filter_tag,
            expert_mask_exclude=expert_mask_exclude,
            subset_size=subset_size,
            **kwargs,
        )

    def doc_to_text(self, doc):
        conversation_input = doc["conversation_input"]
        query = "\n".join(f"{c['role']}: {c['content']}" for c in conversation_input)
        # bp()
        return query + "\n" + "assistant:"

    def doc_to_continuations(self, doc):
        # bp()
        return [" " + doc["references"]["gpt-4"]]

    def doc_to_label(self, doc):
        return 0

    def doc_to_domain_conditional(self, doc):
        return "assistant:"


class NarrativeQA(ICLMultiChoiceTaskDatasetUpdated):
    metric_type = "ce_loss"

    def __init__(
        self,
        tokenizer,
        dataset_path="deepmind/narrativeqa",
        dataset_name=None,
        split="test",
        filter_tag=None,
        expert_mask_exclude=None,
        subset_size=300,
        **kwargs,
    ):
        super().__init__(
            tokenizer,
            dataset_path,
            dataset_name,
            split=split,
            filter_tag=filter_tag,
            expert_mask_exclude=expert_mask_exclude,
            subset_size=subset_size,
        )

    def doc_to_text(self, doc):
        return (
            "Document: "
            + doc["document"]["summary"]["text"]
            + "\nQuestion: "
            + doc["question"]["text"]
            + "\nAnswer:"
        )

    def doc_to_continuations(self, doc):
        return [" " + doc["answers"][0]["text"]]

    def doc_to_label(self, doc):
        return 0

    def doc_to_domain_conditional(self, doc):
        return "Answer:"


LABEL_TO_TASK_MAP_NEW = {
    "xsum": XSum,
    # "wildbench": WildBench,  # TODO: size mismatch error (probably due to brainstorming subset)
    "wildbench_creative_writing": (WildBench, {"filter_tag": "Creative Writing"}),
    "wildbench_data_analysis": (WildBench, {"filter_tag": "Data Analysis"}),
    # "wildbench_brainstorming": (WildBench, {"filter_tag": "Brainstorming"}),  # TODO: size mismatch error
    "wildbench_editing": (WildBench, {"filter_tag": "Editing"}),
    "wildbench_math": (WildBench, {"filter_tag": "Math"}),
    "wildbench_information_seeking": (WildBench, {"filter_tag": "Information seeking"}),
    "wildbench_advice_seeking": (WildBench, {"filter_tag": "Advice seeking"}),
    "wildbench_reasoning": (WildBench, {"filter_tag": "Reasoning"}),
    "wildbench_coding_debugging": (WildBench, {"filter_tag": "Coding & Debugging"}),
    "wildbench_role_playing": (WildBench, {"filter_tag": "Role playing"}),
    "wildbench_planning": (WildBench, {"filter_tag": "Planning"}),
    # "narrativeqa": (
    #     NarrativeQA,
    #     {
    #         "dataset_path": "deepmind/narrativeqa",
    #         "dataset_name": None,
    #         "split": "test",
    #         "filter_tag": None,
    #         "expert_mask_exclude": None,
    #         "subset_size": 300,
    #     },
    # ),
}

LABEL_TO_TASK_MAP.update(LABEL_TO_TASK_MAP_NEW)


def updated_list_tasks():
    return list_tasks() + list(LABEL_TO_TASK_MAP_NEW.keys())


def build_task(
    label: str,
    tokenizer: Tokenizer,
    model_ctx_len: int = 2048,
    fixed_ctx_len: bool = False,
) -> Union[ICLMultiChoiceTaskDataset, ICLMultiChoiceTaskDatasetUpdated]:
    task_class = LABEL_TO_TASK_MAP[label]
    task_kwargs = {}
    if isinstance(task_class, tuple):
        task_class, task_kwargs = task_class

    if issubclass(task_class, ICLMultiChoiceTaskDatasetUpdated):
        return cast(Type[ICLMultiChoiceTaskDatasetUpdated], task_class)(
            tokenizer=tokenizer, **task_kwargs
        )
    return cast(Type[ICLMultiChoiceTaskDataset], task_class)(
        tokenizer=tokenizer, model_ctx_len=model_ctx_len, fixed_ctx_len=fixed_ctx_len, **task_kwargs
    )
