import argparse
from vllm.logger import init_logger
import random
from dataclasses import dataclass, asdict
import os, time, json
from typing import List
from datetime import datetime


from test_model import TestModelRunner, ReqMode, collect_meta, Meta, CaseResult


logger = init_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="The path of the model to load.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["float16", "bfloat16"],
        help="The dtype to be forced upon the model.",
    )
    parser.add_argument(
        "--tp_size",
        type=int,
        required=True,
        help="Tensor Parallel size. Set to 1 to disable tensor parallel.",
    )
    parser.add_argument(
        "--useV1", type=bool, default=True, help="Use vLLM V1. Set False to use V0."
    )
    parser.add_argument(
        "--cases",
        type=str,
        required=True,
        help="Test file in JSON format",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1000,
        help="random seed for dataset selection.",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default=None,
        help="The path to save the JSON results.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to resume from the last unfinished case.",
    )
    return parser.parse_args()


@dataclass
class Results:
    schema_version: str
    meta: Meta
    case_results: List[CaseResult]

    def save_json(self, path):
        abs_dir = os.path.dirname(os.path.abspath(path))
        os.makedirs(abs_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=4)
        logger.info(f"Results saved to {path}")

    @staticmethod
    def from_json(path):
        if not os.path.exists(path):
            res = Results(
                schema_version="v2",
                meta=collect_meta(),
                case_results=[],
            )
            res.save_json(path)
            return res
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        meta = Meta(**data["meta"])
        case_results = [CaseResult(**cr) for cr in data["case_results"]]
        return Results(
            schema_version=data["schema_version"],
            meta=meta,
            case_results=case_results,
        )


class RunRecords:
    def __init__(self, record_pth, resume=False):
        self.record_pth = record_pth
        abs_dir = os.path.dirname(os.path.abspath(record_pth))
        os.makedirs(abs_dir, exist_ok=True)
        if not (resume and os.path.exists(record_pth)):
            json.dump(
                [],
                open(record_pth, "w", encoding="utf-8"),
                ensure_ascii=False,
                indent=4,
            )

    def record(self, case: dict):
        item = "_".join([f"{k}_{v}" for k, v in case.items()])
        records = json.load(open(self.record_pth, "r", encoding="utf-8"))
        records.append(item)
        json.dump(
            records,
            open(self.record_pth, "w", encoding="utf-8"),
            ensure_ascii=False,
            indent=4,
        )

    def is_recorded(self, case: dict):
        item = "_".join([f"{k}_{v}" for k, v in case.items()])
        records = json.load(open(self.record_pth, "r", encoding="utf-8"))
        logger.error(f"records: {records}, item: {item}")
        return item in records


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    logger.info(f"Args: {args}")
    result_pth = args.save_json

    test_cases = json.load(open(args.cases, "r", encoding="utf-8"))
    logger.error(f"Test cases: {test_cases}")

    results = (
        Results.from_json(result_pth)
        if args.resume and result_pth
        else Results(
            schema_version="v2",
            meta=collect_meta(),
            case_results=[],
        )
    )

    max_model_len = max(
        [case["input_length"] + case["max_new_tokens"] for case in test_cases]
    )
    max_num_seqs = max([case["batch"] for case in test_cases])
    max_num_batched_tokens = max(
        [case["input_length"] * case["batch"] for case in test_cases]
    )

    runner = TestModelRunner(
        model_id=args.model_id,
        dtype=args.dtype,
        tp=args.tp_size,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        use_v1=args.useV1,
    )
    run_records = RunRecords(
        os.path.join(os.path.dirname(os.path.abspath(args.cases)), f"run_records.json"),
        resume=args.resume,
    )

    for case in test_cases:
        logger.error(f"case: {case}")
        if args.resume and run_records.is_recorded(case):
            logger.warning(f"Case {case} already recorded, skip.")
            continue
        run_records.record(case)
        batch = case["batch"]
        input_length = case["input_length"]
        max_new_tokens = case["max_new_tokens"]
        mode = ReqMode[case["mode"].upper()]
        image_size = case.get("image_size", None)

        case_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        logger.info(
            f"Starting case at {case_start_time}: batch={batch}, input_length={input_length}, max_new_tokens={max_new_tokens}, mode={mode}"
        )

        case_result = runner.run_test_case(
            batch, input_length, max_new_tokens, mode, image_size
        )

        case_end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        logger.info(
            f"Completed case at {case_end_time}: batch={batch}, input_length={input_length}, max_new_tokens={max_new_tokens}, mode={mode}"
        )

        results.case_results.append(case_result)
        if result_pth:
            results.save_json(result_pth)


"""
cases example:
[
    {
        "batch": 1,
        "input_length": 5,
        "max_new_tokens": 5,
        "mode": "generate"
    },
    {
        "batch": 2,
        "input_length": 10,
        "max_new_tokens": 10,
        "mode": "generate",
        "image_size": "128x128"
    },
    {
        "batch": 3,
        "input_length": 15,
        "max_new_tokens": 15,
        "mode": "chat"
    }
]
"""
