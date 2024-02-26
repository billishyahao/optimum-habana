#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import logging
import time

import PIL.Image
import torch
from transformers import pipeline

from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name_or_path",
        default="Salesforce/blip-vqa-capfilt-large",
        type=str,
        help="Path to pre-trained model",
    )
    parser.add_argument(
        "--image_path",
        default="./vqa_cats.jpg",
        type=str,
        help="Path to image",
    )
    parser.add_argument(
        "--topk",
        default=1,
        type=int,
        help="topk num",
    )
    parser.add_argument(
        "--question",
        default="how many cats are in the picture?",
        type=str,
        help="topk num",
    )
    parser.add_argument(
        "--use_hpu_graphs",
        action="store_true",
        help="Whether to use HPU graphs or not. Using HPU graphs should give better latencies.",
    )
    args = parser.parse_args()

    adapt_transformers_to_gaudi()

    image = PIL.Image.open(args.image_path).convert("RGB")

    generator = pipeline(
        "visual-question-answering",
        model=args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device="hpu",
    )
    if not generator.model.can_generate() and args.use_hpu_graphs:
        from habana_frameworks.torch.hpu import wrap_in_hpu_graph

        generator.model = wrap_in_hpu_graph(generator.model)

    # warm up
    for i in range(5):
        generator(image, args.question, topk=args.topk)

    start = time.time()
    result = generator(image, args.question, topk=args.topk)
    end = time.time()
    print(f"result = {result}, time = {(end-start) * 1000}ms")


if __name__ == "__main__":
    main()
