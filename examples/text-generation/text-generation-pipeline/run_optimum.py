import argparse
import logging
import math
import time

from pipeline import GaudiTextGenerationPipeline
from run_generation import setup_parser


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
args = setup_parser(parser)
args.num_return_sequences = 1
args.model_name_or_path = "/intel/gemma-2-9b"
args.max_new_tokens = 64
args.use_hpu_graphs = True
args.use_kv_cache = True
args.do_sample = True
args.batch_size = 1
args.bf16 = True
args.warmup = 0
args.n_iterations = 1
args.max_input_tokens=8
args.bucket_size=-1
args.bucket_internal=False
args.prompt = ["Once upon a time, there is"]

if args.prompt:
    input_sentences = args.prompt


if args.batch_size > len(input_sentences):
    times_to_extend = math.ceil(args.batch_size / len(input_sentences))
    input_sentences = input_sentences * times_to_extend

input_sentences = input_sentences[: args.batch_size]

logger.info("Initializing text-generation pipeline...")
pipe = GaudiTextGenerationPipeline(args, logger)
outputs = pipe(input_sentences)
response = repr(outputs[0])
print(response)
