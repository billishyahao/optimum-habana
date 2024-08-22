cd /intel/optimum-habana-gemma
pip uninstall -y optimum-habana
python setup.py install
cd /intel/optimum-habana-gemma/examples/text-generation
python run_generation.py --model_name_or_path ../../../Meta-Llama-3-8B/ --bf16 --use_kv_cache --max_input_tokens 128 --max_new_tokens 128 --bf16 --batch_size 1
