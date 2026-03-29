#!/bin/bash
#SBATCH --job-name=gpt2_tests
#SBATCH --output=test_output_%j.log
#SBATCH --error=test_error_%j.log
#SBATCH --partition=columbia
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G

cd ~/hw2
source .venv/bin/activate

python -m pytest tests/hw2_test.py::test_loads_and_forward_pass -v
python -m pytest tests/hw2_test.py::test_probability_tolerance -v
python -m pytest tests/hw2_test.py::test_greedy_sampling -v
python -m pytest tests/hw2_test.py::test_nucleus_sampling -v
