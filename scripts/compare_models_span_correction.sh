#!/bin/bash
# Compare GPT-OSS-20B vs OpenBioLLM-Llama3-8B for span correction

set -e

echo "================================================================"
echo "Model Comparison for Span Correction"
echo "================================================================"
echo

# Kill existing vLLM server
echo "Stopping existing vLLM server..."
pkill -f "vllm.entrypoints.openai.api_server" || true
sleep 3

# Test 1: GPT-OSS-20B (already tested, but include for completeness)
echo
echo "================================================================"
echo "TEST 1: GPT-OSS-20B (MoE, 20B params)"
echo "================================================================"
echo

echo "Starting vLLM with GPT-OSS-20B..."
nohup python -m vllm.entrypoints.openai.api_server \
    --model openai/gpt-oss-20b \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.95 \
    --max-num-seqs 256 \
    > /tmp/vllm_gpt_oss.log 2>&1 &

echo "Waiting for server to start..."
sleep 30

# Check if server is up
while ! curl -s http://localhost:8000/v1/models > /dev/null; do
    echo "Waiting for vLLM server..."
    sleep 5
done

echo "Running smoke test on GPT-OSS-20B..."
python scripts/smoke_test_span_correction.py \
    --models openai/gpt-oss-20b \
    --num-examples 30 \
    > outputs/smoke_test_gpt_oss_20b.txt 2>&1

echo "Results saved to: outputs/smoke_test_gpt_oss_20b.txt"
echo

# Kill server
pkill -f "vllm.entrypoints.openai.api_server"
sleep 3

# Test 2: OpenBioLLM-Llama3-8B
echo
echo "================================================================"
echo "TEST 2: OpenBioLLM-Llama3-8B (Dense, 8B params, medical)"
echo "================================================================"
echo

echo "Starting vLLM with OpenBioLLM-Llama3-8B..."
nohup python -m vllm.entrypoints.openai.api_server \
    --model bartowski/OpenBioLLM-Llama3-8B-AWQ \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 8192 \
    --quantization awq \
    --gpu-memory-utilization 0.95 \
    --max-num-seqs 256 \
    > /tmp/vllm_openbio.log 2>&1 &

echo "Waiting for server to start..."
sleep 30

# Check if server is up
while ! curl -s http://localhost:8000/v1/models > /dev/null; do
    echo "Waiting for vLLM server..."
    sleep 5
done

echo "Running smoke test on OpenBioLLM-Llama3-8B..."
python scripts/smoke_test_span_correction.py \
    --models bartowski/OpenBioLLM-Llama3-8B-AWQ \
    --num-examples 30 \
    > outputs/smoke_test_openbio_llama3_8b.txt 2>&1

echo "Results saved to: outputs/smoke_test_openbio_llama3_8b.txt"
echo

# Display comparison
echo
echo "================================================================"
echo "FINAL COMPARISON"
echo "================================================================"
echo

echo "GPT-OSS-20B Results:"
grep -A 5 "Results for openai/gpt-oss-20b:" outputs/smoke_test_gpt_oss_20b.txt || echo "Parse failed"

echo
echo "OpenBioLLM-Llama3-8B Results:"
grep -A 5 "Results for bartowski/OpenBioLLM-Llama3-8B-AWQ:" outputs/smoke_test_openbio_llama3_8b.txt || echo "Parse failed"

echo
echo "================================================================"
echo "Detailed results in:"
echo "  outputs/smoke_test_gpt_oss_20b.txt"
echo "  outputs/smoke_test_openbio_llama3_8b.txt"
echo "================================================================"
