#!/bin/bash

# ==============================================================================
#  最终解决方案：此脚本将创建“无菌”环境，彻底解决路径污染问题
# ==============================================================================
# 设置在遇到任何错误时立即退出，防止后续错误操作
set -e

# --- 第0步：定义我们将要使用的、干净的Python环境 ---
PYTHON_EXEC="/root/miniconda3/envs/train_env/bin/python"
PIP_EXEC="/root/miniconda3/envs/train_env/bin/pip"
echo "--- Using Python from: $PYTHON_EXEC ---"

# --- 第1步：主动清除潜在的环境变量污染 ---
# 这是最关键的一步，确保我们不受服务器全局配置的干扰
echo "--- Step 1: Clearing potentially contaminating PYTHONPATH... ---"
unset PYTHONPATH

# --- 第2步：强制、干净地安装所有必需的依赖库 ---
echo "--- Step 2: Forcibly reinstalling all required packages... ---"
$PIP_EXEC install --no-cache-dir --upgrade pip
$PIP_EXEC install --no-cache-dir --upgrade torch torchvision torchaudio
$PIP_EXEC install --no-cache-dir --ignore-installed --upgrade datasets transformers accelerate peft bitsandbytes transformer-engine
# --ignore-installed 参数会强制重新安装，即便是同版本

# --- 第3步：终极验证，确认库的来源和版本 ---
echo "--- Step 3: Verifying library versions and paths... ---"
$PYTHON_EXEC -c "import transformers; print(f'>> Using Transformers version: {transformers.__version__}'); print(f'>> Loading from path: {transformers.__file__}')"

# --- 第4步：确保脚本中启用了FP8 ---
echo "--- Step 4: Ensuring fp8=True is set... ---"
sed -i "s/bf16=True/fp8=True/g" run_finetune.py
grep "fp8=True" run_finetune.py

# --- 第5步：在“无菌”环境中正式启动训练！ ---
echo "--- Step 5: All checks passed. Starting the training process... ---"
$PYTHON_EXEC run_finetune.py

echo "--- Training script has finished successfully. ---"
