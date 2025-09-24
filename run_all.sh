#!/bin/bash
# ===============================
# Bearing Transfer Diagnosis - 全流程脚本
# 支持在 Colab + Google Drive 数据
# ===============================

set -e

# 参数
EPOCHS=20       # Colab 上先设短一些，跑通流程
BATCH=16
NCLS=4
MODEL_PATH="results/models/final.pth"

echo ">>> Step 1: 创建结果目录"
mkdir -p results/logs results/models results/figures data

echo ">>> Step 2: 拷贝 Google Drive 数据"
if [ -d "/content/drive/MyDrive/bearing_data" ]; then
  cp -r /content/drive/MyDrive/bearing_data/* data/
  echo ">>> 数据已拷贝到本地 data/ 目录"
else
  echo "⚠️ 未检测到 Google Drive 数据目录，请检查 /content/drive/MyDrive/bearing_data/"
fi

echo ">>> Step 3: 训练迁移诊断模型 (Task 3)"
python train.py --epochs $EPOCHS --batch_size $BATCH --ncls $NCLS \
    | tee results/logs/train.log

echo ">>> Step 4: 可解释性分析 (Task 4)"
python explain.py --model $MODEL_PATH --method gradcam \
    | tee results/logs/explain_gradcam.log
python explain.py --model $MODEL_PATH --method shap \
    | tee results/logs/explain_shap.log
python explain.py --model $MODEL_PATH --method cka \
    | tee results/logs/explain_cka.log
python explain.py --model $MODEL_PATH --method mc_dropout \
    | tee results/logs/explain_mc_dropout.log

echo ">>> Step 5: 完成！结果存放在 results/ 下"
