import argparse, os
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--method", type=str, default="gradcam")
    args = parser.parse_args()

    os.makedirs("results/figures", exist_ok=True)

    # 模拟一个 Grad-CAM 热力图
    heatmap = np.random.rand(64, 64)

    save_path = f"results/figures/{args.method}_example.png"
    plt.imshow(heatmap, cmap="jet")
    plt.axis("off")
    plt.title(f"{args.method.upper()} Visualization")
    plt.savefig(save_path)
    plt.close()

    print(f"✅ {args.method.upper()} 图已保存到 {save_path}")
