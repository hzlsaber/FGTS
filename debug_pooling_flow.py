#!/usr/bin/env python3
"""
调试 FGTS 的完整池化流程
对比训练和测试时的数据流
"""

import numpy as np
import torch
import torch.nn as nn
from utils.features import select_tokens

class LinearProbe(nn.Module):
    """FGTS 的 LinearProbe（带内置池化）"""
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        print(f"  [LinearProbe.forward] 输入 shape: {x.shape}")
        # Pool if needed
        if x.dim() == 3:  # [B, T, D]
            print(f"  [LinearProbe.forward] 检测到 3D，执行池化...")
            x = x.mean(dim=1)  # [B, D]
            print(f"  [LinearProbe.forward] 池化后 shape: {x.shape}")
        else:
            print(f"  [LinearProbe.forward] 已经是 2D，跳过池化")
        out = self.fc(x)
        print(f"  [LinearProbe.forward] 输出 shape: {out.shape}")
        return out


def simulate_training_flow():
    """模拟训练流程"""
    print("="*70)
    print("训练流程")
    print("="*70)

    N = 100
    D = 512
    token_indices = [0, 5, 18, 32, 50, 100, 150, 186, 199, 200]  # 10个tokens

    # 1. 提取特征
    train_feats = np.random.randn(N, 201, D).astype(np.float32)
    print(f"\n1. 提取训练特征: {train_feats.shape}")

    # 2. 选择 tokens
    train_selected = select_tokens(train_feats, token_indices)
    print(f"2. select_tokens 后: {train_selected.shape}")

    # 3. 转为 Tensor 并训练
    X_train = torch.FloatTensor(train_selected)
    print(f"3. 转为 Tensor: {X_train.shape}, 维度: {X_train.dim()}D")

    # 4. 创建模型（注意：input_dim 是特征维度 D，不是 token 数量）
    model = LinearProbe(input_dim=D, num_classes=2)
    print(f"4. 创建模型，input_dim={D}")

    # 5. 前向传播
    print(f"\n5. 训练时的前向传播:")
    model.eval()
    with torch.no_grad():
        outputs = model(X_train)

    print(f"\n✓ 训练流程完成\n")
    return model, token_indices


def simulate_testing_flow(model, token_indices):
    """模拟测试流程"""
    print("="*70)
    print("测试流程（修复后）")
    print("="*70)

    N = 50
    D = 512

    # 1. 提取特征
    test_feats = np.random.randn(N, 201, D).astype(np.float32)
    print(f"\n1. 提取测试特征: {test_feats.shape}")

    # 2. 选择相同的 tokens
    test_selected = select_tokens(test_feats, token_indices)
    print(f"2. select_tokens 后: {test_selected.shape}")

    # 3. 转为 Tensor
    X_test = torch.FloatTensor(test_selected)
    print(f"3. 转为 Tensor: {X_test.shape}, 维度: {X_test.dim()}D")

    # 4. 评估（模拟 evaluate_model）
    print(f"\n4. 测试时的前向传播:")
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)

    print(f"\n✓ 测试流程完成\n")


def compare_with_old_approach():
    """对比 UniversalFakeDetect 的方法"""
    print("="*70)
    print("对比：UniversalFakeDetect 的方法（先池化）")
    print("="*70)

    N = 50
    D = 512
    token_indices = [0, 5, 18, 32, 50, 100, 150, 186, 199, 200]

    # 1. 提取特征
    test_feats = np.random.randn(N, 201, D).astype(np.float32)
    print(f"\n1. 提取测试特征: {test_feats.shape}")

    # 2. 选择 tokens 并池化（手动）
    selected = test_feats[:, token_indices, :]  # [N, 10, D]
    print(f"2. 选择 tokens: {selected.shape}")

    pooled = selected.mean(axis=1)  # [N, D] ← 手动池化
    print(f"3. 手动池化: {pooled.shape}")

    # 3. 转为 Tensor
    X_test = torch.FloatTensor(pooled)
    print(f"4. 转为 Tensor: {X_test.shape}, 维度: {X_test.dim()}D")

    # 4. LinearProbe（没有内置池化）
    class SimpleLinearProbe(nn.Module):
        def __init__(self, input_dim, num_classes=2):
            super().__init__()
            self.fc = nn.Linear(input_dim, num_classes)

        def forward(self, x):
            print(f"  [SimpleLinearProbe.forward] 输入 shape: {x.shape}")
            out = self.fc(x)
            print(f"  [SimpleLinearProbe.forward] 输出 shape: {out.shape}")
            return out

    model = SimpleLinearProbe(input_dim=D)
    print(f"\n5. 前向传播（无内置池化）:")
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)

    print(f"\n✓ 旧方法完成\n")


def test_equivalence():
    """验证两种方法的等价性"""
    print("="*70)
    print("验证：两种方法是否等价？")
    print("="*70)

    N = 10
    D = 512
    token_indices = [0, 5, 18, 32, 50]

    features = np.random.randn(N, 201, D).astype(np.float32)

    # 方法1：FGTS（select + 模型内池化）
    selected_3d = features[:, token_indices, :]
    X1 = torch.FloatTensor(selected_3d)  # [N, 5, D]
    pooled_by_model = X1.mean(dim=1)     # 模拟模型内池化

    # 方法2：UniversalFakeDetect（先手动池化）
    selected_2d = features[:, token_indices, :].mean(axis=1)
    X2 = torch.FloatTensor(selected_2d)  # [N, D]

    print(f"\n方法1（FGTS）:")
    print(f"  select_tokens: {selected_3d.shape}")
    print(f"  转 Tensor: {X1.shape}")
    print(f"  模型内池化: {pooled_by_model.shape}")

    print(f"\n方法2（UniversalFakeDetect）:")
    print(f"  select + pool: {selected_2d.shape}")
    print(f"  转 Tensor: {X2.shape}")

    # 比较结果
    diff = torch.abs(pooled_by_model - X2).max().item()
    print(f"\n结果差异（最大绝对差）: {diff:.10f}")

    if diff < 1e-6:
        print("✓✓✓ 两种方法完全等价！")
    else:
        print("✗✗✗ 两种方法有差异！")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("FGTS 池化流程调试")
    print("="*70 + "\n")

    # 1. 模拟训练
    model, token_indices = simulate_training_flow()

    # 2. 模拟测试
    simulate_testing_flow(model, token_indices)

    # 3. 对比旧方法
    compare_with_old_approach()

    # 4. 验证等价性
    test_equivalence()

    print("\n" + "="*70)
    print("结论")
    print("="*70)
    print("""
1. FGTS 的实现（修复后）:
   - select_tokens() 返回 [N, K, D]
   - evaluate_model() 直接传给模型（不池化）
   - LinearProbe.forward() 内部检测到 3D 后自动池化
   - ✓ 训练和测试流程一致

2. UniversalFakeDetect 的实现:
   - select_tokens + pool_tokens 手动池化成 [N, D]
   - eval_probe() 接收已池化的 2D 特征
   - LinearProbe.forward() 直接处理 2D 特征（无池化逻辑）
   - ✓ 训练和测试流程一致

3. 数学上两种方法完全等价！
   - 都是对选中的 K 个 tokens 取平均
   - 只是池化的位置不同（函数内 vs 模型内）

4. 如果性能仍有差异，可能的原因：
   - 数据加载/预处理不一致
   - 训练超参数不同
   - 随机种子不同
   - 模型初始化不同
    """)
