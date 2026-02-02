import torch
from torch.testing import assert_close

def apply_inplace(actions: torch.Tensor, buf: torch.Tensor) -> torch.Tensor:
    """复现你那三行逻辑：buf <- actions; tanh; thrust维(0)映射到[0,1]."""
    buf.copy_(actions)
    buf.tanh_()
    buf[:, 0].add_(1.0).mul_(0.5)
    return buf

def test_action_postprocess_correctness(device: str = "cpu"):
    print(f"\n########## Start test, device={device} ##########")
    # 构造覆盖边界的输入：大正/大负/0，确保 tanh 饱和与线性区都测到
    actions = torch.tensor(
        [
            [0.0,   0.0,   0.0,  0.0],
            [10.0,  1.0,  -1.0,  0.5],
            [-10.0, -2.0,  2.0, -0.5],
            [0.2,   -0.3,  0.4, -0.9],
        ],
        dtype=torch.float32,
        device=device,
    )

    actions_before = actions.clone()
    buf = torch.empty_like(actions)

    out = apply_inplace(actions, buf)

    # 期望值：先 tanh 全维到 [-1,1]，再仅第0维做 [-1,1]->[0,1]
    expected = torch.tanh(actions_before)
    expected[:, 0] = (expected[:, 0] + 1.0) * 0.5

    # 1) 数值等价
    assert_close(out, expected, rtol=0, atol=1e-6)

    # 2) 输入 actions 不应被修改（因为用的是 copy_ 到 buf）
    assert_close(actions, actions_before, rtol=0, atol=0)

    # 3) 范围检查：第0维在 [0,1]；其余维在 [-1,1]
    assert torch.all(out[:, 0] >= 0.0) and torch.all(out[:, 0] <= 1.0)
    if out.shape[1] > 1:
        assert torch.all(out[:, 1:] >= -1.0) and torch.all(out[:, 1:] <= 1.0)

    print(out)

if __name__ == "__main__":
    test_action_postprocess_correctness("cpu")
    # 如果你想顺带测 GPU：
    if torch.cuda.is_available():
        test_action_postprocess_correctness("cuda")
    print("\n########## Test pass ##########\n")
