import math
import torch


def compute_goal_feat(desired_pos_w: torch.Tensor, pos_w: torch.Tensor) -> torch.Tensor:
    """
    复现你当前的实现：
      dir_xy = err_xy / ||err_xy||_2.clamp_min(eps)
      goal_feat = [dir_x, dir_y, err_z]
    输入:
      desired_pos_w: (N, 3)
      pos_w:         (N, 3)
    输出:
      goal_feat:     (N, 3)
    """
    relative_pos_to_goal = desired_pos_w - pos_w
    err_xy = relative_pos_to_goal[:, :2]

    eps = 1e-6
    if err_xy.dtype in (torch.float16, torch.bfloat16):
        eps = 1e-3

    denom = torch.linalg.vector_norm(err_xy, dim=1, keepdim=True).clamp_min(eps)
    dir_xy = err_xy / denom

    err_z = relative_pos_to_goal[:, 2]
    goal_feat = torch.cat([dir_xy, err_z.unsqueeze(-1)], dim=1)
    return goal_feat


def _assert_no_nan_inf(x: torch.Tensor, name: str):
    if not torch.isfinite(x).all().item():
        bad = x[~torch.isfinite(x)]
        raise AssertionError(f"{name} contains NaN/Inf. Example bad values: {bad[:10]}")


def _assert_close(a: torch.Tensor, b: torch.Tensor, atol: float, rtol: float, msg: str):
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol, msg=msg)


def test_known_cases(device: torch.device, dtype: torch.dtype):
    # Case 1
    desired = torch.tensor([[1.0, 0.0, 2.0]], device=device, dtype=dtype)
    pos = torch.tensor([[0.0, 0.0, 0.0]], device=device, dtype=dtype)
    feat = compute_goal_feat(desired, pos)
    expected = torch.tensor([[1.0, 0.0, 2.0]], device=device, dtype=dtype)
    _assert_close(feat, expected, atol=1e-4, rtol=1e-4, msg="known case 1 failed")

    # Case 2: (-3,4) -> norm 5 => (-0.6, 0.8)
    desired = torch.tensor([[-3.0, 4.0, -1.0]], device=device, dtype=dtype)
    pos = torch.tensor([[0.0, 0.0, 0.0]], device=device, dtype=dtype)
    feat = compute_goal_feat(desired, pos)
    expected = torch.tensor([[-0.6, 0.8, -1.0]], device=device, dtype=dtype)
    _assert_close(feat, expected, atol=2e-3, rtol=2e-3, msg="known case 2 failed")

    # Case 3: err_xy=0 => dir_xy should be (0,0), err_z preserved
    desired = torch.tensor([[0.0, 0.0, 5.0]], device=device, dtype=dtype)
    pos = torch.tensor([[0.0, 0.0, 0.0]], device=device, dtype=dtype)
    feat = compute_goal_feat(desired, pos)
    expected = torch.tensor([[0.0, 0.0, 5.0]], device=device, dtype=dtype)
    _assert_close(feat, expected, atol=1e-4, rtol=1e-4, msg="known case 3 failed")

    _assert_no_nan_inf(feat, "goal_feat (known cases)")


def test_random_no_nan_and_norm_properties(device: torch.device, dtype: torch.dtype):
    N = 4096
    desired = torch.randn(N, 3, device=device, dtype=dtype)
    pos = torch.randn(N, 3, device=device, dtype=dtype)

    rel = desired - pos
    err_xy = rel[:, :2]

    eps = 1e-6
    if dtype in (torch.float16, torch.bfloat16):
        eps = 1e-3

    feat = compute_goal_feat(desired, pos)
    _assert_no_nan_inf(feat, "goal_feat (random)")

    dir_xy = feat[:, :2]
    norm_raw = torch.linalg.vector_norm(err_xy, dim=1)            # (N,)
    norm_dir = torch.linalg.vector_norm(dir_xy, dim=1)            # (N,)

    # 1) norm_dir should never exceed 1 (因为 denom >= norm_raw)
    # 允许的最大超出：按 dtype 的 eps 来
    max_allow = 1.0 + 4.0 * torch.finfo(dtype).eps  # fp16: 1 + 4*0.000976.. = 1.0039
    if (norm_dir > max_allow).any().item():
        mx = norm_dir.max().item()
        raise AssertionError(f"||dir_xy|| should be <= {max_allow}, but max is {mx}")


    # 2) 当 norm_raw > eps 时，dir_xy 应该接近单位向量 => ||dir_xy||≈1
    mask_far = norm_raw > (10.0 * eps)  # 给一点余量，避免边界数值效应
    if mask_far.any().item():
        mean_norm = norm_dir[mask_far].mean().item()
        # 容忍度按 dtype 调整
        tol = 2e-3 if dtype in (torch.float16, torch.bfloat16) else 2e-4
        if abs(mean_norm - 1.0) > tol:
            raise AssertionError(f"Mean ||dir_xy|| for far samples should be ~1, got {mean_norm}")

    # 3) 当 norm_raw 很小时，||dir_xy|| 应该显著 < 1（因为被 eps 截断）
    mask_near = norm_raw < (0.1 * eps)
    if mask_near.any().item():
        max_near = norm_dir[mask_near].max().item()
        if max_near > 0.2:  # 经验阈值：0.1*eps / eps = 0.1，考虑数值误差给到 0.2
            raise AssertionError(f"Near samples expected small ||dir_xy||, but max is {max_near}")


def test_fp16_bf16_eps_branch_behavior():
    """
    验证 fp16/bf16 分支把 eps 提到 1e-3 是否真的起作用。
    需要 CUDA（因为 fp16 在 CPU 上很多算子支持不完整/不稳定）。
    """
    if not torch.cuda.is_available():
        print("[SKIP] fp16/bf16 eps-branch test: CUDA not available")
        return

    device = torch.device("cuda")

    for dtype in (torch.float16, torch.bfloat16):
        # 构造一个非常小的 err_xy：如果 eps=1e-3，则 dir_xy 的模长约为 0.1
        # err_xy = (1e-4, 0) => norm=1e-4，denom=clamp_min(1e-3)=1e-3 => dir=(0.1,0)
        desired = torch.tensor([[1e-4, 0.0, 0.0]], device=device, dtype=dtype)
        pos = torch.zeros_like(desired)

        feat = compute_goal_feat(desired, pos)
        dir_xy = feat[0, :2].float()  # 转 float32 比较更稳
        norm_dir = torch.linalg.vector_norm(dir_xy).item()

        if not (0.05 <= norm_dir <= 0.2):
            raise AssertionError(
                f"{dtype}: expected ||dir_xy|| ~ 0.1 due to eps=1e-3, got {norm_dir}. "
                f"(If eps was tiny, this would be ~1.)"
            )

    print("[OK] fp16/bf16 eps-branch behavior looks correct")


def test_gradients_finite(device: torch.device, dtype: torch.dtype):
    """
    可选：检查反向传播不会产生 NaN/Inf（如果你后续对这个特征做可微运算/学习模块会有用）。
    """
    N = 1024
    desired = torch.randn(N, 3, device=device, dtype=dtype, requires_grad=True)
    pos = torch.randn(N, 3, device=device, dtype=dtype, requires_grad=True)

    feat = compute_goal_feat(desired, pos)
    loss = (feat ** 2).mean()
    loss.backward()

    _assert_no_nan_inf(desired.grad, "desired_pos_w.grad")
    _assert_no_nan_inf(pos.grad, "pos_w.grad")


def run_all_tests():
    # 优先在 CUDA 上测（更贴近你的训练场景）
    devices = [torch.device("cuda")] if torch.cuda.is_available() else []
    devices.append(torch.device("cpu"))

    for device in devices:
        # CPU 上不强测 float16（很多算子支持不一致），float32/bfloat16 足够覆盖
        dtypes = [torch.float32]
        if device.type == "cuda":
            dtypes += [torch.float16, torch.bfloat16]
        else:
            # CPU bfloat16 通常可用；如果你环境不支持可以删掉
            dtypes += [torch.bfloat16]

        for dtype in dtypes:
            print(f"[RUN] device={device.type}, dtype={dtype}")
            test_known_cases(device, dtype)
            test_random_no_nan_and_norm_properties(device, dtype)
            test_gradients_finite(device, dtype)
            print(f"[OK ] device={device.type}, dtype={dtype}")

    test_fp16_bf16_eps_branch_behavior()
    print("All tests passed ✅")


if __name__ == "__main__":
    run_all_tests()
