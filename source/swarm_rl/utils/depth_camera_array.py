# depth_camera_array.py
from __future__ import annotations

from dataclasses import MISSING
from typing import Any, Literal, Sequence

import torch

# ---------------------------
# IsaacLab import compatibility
# ---------------------------
from isaaclab.utils import configclass
from isaaclab.sensors import RayCasterCameraCfg
from isaaclab.sensors.ray_caster import patterns


# ============================================================
# Configs
# ============================================================

@configclass
class DepthCameraItemCfg:
    """One camera in the array. Mandatory extrinsics, optional name + per-camera overrides."""

    # optional
    name: str | None = None

    # required
    pos_BC: tuple[float, float, float] = MISSING
    quat_BC: tuple[float, float, float, float] = MISSING  # (w, x, y, z)

    # ---- per-camera overrides (optional; if None -> use global)
    resolution: tuple[int, int] | None = None  # (W, H)
    K: list[float] | tuple[float, ...] | list[list[float]] | tuple[tuple[float, ...], ...] | torch.Tensor | None = None  # 3x3 or flat-9
    max_distance: float | None = None
    data_type: str | None = None

    prim_path: str | None = None
    mesh_prim_paths: list[str] | None = None
    depth_clipping_behavior: Literal["max", "zero", "none"] | None = None
    update_period: float | None = None
    debug_vis: bool | None = None


@configclass
class DepthCameraArrayCfg:

    cameras: list[DepthCameraItemCfg] = MISSING

    # defaults (global)
    resolution: tuple[int, int] = (1, 1)        # (W, H)
    K: Any = MISSING                            # 单个K：3x3 / flat-9 / Tensor(9)
    max_distance: float = 4.0
    data_type: str = "distance_to_image_plane"

    prim_path: str | None = None
    mesh_prim_paths: list[str] | None = None
    depth_clipping_behavior: Literal["max", "zero", "none"] = "max"
    update_period: float = 0.0
    debug_vis: bool = False

    offset_convention: Literal["opengl", "ros", "world"] = "world"
    usd_focal_length: float = 24.0

    # postprocess (global scalar)
    invalid_rate_max: float = 0.0
    invalid_sampling: Literal["per_frame", "per_env", "per_pixel"] = "per_frame"
    invalid_fill_value: float | Literal["max_distance"] = "max_distance"
    clamp_to_max_distance: bool = True
    normalize: Literal["none", "0_1", "-1_1"] = "none"
    flatten: bool = False

    @configclass
    class ConcatCfg:
        enable: bool = False
        axis: Literal["width", "height"] = "width"
        order: list[str] | list[int] | None = None
        require_same_resolution: bool = True

    concat: ConcatCfg = ConcatCfg()


# ============================================================
# Runtime wrapper
# ============================================================

class DepthCameraArray:
    """
    Runtime wrapper over N RayCasterCamera-like objects.

    Usage model:
      - In __init__, it can auto-build RayCasterCameraCfgs from DepthCameraArrayCfg, then instantiate sensors via cfg.class_type(cfg).
      - Then you can export or register them into scene.sensors.

    Expected camera object interface:
      - camera.data.output[data_type] -> torch.Tensor with shape (num_envs, H, W, 1) or (num_envs, H, W)
    """

    def __init__(
        self,
        cfg: DepthCameraArrayCfg,
        device: torch.device,
        *,
        camera_cfg_cls: type = RayCasterCameraCfg,
    ):
        self.cfg = cfg
        self.device = device

        # finalize names & name->index
        self.names: list[str] = []
        self.name_to_index: dict[str, int] = {}
        for i, item in enumerate(self.cfg.cameras):
            nm = item.name if item.name else f"cam_{i}"
            if nm in self.name_to_index:
                raise ValueError(f"Duplicate camera name: '{nm}'")
            self.names.append(nm)
            self.name_to_index[nm] = i

        # resolve per-camera effective params once
        self._effective = [self._resolve_effective_for_camera(i) for i in range(len(self.cfg.cameras))]

        # build RayCasterCameraCfg list + keys
        self._sensor_keys, self._camera_cfgs = self._build_camera_cfgs(
            camera_cfg_cls=camera_cfg_cls,
        )

        # instantiate cameras in init
        self.cameras = [cam_cfg.class_type(cam_cfg) for cam_cfg in self._camera_cfgs]


        if len(self.cameras) != len(self.cfg.cameras):
            raise ValueError(
                f"len(cameras)={len(self.cameras)} must match len(cfg.cameras)={len(self.cfg.cameras)}"
            )

    # ============================================================
    # Scene registration helpers
    # ============================================================

    def export_instances(self) -> tuple[list[str], list[Any]]:
        """Return (sensor_keys, camera_instances) aligned with cfg.cameras order."""
        return list(self._sensor_keys), list(self.cameras)

    def register_to_scene(self, scene: Any) -> None:
        """Register instantiated cameras into scene.sensors using generated keys."""
        keys, cams = self.export_instances()
        for k, cam in zip(keys, cams):
            scene.sensors[k] = cam

    def reload_cameras(self) -> None:
        """Request mesh reload for all cameras."""
        keys, cams = self.export_instances()
        # Request mesh reload for all cameras
        for k, cam in zip(keys, cams):
            if hasattr(cam, 'request_mesh_reload'):
                cam.request_mesh_reload()
                print(f"Requested mesh reload for {k}")
            else:
                print(f"Warning: {k} does not support mesh reloading")

    # ============================================================
    # Public API
    # ============================================================

    def read(
        self,
        i_or_name: int | str,
        *,
        env_ids: Sequence[int] | torch.Tensor | None = None,
        stage: Literal["raw", "processed"] = "raw",
    ) -> torch.Tensor:
        idx = self._to_index(i_or_name)
        return self._read_one(idx, env_ids=env_ids, stage=stage)

    def read_batch(
        self,
        indices_or_names: Sequence[int | str] | None = None,
        *,
        env_ids: Sequence[int] | torch.Tensor | None = None,
        stage: Literal["raw", "processed"] = "raw",
        layout: Literal["list", "stack", "concat"] = "list",
        order: list[str] | list[int] | None = None,
        axis: Literal["width", "height"] | None = None,
    ) -> Any:
        indices = self._select_indices(indices_or_names)

        imgs = [self._read_one(i, env_ids=env_ids, stage=stage) for i in indices]

        if layout == "list":
            return imgs

        if layout == "stack":
            # (N_env, N_cam, H, W)
            return torch.stack(imgs, dim=1)

        if layout == "concat":
            return self._concat(imgs, indices=indices, order=order, axis=axis)

        raise ValueError(f"Unknown layout: {layout}")

    def read_obs(
        self,
        *,
        env_ids: Sequence[int] | torch.Tensor | None = None,
        indices_or_names: Sequence[int | str] | None = None,
        layout: Literal["list", "stack", "concat"] | None = None,
        flatten: bool | None = None,
        normalize: Literal["none", "0_1", "-1_1"] | None = None,
        order: list[str] | list[int] | None = None,
        axis: Literal["width", "height"] | None = None,
    ) -> Any:
        """
        Thin wrapper for RL observations.

        Defaults:
          - stage="processed"
          - layout="stack" (unless caller uses concat explicitly)
          - flatten uses cfg.flatten unless overridden
          - normalize: temporarily override cfg.normalize (保持你原行为)
        """
        if layout is None:
            layout = "stack"
        if flatten is None:
            flatten = bool(self.cfg.flatten)
        if normalize is not None:
            old_norm = self.cfg.normalize
            self.cfg.normalize = normalize

        out = self.read_batch(
            indices_or_names,
            env_ids=env_ids,
            stage="processed",
            layout=layout,
            order=order,
            axis=axis,
        )

        out = self._apply_flatten(out, flatten=flatten)

        if normalize is not None:
            self.cfg.normalize = old_norm
        return out

    # ============================================================
    # Internals
    # ============================================================

    def _to_index(self, i_or_name: int | str) -> int:
        if isinstance(i_or_name, int):
            if not (0 <= i_or_name < len(self.cameras)):
                raise IndexError(f"Camera index out of range: {i_or_name}")
            return i_or_name
        if i_or_name not in self.name_to_index:
            raise KeyError(f"Unknown camera name '{i_or_name}'. Known: {self.names}")
        return self.name_to_index[i_or_name]

    def _select_indices(self, indices_or_names: Sequence[int | str] | None) -> list[int]:
        if indices_or_names is None:
            return list(range(len(self.cameras)))
        return [self._to_index(x) for x in indices_or_names]

    @staticmethod
    def _is_number(x: Any) -> bool:
        return isinstance(x, (int, float)) and not isinstance(x, bool)

    def _parse_K_flat9(self, K: Any) -> list[float]:
        """
        Accept:
          - flat-9 list/tuple
          - 3x3 nested
          - torch.Tensor (3x3 or flat-9)
        Return:
          - python list[float] length 9, row-major
        """
        if isinstance(K, torch.Tensor):
            Kt = K.detach().cpu().flatten().tolist()
            if len(Kt) != 9:
                raise ValueError("K tensor must be 3x3 or flat-9 (numel==9).")
            return [float(x) for x in Kt]

        if isinstance(K, (list, tuple)):
            # 3x3
            if len(K) == 3 and all(isinstance(r, (list, tuple)) and len(r) == 3 for r in K):
                return [float(K[r][c]) for r in range(3) for c in range(3)]
            # flat-9
            if len(K) == 9 and all(self._is_number(x) for x in K):
                return [float(x) for x in K]

        raise ValueError("Unsupported K format. Provide 3x3, flat-9, or Tensor(numel==9).")

    def _resolve_effective_for_camera(self, i: int) -> dict[str, Any]:
        item = self.cfg.cameras[i]

        # ---- override > global
        res = item.resolution if item.resolution is not None else self.cfg.resolution

        if self.cfg.K is MISSING and item.K is None:
            raise ValueError("cfg.K must be set (or set per-camera item.K).")
        K_any = item.K if item.K is not None else self.cfg.K

        max_d = item.max_distance if item.max_distance is not None else self.cfg.max_distance
        dtp = item.data_type if item.data_type is not None else self.cfg.data_type
        clip = item.depth_clipping_behavior if item.depth_clipping_behavior is not None else self.cfg.depth_clipping_behavior
        upd = item.update_period if item.update_period is not None else self.cfg.update_period
        dbg = item.debug_vis if item.debug_vis is not None else self.cfg.debug_vis
        
        # ---- sensor key (用于 scene.sensors 注册名 & 默认 prim 子路径)
        sensor_key = f"cam_{item.name}" if item.name else f"cam_{i}"

        prim_path = item.prim_path if item.prim_path is not None else self.cfg.prim_path
        mesh_paths = item.mesh_prim_paths if item.mesh_prim_paths is not None else self.cfg.mesh_prim_paths

        # ---- validation: resolution
        if not (isinstance(res, tuple) and len(res) == 2):
            raise ValueError(f"resolution for camera[{i}] must be (W,H). Got: {res}")
        W, H = int(res[0]), int(res[1])
        if W <= 0 or H <= 0:
            raise ValueError(f"resolution for camera[{i}] must be positive. Got: {res}")

        # ---- K -> flat9
        K_flat = self._parse_K_flat9(K_any)

        # ---- sanity: K vs resolution
        fx, fy = K_flat[0], K_flat[4]
        cx, cy = K_flat[2], K_flat[5]
        if fx <= 0 or fy <= 0:
            raise ValueError(f"K has non-positive focal(s) for camera[{i}]. fx={fx}, fy={fy}")
        if not (0.0 <= cx < float(W) and 0.0 <= cy < float(H)):
            raise ValueError(f"K principal point out of bounds for camera[{i}]. cx={cx}, cy={cy}, W={W}, H={H}")

        # ---- quat normalize (wxyz)
        qw, qx, qy, qz = item.quat_BC
        q = torch.tensor([qw, qx, qy, qz], dtype=torch.float32)
        n = torch.linalg.vector_norm(q).item()
        if n <= 1e-12:
            raise ValueError(f"quat_BC norm too small for camera[{i}]")
        q = (q / n).tolist()
        quat_normed = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))

        return {
            "name": self.names[i],
            "pos_BC": item.pos_BC,
            "quat_BC": quat_normed,
            "resolution": (W, H),
            "K_flat": K_flat,
            "max_distance": float(max_d),
            "data_type": str(dtp),
            "depth_clipping_behavior": clip,
            "update_period": float(upd),
            "debug_vis": bool(dbg),
            "sensor_key": sensor_key,
            "prim_path": prim_path,
            "mesh_prim_paths": mesh_paths,
            "invalid_rate_max": float(self.cfg.invalid_rate_max),
        }

    def _build_camera_cfgs(
        self,
        *,
        camera_cfg_cls: type,
    ) -> tuple[list[str], list[Any]]:
        """
        Build RayCasterCameraCfg list aligned with cfg.cameras order.

        IMPORTANT: prim_path / mesh_prim_paths must be provided (global or per-camera).
        """
        cfgs: list[Any] = []
        keys: list[str] = []
        key_set: set[str] = set()

        for i in range(len(self.cfg.cameras)):
            eff = self._effective[i]
            item = self.cfg.cameras[i]

            if eff["prim_path"] is None:
                raise ValueError("prim_path must be set (top-level or per-camera override) to build camera cfgs.")
            if eff["mesh_prim_paths"] is None:
                raise ValueError("mesh_prim_paths must be set (top-level or per-camera override) to build camera cfgs.")
            
            key = eff["sensor_key"]
            if key in key_set:
                raise ValueError(f"Duplicate sensor key generated: '{key}'. Check camera item names.")
            key_set.add(key)
            keys.append(key)

            W, H = eff["resolution"]
            K_flat = eff["K_flat"]

            pattern_cfg = patterns.PinholeCameraPatternCfg.from_intrinsic_matrix(
                intrinsic_matrix=K_flat,
                width=W,
                height=H,
                focal_length=float(self.cfg.usd_focal_length),
            )

            cam_cfg = camera_cfg_cls(
                prim_path=eff["prim_path"],
                offset=camera_cfg_cls.OffsetCfg(
                    pos=eff["pos_BC"],
                    rot=eff["quat_BC"],
                    convention=self.cfg.offset_convention,
                ),
                mesh_prim_paths=eff["mesh_prim_paths"],
                max_distance=eff["max_distance"],
                depth_clipping_behavior=eff["depth_clipping_behavior"],
                pattern_cfg=pattern_cfg,
                data_types=[eff["data_type"]],
                update_period=eff["update_period"],
                debug_vis=eff["debug_vis"],
            )
            cfgs.append(cam_cfg)

        return keys, cfgs

    def _read_one(
        self,
        idx: int,
        *,
        env_ids: Sequence[int] | torch.Tensor | None,
        stage: Literal["raw", "processed"],
    ) -> torch.Tensor:
        eff = self._effective[idx]
        cam = self.cameras[idx]
        data_type = eff["data_type"]
        max_d = eff["max_distance"]

        img = cam.data.output[data_type]  # expected torch.Tensor
        if img.dim() == 4 and img.shape[-1] == 1:
            img = img[..., 0]
        if img.dim() != 3:
            raise RuntimeError(
                f"Camera[{idx}] output '{data_type}' must be (N,H,W[,1]). Got shape={tuple(img.shape)}"
            )

        if env_ids is not None:
            # keep your original semantics (supports slice/list/tensor)
            img = img[env_ids]

        if stage == "raw":
            return img

        x = img.clone()
        x = torch.nan_to_num(x, nan=max_d, posinf=max_d, neginf=0.0)

        inv_max = float(eff["invalid_rate_max"])
        if inv_max > 0.0:
            if self.cfg.invalid_sampling == "per_frame":
                invalid_rate = torch.rand((), device=x.device).item() * inv_max
                mask = torch.rand_like(x) < invalid_rate
            elif self.cfg.invalid_sampling == "per_env":
                invalid_rate = torch.rand((x.shape[0], 1, 1), device=x.device) * inv_max
                mask = torch.rand_like(x) < invalid_rate
            elif self.cfg.invalid_sampling == "per_pixel":
                invalid_rate = torch.rand_like(x) * inv_max
                mask = torch.rand_like(x) < invalid_rate
            else:
                raise ValueError(f"Unknown invalid_sampling: {self.cfg.invalid_sampling}")

            if self.cfg.invalid_fill_value == "max_distance":
                fill = torch.tensor(max_d, device=x.device, dtype=x.dtype)
            else:
                fill = torch.tensor(float(self.cfg.invalid_fill_value), device=x.device, dtype=x.dtype)

            x = torch.where(mask, fill, x)

        if self.cfg.clamp_to_max_distance:
            x = torch.clamp(x, 0.0, max_d)

        if self.cfg.normalize == "0_1":
            x = (x / max_d).clamp(0.0, 1.0)
        elif self.cfg.normalize == "-1_1":
            x = (x / max_d).clamp(0.0, 1.0) * 2.0 - 1.0
        elif self.cfg.normalize == "none":
            pass
        else:
            raise ValueError(f"Unknown normalize: {self.cfg.normalize}")

        return x

    def _concat(
        self,
        imgs: list[torch.Tensor],
        *,
        indices: list[int],
        order: list[str] | list[int] | None,
        axis: Literal["width", "height"] | None,
    ) -> torch.Tensor:
        if order is None:
            order = self.cfg.concat.order
        if axis is None:
            axis = self.cfg.concat.axis

        if order is None:
            ordered_imgs = imgs
        else:
            if all(isinstance(x, str) for x in order):
                ord_indices = [self._to_index(x) for x in order]
            else:
                ord_indices = [int(x) for x in order]

            selected_set = set(indices)
            ord_indices = [i for i in ord_indices if i in selected_set]
            if len(ord_indices) != len(indices):
                raise ValueError("concat.order does not match the selected camera set.")

            idx_map = {abs_i: local_k for local_k, abs_i in enumerate(indices)}
            ordered_imgs = [imgs[idx_map[abs_i]] for abs_i in ord_indices]

        if self.cfg.concat.require_same_resolution:
            H0, W0 = ordered_imgs[0].shape[1], ordered_imgs[0].shape[2]
            for k, im in enumerate(ordered_imgs):
                if im.shape[1] != H0 or im.shape[2] != W0:
                    raise ValueError(
                        f"concat requires same resolution, but got cam[{k}] shape={tuple(im.shape)} vs first={tuple(ordered_imgs[0].shape)}"
                    )

        if axis == "width":
            return torch.cat(ordered_imgs, dim=2)
        if axis == "height":
            return torch.cat(ordered_imgs, dim=1)
        raise ValueError(f"Unknown concat axis: {axis}")

    def _apply_flatten(self, out: Any, *, flatten: bool) -> Any:
        if not flatten:
            return out
        if isinstance(out, list):
            return [x.reshape(x.shape[0], -1) for x in out]
        if torch.is_tensor(out):
            return out.reshape(out.shape[0], -1)
        return out
