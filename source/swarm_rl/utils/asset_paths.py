from __future__ import annotations

from importlib.resources import files


def get_dji_fpv_usd_path() -> str:
    return str(files("swarm_rl").joinpath("assets/dji_fpv/v1/dji_fpv.usd"))


def get_crazyflie_usd_path() -> str:
    return str(files("swarm_rl").joinpath("assets/e2e_drone/Assets/Isaac/4.5/Isaac/Robots/Crazyflie/cf2x.usd"))


def get_ui_arrow_usd_path() -> str:
    return str(files("swarm_rl").joinpath("assets/e2e_drone/Assets/Isaac/4.5/Isaac/Props/UIElements/arrow_x.usd"))


def get_ui_frame_usd_path() -> str:
    return str(files("swarm_rl").joinpath("assets/e2e_drone/Assets/Isaac/4.5/Isaac/Props/UIElements/frame_prim.usd"))