# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="FAST-Single-Bodyrate",
    entry_point=f"{__name__}.single_bodyrate_env:QuadcopterEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.single_bodyrate_env_cfg:QuadcopterEnvCfg",

        "skrl_ppo_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)