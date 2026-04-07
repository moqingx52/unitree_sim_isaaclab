# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""向后兼容：最小 Omnipicker 配置已迁至 g1_omnipicker_cfg.py。"""

from robots.g1_omnipicker_cfg import (
    G1_OMNIPICKER_MINIMAL_CFG,
    G1_OMNIPICKER_USD_PATH,
    replace_usd_path,
)

__all__ = ["G1_OMNIPICKER_MINIMAL_CFG", "G1_OMNIPICKER_USD_PATH", "replace_usd_path"]
