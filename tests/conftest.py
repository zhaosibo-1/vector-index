"""pytest 全局配置。

只做两件事：把项目根加入 ``sys.path``（这样测试可以从任何目录跑），
以及让 ``pytest.approx`` 的默认精度显式化 —— 浮点断言不写精度
是这个项目里最容易被"看起来通过"糊弄过去的地方。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def approx():
    """带默认精度的 approx。

    距离计算涉及 float32 存储 + 浮点累加，双精度下相等不一定成立。
    统一用 ``rel=1e-6``（float32 的 eps 约 1.2e-7，留一个数量级余量）。
    """
    return pytest.approx
