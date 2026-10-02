"""零依赖门禁：核心算法层不许 import 任何第三方包。

这是这个项目最重要的一条 CI 规则，理由写在 README 里：
"零第三方依赖"是这个项目的**题目本身**（要量化的一个指标就是内存，
而用了 numpy 之后内存就说不清了）。如果只是写在文档里，
一次无心的 import 就会让它悄悄失效 ——
而恰恰是这种失效最难发现：代码照跑、测试照绿。

所以这里做两件事：

**1. 静态 AST 扫描**：看顶层 ``import X`` 的 X 是否在标准库名单里。
    快，能给出精确到行的报错位置。

**2. 动态导入**：在子进程里装一个 meta-path finder，
    把所有非标准库的顶层模块都拦下来，然后真的 ``import_module``。
    静态扫描漏掉的间接依赖（比如某个包被另一个包的 ``__init__`` 拉进来）
    会在这一步暴露。

再加一条**反向校验**：被豁免的模块必须真的依赖第三方。
否则豁免名单会无限膨胀 —— 等到"所有人都被豁免"，
这条规则就名存实亡了。
"""

from __future__ import annotations

import ast
import subprocess
import sys
import sysconfig
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: 核心模块。**这里不许出现 FastAPI / uvicorn / pydantic。**
CORE_MODULES: tuple[str, ...] = (
    "app.metrics",
    "app.dataset",
    "app.bruteforce",
    "app.hnsw",
    "app.ivfpq",
    "app.benchmark",
    "app.registry",
)

#: 允许引入第三方依赖的模块（HTTP 边界）
EXEMPT_MODULES: tuple[str, ...] = ("app.main",)

#: 豁免模块**必须**依赖的第三方包。反向校验用。
EXEMPT_MUST_IMPORT: dict[str, tuple[str, ...]] = {
    "app.main": ("fastapi",),
}

_STDLIB = frozenset(sys.stdlib_module_names)
_STDLIB |= {"__future__", "_thread", "nt", "posix", "os"}
# site-packages 里的 stdlib 前缀要排除：sitecustomize 之类会被算进
# stdlib_module_names，但它们未必真的可用
_LOCAL = frozenset({"app", "app.*"})


def _emit(level: str, module: str, message: str) -> None:
    print(f"  [{level}] {module}: {message}")


def _is_third_party(name: str) -> bool:
    top = name.split(".")[0]
    if top in _STDLIB:
        return False
    if top == "app":
        return False
    return True


# ---------------------------------------------------------------------------
# 1. 静态扫描
# ---------------------------------------------------------------------------

def scan_static(module: str) -> list[str]:
    """扫描一个模块的顶层 import，返回违规列表。"""
    path = ROOT / module.replace(".", "/")
    source_file = path.with_suffix(".py")
    if not source_file.exists():
        path = path / "__init__.py"
        if not path.exists():
            return [f"找不到模块文件 {source_file}"]
    else:
        path = source_file

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    problems: list[str] = []

    def check(node: ast.Import | ast.ImportFrom, depth: int) -> None:
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif node.level and node.level > 0:
            # 相对导入（``from .metrics import X``）：AST 里 node.module 是
            # 去掉点号的裸名字，直接拿它会把 "metrics" 当成顶层第三方包。
            # 相对导入指向的必然是本项目内部模块，不可能是第三方 —— 跳过。
            # 真正的第三方依赖由第 2 步的动态导入检出，不会漏。
            return
        else:
            names = [node.module or ""]
        for name in names:
            if not name:
                continue
            # 函数内的 import 不追究：**可选的**第三方依赖放在调用点
            # 而不是模块顶层，是保持零依赖的正规做法。
            if depth == 0 and _is_third_party(name):
                problems.append(f"第 {node.lineno} 行 import {name}")

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            depth = _node_depth(tree, node)
            check(node, depth)
    return problems


def _node_depth(tree: ast.Module, target: ast.AST) -> int:
    """返回节点在多少层函数/类定义里。0 表示模块顶层。"""
    depth = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for child in ast.walk(node):
                if child is target:
                    return depth + 1
    return 0


# ---------------------------------------------------------------------------
# 2. 动态导入
# ---------------------------------------------------------------------------

_CHILD = r'''
import sys, importlib
target = sys.argv[1]

class Blocker:
    def find_module(self, name, path=None):
        return None

    def find_spec(self, name, path=None, target=None):
        top = name.split(".")[0]
        if top in sys.stdlib_module_names or top in ("app", "__future__"):
            return None
        raise ImportError(f"被拦截：核心层不允许引入第三方包 {name!r}")

sys.meta_path.insert(0, Blocker())
importlib.import_module(target)
print("OK")
'''


def check_dynamic(module: str) -> str | None:
    """在屏蔽第三方包的子进程里真的 import 一次。"""
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, module],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()
        for line in reversed(tail):
            if "Error" in line or "被拦截" in line:
                return line.strip()
        return tail[-1] if tail else f"退出码 {proc.returncode}"
    return None


# ---------------------------------------------------------------------------
# 3. 反向校验豁免名单
# ---------------------------------------------------------------------------

def verify_exemptions() -> list[str]:
    """被豁免的模块必须**真的**依赖第三方，否则豁免是无意义的。"""
    problems: list[str] = []
    for module, must in EXEMPT_MUST_IMPORT.items():
        source = (ROOT / (module.replace(".", "/") + ".py")).read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        for package in must:
            if package not in imported:
                problems.append(
                    f"{module} 被豁免却没有 import {package} —— "
                    "豁免名单不该留着过期的条目"
                )
    return problems


# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 68)
    print("零依赖门禁 · core modules")
    print("=" * 68)

    failures = 0

    print(f"\n[1/3] 静态扫描 {len(CORE_MODULES)} 个核心模块")
    print("-" * 68)
    for module in CORE_MODULES:
        problems = scan_static(module)
        if problems:
            for p in problems:
                _emit("FAIL", module, p)
            failures += 1
        else:
            _emit("ok", module, "顶层 import 全部来自标准库")
        if module in EXEMPT_MUST_IMPORT:
            _emit("WARN", module, "既是核心模块又在豁免名单里，逻辑冲突")

    print(f"\n[2/3] 动态导入验证（拦截第三方包）")
    print("-" * 68)
    for module in CORE_MODULES:
        error = check_dynamic(module)
        if error:
            _emit("FAIL", module, error)
            failures += 1
        else:
            _emit("ok", module, "在纯净环境里可导入")

    print(f"\n[3/3] 反向校验 {len(EXEMPT_MODULES)} 个豁免模块")
    print("-" * 68)
    exempt_problems = verify_exemptions()
    for p in exempt_problems:
        _emit("FAIL", "exemptions", p)
        failures += 1
    if not exempt_problems:
        for module in EXEMPT_MODULES:
            _emit("ok", module, "确实依赖第三方，豁免成立")

    print("\n" + "=" * 68)
    if failures:
        print(f"✗ {failures} 项不合格 —— 核心层必须是零第三方依赖的")
    else:
        print(f"✓ 通过：{len(CORE_MODULES)} 个核心模块零第三方依赖，"
              f"{len(EXEMPT_MODULES)} 个豁免模块理由成立")
    print("=" * 68)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
