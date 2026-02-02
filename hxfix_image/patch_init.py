path = "Exeter_CFD_Problems/__init__.py"
content = """# patched: HeatExchanger-only (avoid PyFoam crash from other problems)
from .heatexchanger import HeatExchanger
__all__ = ["HeatExchanger"]
"""
open(path, "w", encoding="utf-8").write(content)
print("patched", path)
