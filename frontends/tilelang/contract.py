"""
TileLang contract evaluation (MVP).

Contract V2 is frontend-agnostic, so TileLang reuses the common implementation.
"""

from frontends.common.contract_v2 import ContractReport, evaluate_contract_v2

__all__ = ["ContractReport", "evaluate_contract_v2"]

