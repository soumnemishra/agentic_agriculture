"""
ACA Skills Module
=================

Provides reusable agricultural competencies and the central SkillRegistry:
    - ``BaseSkill``, ``SkillSchema``, ``SkillParameter``, ``SkillResult``
    - ``SkillRegistry``
    - ``TomatoDiagnosisSkill``, ``TOMATO_CLASSES``, ``CondConViT_V2``
"""

from aca.skills.base_skill import (
    BaseSkill,
    SkillParameter,
    SkillResult,
    SkillSchema,
)
from aca.skills.registry import SkillRegistry
from aca.skills.tomato_diagnosis_skill import (
    CondConViT_V2,
    TOMATO_CLASSES,
    TomatoDiagnosisSkill,
)

__all__ = [
    "BaseSkill",
    "SkillParameter",
    "SkillSchema",
    "SkillResult",
    "SkillRegistry",
    "TomatoDiagnosisSkill",
    "TOMATO_CLASSES",
    "CondConViT_V2",
]
