"""
Generate portfolio charts for README.

Delegates to individual chart scripts in reports/charts/.  Each script is
self-contained and can be run directly; this wrapper re-runs them all.

Charts produced:
  reports/charts/leaderboard_progression.png  — private LB over time with phase annotations
  reports/charts/per_category_journey.png     — FOODS/HOUSEHOLD/HOBBIES across key methods
  reports/charts/blend_dynamics.png           — when blending helped vs hurt
  reports/charts/hierarchical_aggregation.png — top-down vs bottom-up Prophet
"""
import os
import runpy

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHARTS    = os.path.join(PROJ_ROOT, "reports", "charts")

for script in [
    "leaderboard_progression.py",
    "per_category_journey.py",
    "blend_dynamics.py",
    "hierarchical_aggregation.py",
]:
    runpy.run_path(os.path.join(CHARTS, script), run_name="__main__")

print("\nAll 4 charts regenerated.")
