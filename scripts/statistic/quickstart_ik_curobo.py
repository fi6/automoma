#!/usr/bin/env python3
"""
Quick start guide for IK cuRobo num_seeds analysis.

This script demonstrates how to use the new analyze_ik_curobo.py module.
"""

import sys
from pathlib import Path

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "statistic"))

from config import IK_CUROBO, IK_CUROBO_VIZ


def print_config_summary():
    """Print configuration summary."""
    print("\n" + "="*80)
    print("IK cuRobo num_seeds Analysis - Configuration Summary")
    print("="*80)
    
    print("\n📊 Analysis Configuration:")
    print(f"  num_seeds values to test: {IK_CUROBO['num_seeds_values']}")
    print(f"  Reference num_seeds: {IK_CUROBO['reference_num_seeds']}")
    print(f"  Number of scenes: {len(IK_CUROBO['analysis_scenes'])}")
    print(f"  Grasp IDs to analyze: {IK_CUROBO['analysis_grasp_ids']}")
    print(f"  Cache enabled: {IK_CUROBO['enable_cache']}")
    print(f"  Cache directory: {IK_CUROBO['cache_dir']}")
    
    print("\n🎨 Visualization Configuration:")
    print("  Colors and markers for each num_seeds:")
    for ns in IK_CUROBO['num_seeds_values']:
        color = IK_CUROBO_VIZ['colors'].get(ns, 'N/A')
        marker = IK_CUROBO_VIZ['markers'].get(ns, 'N/A')
        size = IK_CUROBO_VIZ['marker_sizes'].get(ns, 'N/A')
        is_ref = " (REFERENCE)" if ns == IK_CUROBO['reference_num_seeds'] else ""
        print(f"    {ns:6d} seeds: color={color:8s}, marker={marker}, size={size}{is_ref}")
    
    print("\n📈 Expected Results:")
    print("  Typical solution counts for summit_franka robot:")
    expected_ratios = {
        1000: 0.25,   # ~1750 solutions
        2000: 0.40,   # ~2800 solutions
        5000: 0.65,   # ~4550 solutions
        10000: 0.85,  # ~5950 solutions
        20000: 1.0,   # ~7000 solutions (reference)
        40000: 1.1,   # ~7700 solutions
    }
    
    base_count = 7000  # Reference for 20000 seeds
    for ns in IK_CUROBO['num_seeds_values']:
        ratio = expected_ratios.get(ns, 1.0)
        count = int(base_count * ratio)
        print(f"    {ns:6d} seeds: ~{count:5d} solutions (ratio: {ratio:.2f}x)")
    
    print("\n💡 Key Insights:")
    print("  • Lower num_seeds = fewer solutions but potentially better spread")
    print("  • Higher num_seeds = more solutions but diminishing returns")
    print("  • Wasserstein distance shows how well solutions cover the distribution")
    print("  • Diversity metrics quantify solution spread in joint space")
    
    print("\n🚀 How to Run:")
    print("  python scripts/statistic/analyze_ik_curobo.py")
    
    print("\n📝 How to Customize:")
    print("  Edit scripts/statistic/config.py:")
    print("    - IK_CUROBO['num_seeds_values']")
    print("    - IK_CUROBO['analysis_scenes']")
    print("    - IK_CUROBO['analysis_grasp_ids']")
    print("    - IK_CUROBO['enable_cache']")
    
    print("\n📂 Output Directory:")
    print(f"  {PROJECT_ROOT / 'output' / 'statistics' / 'figures' / 'ik_curobo'}")
    
    print("\n" + "="*80 + "\n")


def show_usage_examples():
    """Show code examples."""
    print("\n" + "="*80)
    print("Code Examples")
    print("="*80 + "\n")
    
    print("Example 1: Basic usage with default num_seeds (20000)")
    print("-" * 80)
    print("""
from cuakr.planner.planner import AKRPlanner

planner = AKRPlanner(scene_cfg, object_cfg, robot_cfg)

# Default: 20000 seeds, ~7000 solutions
result = planner.plan_ik(
    grasp_pose=grasp_pose,
    start_angle=0.0,
    goal_angle=1.57,
    robot_cfg=robot_cfg
)
print(f"Start IKs: {result.start_ik.shape[0]}")  # ~7000
print(f"Goal IKs: {result.goal_ik.shape[0]}")    # ~7000
""")
    
    print("\nExample 2: Using fewer seeds for faster computation")
    print("-" * 80)
    print("""
# With 5000 seeds, ~4550 solutions
result = planner.plan_ik(
    grasp_pose=grasp_pose,
    start_angle=0.0,
    goal_angle=1.57,
    robot_cfg=robot_cfg,
    num_ik_seeds=5000  # Control solution count here!
)
print(f"Start IKs: {result.start_ik.shape[0]}")  # ~4550
print(f"Goal IKs: {result.goal_ik.shape[0]}")    # ~4550
""")
    
    print("\nExample 3: Using more seeds for comprehensive coverage")
    print("-" * 80)
    print("""
# With 40000 seeds, ~7700 solutions
result = planner.plan_ik(
    grasp_pose=grasp_pose,
    start_angle=0.0,
    goal_angle=1.57,
    robot_cfg=robot_cfg,
    num_ik_seeds=40000  # Get more solutions
)
print(f"Start IKs: {result.start_ik.shape[0]}")  # ~7700
print(f"Goal IKs: {result.goal_ik.shape[0]}")    # ~7700
""")
    
    print("\nExample 4: Running comparative analysis (in analyze_ik_curobo.py)")
    print("-" * 80)
    print("""
# The analyze_ik_curobo.py script automatically:
# 1. Plans IK for each num_seeds value
# 2. Caches results for reuse
# 3. Computes statistics and diversity metrics
# 4. Generates beautiful visualizations
# 5. Computes Wasserstein distances

# Just run: python scripts/statistic/analyze_ik_curobo.py
""")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    print("\n")
    print("   ╔═════════════════════════════════════════════════════════════════════╗")
    print("   ║     IK cuRobo num_seeds Impact Analysis - Quick Start Guide         ║")
    print("   ║                                                                     ║")
    print("   ║     Analyze how num_seeds parameter affects IK solution counts,    ║")
    print("   ║     distributions, and diversity across the joint space.          ║")
    print("   ╚═════════════════════════════════════════════════════════════════════╝")
    
    print_config_summary()
    show_usage_examples()
    
    print("\n✅ Setup complete! You can now run:")
    print("   python scripts/statistic/analyze_ik_curobo.py")
