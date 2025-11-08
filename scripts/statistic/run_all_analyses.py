#!/usr/bin/env python3
"""
Run all statistical analyses in sequence.

This script runs all analysis modules in the recommended order
and generates a comprehensive report.

Usage:
    python scripts/statistic/run_all_analyses.py [--quick]

Options:
    --quick     Use smaller data subsets for faster testing

Author: Statistical Analysis Suite for AutoMoMA
Date: 2025-10-16
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime

# Add the src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from config import create_output_dirs, OUTPUT_ROOT


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80 + "\n")


def run_all_analyses(analyses_to_run: list = None):
    """Run selected analysis scripts in sequence.
    
    Args:
        analyses_to_run: List of analyses to run. Options: 'IK', 'TRAJ', 'IK_CLUSTERING', 'IK_MMD'.
                        If None, runs all analyses.
    """
    
    if analyses_to_run is None:
        analyses_to_run = ['IK', 'TRAJ', 'IK_CLUSTERING', 'IK_MMD']
    
    print_header("AutoMoMA Statistical Analysis Suite")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\nAnalyses to run: {', '.join(analyses_to_run)}\n")
    
    # Create output directories
    print("Setting up output directories...")
    create_output_dirs()
    print("✓ Output directories created\n")
    
    # Track results
    results = {
        'start_time': datetime.now().isoformat(),
        'analyses_to_run': analyses_to_run,
        'analyses': {}
    }
    
    analysis_index = 1
    
    # Analysis 1: Precomputed IK
    if 'IK' in analyses_to_run:
        print_header(f"Analysis {analysis_index}/{len(analyses_to_run)}: Precomputed IK Data Analysis")
        print("Analyzing IK data from ik_data.pt files...")
        print("This compares precomputed IK with trajectory-derived IK.")
        
        try:
            from analyze_ik import main as analyze_ik_main
            analyze_ik_main()
            results['analyses']['ik_analysis'] = {
                'status': 'success',
                'timestamp': datetime.now().isoformat()
            }
            print("\n✓ IK analysis completed successfully")
        except Exception as e:
            print(f"\n✗ IK analysis failed: {e}")
            results['analyses']['ik_analysis'] = {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
        analysis_index += 1
    else:
        print_header("Skipping IK Analysis")
        print("⏭️  IK analysis not in the requested list")
        results['analyses']['ik_analysis'] = {
            'status': 'skipped',
            'reason': 'not_requested',
            'timestamp': datetime.now().isoformat()
        }
    
    # Analysis 2: Trajectory Data
    if 'TRAJ' in analyses_to_run:
        print_header(f"Analysis {analysis_index}/{len(analyses_to_run)}: Trajectory Data Analysis")
        print("Analyzing trajectory data (scale, diversity, quality)...")
        print("This is the main comprehensive analysis.")
        
        try:
            from analyze_traj import main as analyze_traj_main
            analyze_traj_main()
            results['analyses']['trajectory_analysis'] = {
                'status': 'success',
                'timestamp': datetime.now().isoformat()
            }
            print("\n✓ Trajectory analysis completed successfully")
        except Exception as e:
            print(f"\n✗ Trajectory analysis failed: {e}")
            results['analyses']['trajectory_analysis'] = {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
        analysis_index += 1
    else:
        print_header("Skipping Trajectory Analysis")
        print("⏭️  TRAJ analysis not in the requested list")
        results['analyses']['trajectory_analysis'] = {
            'status': 'skipped',
            'reason': 'not_requested',
            'timestamp': datetime.now().isoformat()
        }
    
    # Analysis 3: IK Clustering
    if 'IK_CLUSTERING' in analyses_to_run:
        print_header(f"Analysis {analysis_index}/{len(analyses_to_run)}: IK Clustering Analysis")
        print("⚠️  This analysis requires running the planner and takes much longer!")
        print("   It generates raw and clustered IK from scratch to show coverage.")
        
        try:
            from analyze_ik_clustering import main as analyze_clustering_main
            analyze_clustering_main()
            results['analyses']['ik_clustering_analysis'] = {
                'status': 'success',
                'timestamp': datetime.now().isoformat()
            }
            print("\n✓ IK clustering analysis completed successfully")
        except Exception as e:
            print(f"\n✗ IK clustering analysis failed: {e}")
            results['analyses']['ik_clustering_analysis'] = {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
        analysis_index += 1
    else:
        print_header("Skipping IK Clustering Analysis")
        print("⏭️  IK_CLUSTERING analysis not in the requested list")
        results['analyses']['ik_clustering_analysis'] = {
            'status': 'skipped',
            'reason': 'not_requested',
            'timestamp': datetime.now().isoformat()
        }
    
    # Analysis 4: IK MMD Analysis
    if 'IK_MMD' in analyses_to_run:
        print_header(f"Analysis {analysis_index}/{len(analyses_to_run)}: MMD-based Seed Quantity Ablation")
        print("Computing MMD scores with IMQ kernel for seed quantity ablation study...")
        print("This uses cached IK data from analyze_ik_curobo.py")
        
        try:
            from analyze_ik_mmd import main as analyze_mmd_main
            analyze_mmd_main()
            results['analyses']['ik_mmd_analysis'] = {
                'status': 'success',
                'timestamp': datetime.now().isoformat()
            }
            print("\n✓ IK MMD analysis completed successfully")
        except Exception as e:
            print(f"\n✗ IK MMD analysis failed: {e}")
            results['analyses']['ik_mmd_analysis'] = {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
        analysis_index += 1
    else:
        print_header("Skipping IK MMD Analysis")
        print("⏭️  IK_MMD analysis not in the requested list")
        results['analyses']['ik_mmd_analysis'] = {
            'status': 'skipped',
            'reason': 'not_requested',
            'timestamp': datetime.now().isoformat()
        }
    
    # Generate summary report
    print_header("Generating Summary Report")
    
    # Generate summary report
    print_header("Generating Summary Report")
    
    results['end_time'] = datetime.now().isoformat()
    
    # Count successes
    successful = sum(1 for a in results['analyses'].values() if a['status'] == 'success')
    failed = sum(1 for a in results['analyses'].values() if a['status'] == 'failed')
    skipped = sum(1 for a in results['analyses'].values() if a['status'] == 'skipped')
    
    print(f"\nAnalysis Summary:")
    print(f"  ✓ Successful: {successful}")
    print(f"  ✗ Failed: {failed}")
    print(f"  ⏭️  Skipped: {skipped}")
    
    # Save results
    import json
    results_path = OUTPUT_ROOT / "analysis_run_summary.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Run summary saved to: {results_path}")
    
    # Print output locations
    print("\nOutput Locations:")
    print(f"  Figures: {OUTPUT_ROOT / 'figures'}")
    print(f"  Summaries: {OUTPUT_ROOT}")
    
    print_header("Analysis Complete")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if failed > 0:
        print("\n⚠️  Some analyses failed. Check the error messages above.")
        print("   Run individual analyses for more details:")
        print("     python scripts/statistic/analyze_ik.py")
        print("     python scripts/statistic/analyze_traj.py")
        print("     python scripts/statistic/analyze_ik_clustering.py")
        print("     python scripts/statistic/analyze_ik_mmd.py")
    else:
        print("\n🎉 All analyses completed successfully!")
        print("\nNext steps:")
        print("  1. View figures in: output/statistics/figures/")
        print("  2. Review summaries: output/statistics/*.json")
        print("  3. Create custom analyses using the utilities")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run selected statistical analyses for AutoMoMA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all analyses
  python scripts/statistic/run_all_analyses.py IK TRAJ IK_CLUSTERING IK_MMD
  
  # Run only IK analysis
  python scripts/statistic/run_all_analyses.py IK
  
  # Run IK and trajectory analysis
  python scripts/statistic/run_all_analyses.py IK TRAJ
  
  # Run only clustering analysis
  python scripts/statistic/run_all_analyses.py IK_CLUSTERING
  
  # Run MMD analysis (seed quantity ablation)
  python scripts/statistic/run_all_analyses.py IK_MMD
        """
    )
    
    parser.add_argument(
        'analyses',
        nargs='*',
        default=['IK', 'TRAJ', 'IK_CLUSTERING', 'IK_MMD'],
        choices=['IK', 'TRAJ', 'IK_CLUSTERING', 'IK_MMD'],
        help='List of analyses to run (default: all four)'
    )
    
    args = parser.parse_args()
    
    # If no arguments provided, use all analyses
    analyses_to_run = args.analyses if args.analyses else ['IK', 'TRAJ', 'IK_CLUSTERING', 'IK_MMD']
    
    try:
        run_all_analyses(analyses_to_run=analyses_to_run)
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
