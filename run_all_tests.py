#!/usr/bin/env python3
"""
Run all controller tests sequentially and save results
"""
import subprocess
import time
import re
from pathlib import Path

controllers = [
    ('neural', 'Neural Network'),
    ('tdof', '2DOF'),
    ('adaptive', 'Adaptive'),
    ('datadriven', 'Data-Driven'),
    ('optimizer', 'Optimization-Based'),
]

results = {}

print("="*70)
print("COMPREHENSIVE CONTROLLER TESTING - 2000 SEGMENTS EACH")
print("="*70)
print()

for ctrl_name, ctrl_display in controllers:
    print(f"Testing {ctrl_display} Controller ({ctrl_name})...")
    print("-"*70)

    # Run evaluation
    cmd = [
        'python', 'eval.py',
        '--model_path', './models/tinyphysics.onnx',
        '--data_path', './data',
        '--num_segs', '2000',
        '--test_controller', ctrl_name,
        '--baseline_controller', 'pid'
    ]

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start_time

    if result.returncode != 0:
        print(f"ERROR: {ctrl_name} test failed!")
        print(result.stderr)
        results[ctrl_name] = {
            'display_name': ctrl_display,
            'status': 'FAILED',
            'error': result.stderr[:200]
        }
        continue

    # Extract scores from report.html
    try:
        with open('report.html', 'r', encoding='utf-8') as f:
            html = f.read()

        pattern = r'<td>(\w+)</td>\s*<td>([\d.]+)</td>\s*<td>([\d.]+)</td>\s*<td>([\d.]+)</td>'
        matches = re.findall(pattern, html)

        if matches:
            for match in matches:
                controller, lataccel, jerk, total = match
                if controller == 'baseline':
                    baseline_score = float(total)
                elif controller == 'test':
                    test_score = float(total)

            results[ctrl_name] = {
                'display_name': ctrl_display,
                'status': 'SUCCESS',
                'baseline_score': baseline_score,
                'test_score': test_score,
                'ratio': test_score / baseline_score if baseline_score > 0 else float('inf'),
                'elapsed_time': elapsed
            }

            print(f"✓ {ctrl_display}: {test_score:.2f} (baseline: {baseline_score:.2f}, ratio: {test_score/baseline_score:.2f}x)")
        else:
            results[ctrl_name] = {
                'display_name': ctrl_display,
                'status': 'FAILED',
                'error': 'Could not parse results'
            }
            print(f"✗ {ctrl_display}: Could not parse results")

    except Exception as e:
        results[ctrl_name] = {
            'display_name': ctrl_display,
            'status': 'FAILED',
            'error': str(e)
        }
        print(f"✗ {ctrl_display}: {str(e)}")

    print(f"Time: {elapsed:.1f}s")
    print()

# Print final summary
print("="*70)
print("FINAL RESULTS SUMMARY - 2000 SEGMENTS")
print("="*70)
print(f"{'Controller':<20} {'Score':<15} {'vs Baseline':<15} {'Status':<10}")
print("-"*70)

for ctrl_name, data in results.items():
    if data['status'] == 'SUCCESS':
        print(f"{data['display_name']:<20} {data['test_score']:<15.2f} {data['ratio']:<15.2f}x {data['status']:<10}")
    else:
        print(f"{data['display_name']:<20} {'N/A':<15} {'N/A':<15} {data['status']:<10}")

print("="*70)

# Find best controller
successful = [(name, data) for name, data in results.items() if data['status'] == 'SUCCESS']
if successful:
    best = min(successful, key=lambda x: x[1]['test_score'])
    print(f"\nBest Controller: {best[1]['display_name']} with score {best[1]['test_score']:.2f}")

    # Check if any beat baseline
    beat_baseline = [(name, data) for name, data in successful if data['test_score'] < data['baseline_score']]
    if beat_baseline:
        print(f"\nControllers that BEAT baseline:")
        for name, data in beat_baseline:
            improvement = (1 - data['ratio']) * 100
            print(f"  - {data['display_name']}: {data['test_score']:.2f} ({improvement:.1f}% better)")
    else:
        print(f"\nNO controllers beat baseline PID (score ~{successful[0][1]['baseline_score']:.2f})")
else:
    print("\nAll tests failed!")
