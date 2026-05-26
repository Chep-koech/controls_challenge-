#!/usr/bin/env python3
import re

with open('report.html', 'r', encoding='utf-8') as f:
    html = f.read()

# Find the table data
# Look for pattern like: <td>baseline</td><td>1.234</td><td>5.678</td><td>123.45</td>
pattern = r'<td>(\w+)</td>\s*<td>([\d.]+)</td>\s*<td>([\d.]+)</td>\s*<td>([\d.]+)</td>'
matches = re.findall(pattern, html)

if matches:
    print("\n" + "="*60)
    print("RESULTS FROM 100 SEGMENTS:")
    print("="*60)
    print(f"{'Controller':<15} {'lataccel_cost':<15} {'jerk_cost':<15} {'total_cost':<15}")
    print("-"*60)
    for match in matches:
        controller, lataccel, jerk, total = match
        print(f"{controller:<15} {lataccel:<15} {jerk:<15} {total:<15}")
    print("="*60)
else:
    print("Could not find scores in report.html")
