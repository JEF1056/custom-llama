#!/usr/bin/env python3
"""Print detailed benchmark results with GPU prompt speeds."""
import json

results = json.load(open('/tmp/benchmark_results_v3.json'))

print('='*90)
print('  DETAILED RESULTS: 3 Prompts x 3 Machines x 3 Context Sizes')
print('='*90)
print()
print('  Note: GPU P-tok/s = server-side prompt processing speed (excludes network)')
print('        Eff P-tok/s = end-to-end speed (includes network transfer)')
print()

for size_name in ['short', 'medium', 'long']:
    print(f'  [{size_name.upper()} CONTEXT]')
    header = '  Prompt | {:<22} {:>8} {:>12} {:>12} {:>8}'
    print(header.format('Machine', 'Time(s)', 'GPU P-tok/s', 'D-tok/s', 'Decode'))
    print('  -------|' + '_'*22 + ' ' + '_'*8 + ' ' + '_'*12 + ' ' + '_'*12 + ' ' + '_'*8)
    
    for prompt_idx in range(len(results[size_name])):
        prompt_key = str(prompt_idx)
        for machine_name in results[size_name][prompt_key].keys():
            result = results[size_name][prompt_key].get(machine_name, {})
            if result.get('success'):
                print(f'  P{prompt_idx+1}    | {machine_name:<22} {result["elapsed_s"]:>8.1f} {result["gpu_prompt_tok_per_sec"]:>12.1f} {result["decode_tok_per_sec"]:>12.1f} {result["completion_tokens"]:>8,}')
            else:
                print(f'  P{prompt_idx+1}    | {machine_name:<22} {"ERROR":>8} {"":>12} {"":>12} {"":>8}')

print()
print('='*90)
print('  COMPARISON CHART (Average Decode Tokens/sec)')
print('='*90)
print()
print(f'  {"Machine":<22} {"Short":>10} {"Medium":>10} {"Long":>10}')
print(f'  {"_"*22} {"_"*10} {"_"*10} {"_"*10}')

machine_names = list(results['short']['0'].keys())
for machine_name in machine_names:
    vals = []
    for size in ['short', 'medium', 'long']:
        avg = 0
        count = 0
        for pidx in range(len(results[size])):
            r = results[size][str(pidx)].get(machine_name, {})
            if r.get('success'):
                avg += r['decode_tok_per_sec']
                count += 1
        vals.append(avg / count if count > 0 else 0)
    print(f'  {machine_name:<22} {vals[0]:>10.1f} {vals[1]:>10.1f} {vals[2]:>10.1f}')

print()
print('='*90)
print('  COMPARISON CHART (Average GPU Prompt Processing Tokens/sec)')
print('='*90)
print()
print(f'  {"Machine":<22} {"Short":>10} {"Medium":>10} {"Long":>10}')
print(f'  {"_"*22} {"_"*10} {"_"*10} {"_"*10}')

for machine_name in machine_names:
    vals = []
    for size in ['short', 'medium', 'long']:
        avg = 0
        count = 0
        for pidx in range(len(results[size])):
            r = results[size][str(pidx)].get(machine_name, {})
            if r.get('success'):
                avg += r['gpu_prompt_tok_per_sec']
                count += 1
        vals.append(avg / count if count > 0 else 0)
    print(f'  {machine_name:<22} {vals[0]:>10.1f} {vals[1]:>10.1f} {vals[2]:>10.1f}')

print()
print('='*90)
print('  COMPARISON CHART (Average Total Time in Seconds)')
print('='*90)
print()
print(f'  {"Machine":<22} {"Short":>10} {"Medium":>10} {"Long":>10}')
print(f'  {"_"*22} {"_"*10} {"_"*10} {"_"*10}')

for machine_name in machine_names:
    vals = []
    for size in ['short', 'medium', 'long']:
        avg = 0
        count = 0
        for pidx in range(len(results[size])):
            r = results[size][str(pidx)].get(machine_name, {})
            if r.get('success'):
                avg += r['elapsed_s']
                count += 1
        vals.append(avg / count if count > 0 else 0)
    print(f'  {machine_name:<22} {vals[0]:>10.1f} {vals[1]:>10.1f} {vals[2]:>10.1f}')

print()
print('='*90)
print('  KEY FINDINGS')
print('='*90)
print()
print('  1. GPU Prompt Processing (Prefill):')
print('     - localhost (RTX 3090): ~2700-3300 tok/s')
print('     - ml-2 (Apple Silicon): ~1100-1900 tok/s')
print('     - ml-3 (Apple Silicon): ~530-820 tok/s')
print()
print('  2. Decode Speeds are consistent across all machines:')
print('     - All machines: 65-95 tok/s')
print('     - No significant degradation with larger contexts')
print()
print('  3. Total Time (including network):')
print('     - localhost: 26-52s (fast prompt, but large payload transfer)')
print('     - ml-2: 11-88s (balanced)')
print('     - ml-3: 10-176s (fast for short, slow for long)')
print()
print('  4. The original benchmark was flawed:')
print('     - Context was generated but NOT included in the prompt')
print('     - GPU prompt speed appeared as ~200 tok/s instead of ~3000 tok/s')
print('     - KV cache was matching similar contexts, skewing results')
