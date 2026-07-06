#!/usr/bin/env python3
"""Smoke test for smf-bench: load model manifest + test suites, verify capability gating."""
import sys
sys.path.insert(0, '.')

from smf_bench import ModelRegistry, TestRegistry

# 1. Load model manifests
reg = ModelRegistry()
n_models = reg.load_dir('models/')
print(f'Models loaded: {n_models}')
for mid in reg.list_models():
    m = reg.get(mid)
    print(f'  {mid}: caps={sorted(c.value for c in m.capabilities)}, '
          f'in_mods={sorted(m.value for m in m.input_modalities)}')

# Focus on Qwen3.6-35B
manifest = reg.get('qwen3.6-35b-a3b-nvfp4')
print(f'\nFocus model: {manifest.model_id}')
print(f'  Capabilities: {sorted(c.value for c in manifest.capabilities)}')
print(f'  Input modalities: {sorted(m.value for m in manifest.input_modalities)}')
print()

# 2. Load all test suites
treg = TestRegistry()
n_tests = treg.load_dir('suites/')
print(f'Total tests loaded: {n_tests}')
print(f'Categories: {treg.categories()}')
print()

# 3. Partition into applicable / NA
applicable, na = reg.applicable_tests(treg, 'qwen3.6-35b-a3b-nvfp4')
print(f'Applicable tests: {len(applicable)}')
print(f'NA tests: {len(na)}')
print()

# 4. Show NA categories
na_cats = set(t.category for t in na)
print('NA categories (model lacks required capability/modality):')
for c in sorted(na_cats):
    count = sum(1 for t in na if t.category == c)
    print(f'  {c}: {count} tests NA')
print()

# 5. Show applicable categories
app_cats = set(t.category for t in applicable)
print('Applicable categories:')
for c in sorted(app_cats):
    count = sum(1 for t in applicable if t.category == c)
    print(f'  {c}: {count} tests')
print()

# 6. Show sample test IDs
print('Sample applicable test IDs (first 10):')
for t in applicable[:10]:
    print(f'  {t.test_id} [{t.category}] evaluator={t.evaluator}')
print()
print('Sample NA test IDs (first 5):')
for t in na[:5]:
    req_caps = sorted(c.value for c in t.required_capabilities)
    req_mods = sorted(m.value for m in t.required_modalities)
    print(f'  {t.test_id} [{t.category}] requires caps={req_caps} mods={req_mods}')