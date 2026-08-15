import json, sys

sys.stdout.reconfigure(encoding='utf-8')

with open('model-perception.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Print full source for cell 15 (CondConViT_V2) and cell 8 (CondConv2D)
for idx in [8, 15]:
    c = nb['cells'][idx]
    src = ''.join(c['source'])
    print(f"======== Cell {idx} (type={c['cell_type']}, exec={c.get('execution_count', 'N/A')}) ========")
    print(src)
    print()
