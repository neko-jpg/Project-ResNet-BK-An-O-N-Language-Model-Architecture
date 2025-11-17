"""
Script to fix Koopman notebook configuration
"""
import json
import sys

# Read notebook
with open('notebooks/step2_phase2_koopman.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find and modify the configuration cell
modified = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and cell['source']:
        source_text = ''.join(cell['source'])
        
        # Check if this is the configuration cell
        if 'KOOPMAN_WEIGHT_MAX = 0.1' in source_text:
            print("Found configuration cell, modifying...")
            
            # Modify the source
            new_source = []
            for line in cell['source']:
                if 'KOOPMAN_WEIGHT_MAX = 0.1' in line:
                    new_source.append('KOOPMAN_WEIGHT_MAX = 0.5  # Maximum Koopman loss weight (increased for stronger signal)\n')
                    print(f"  Modified: {line.strip()} -> KOOPMAN_WEIGHT_MAX = 0.5")
                elif 'NUM_EPOCHS = 5' in line:
                    new_source.append('NUM_EPOCHS = 10  # Increased from 5 for better Koopman learning\n')
                    print(f"  Modified: {line.strip()} -> NUM_EPOCHS = 10")
                else:
                    new_source.append(line)
            
            cell['source'] = new_source
            modified = True
            break

if modified:
    # Write back
    with open('notebooks/step2_phase2_koopman.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("\n✓ Notebook updated successfully!")
    print("  - KOOPMAN_WEIGHT_MAX: 0.1 → 0.5")
    print("  - NUM_EPOCHS: 5 → 10")
else:
    print("✗ Configuration cell not found")
    sys.exit(1)
