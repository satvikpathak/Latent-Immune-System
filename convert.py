import json
import re
import sys

def py_to_ipynb(py_path, ipynb_path):
    with open(py_path, 'r', encoding='utf-8') as f:
        content = f.read()

    cells = []
    # Split content by cell markers
    # A cell starts with "# %%" or "# %% [markdown]"
    parts = re.split(r'^#\s*%%\s*', content, flags=re.MULTILINE)
    
    for part in parts:
        if not part.strip():
            continue
        
        # Check if it is a markdown cell
        if part.startswith('[markdown]'):
            lines = part.split('\n')[1:]
            # Clean up the lines by stripping leading '# ' if present
            md_lines = []
            for line in lines:
                if line.startswith('# '):
                    md_lines.append(line[2:] + '\n')
                elif line.startswith('#'):
                    md_lines.append(line[1:] + '\n')
                else:
                    md_lines.append(line + '\n')
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": md_lines
            })
        else:
            # Code cell
            lines = [line + '\n' for line in part.split('\n')]
            # Clean up trailing newlines
            if lines and lines[-1] == '\n':
                lines = lines[:-1]
            if lines:
                lines[-1] = lines[-1].rstrip('\n')
            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": lines
            })

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    with open(ipynb_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    print(f"Successfully converted {py_path} to {ipynb_path}")

if __name__ == "__main__":
    py_to_ipynb("steer_alignment.py", "steer_alignment.ipynb")
