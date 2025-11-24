#!/bin/bash
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

set -e

echo "=========================================="
echo "Migrating CDM from manip-as-in-sim-suite"
echo "=========================================="
echo ""

# Configuration
OLD_REPO_PATH="${1:-../manip-as-in-sim-suite}"
CDM_PATH="$OLD_REPO_PATH/cdm"

if [ ! -d "$CDM_PATH" ]; then
    echo "Error: CDM directory not found at $CDM_PATH"
    echo "Usage: $0 [path/to/manip-as-in-sim-suite]"
    exit 1
fi

echo "Source: $CDM_PATH"
echo ""

# Copy core package
echo "1. Copying rgbddepth package..."
if [ -d "$CDM_PATH/rgbddepth" ]; then
    rsync -av --exclude="__pycache__" --exclude="*.pyc" "$CDM_PATH/rgbddepth/" ./rgbddepth/
    echo "✓ Package copied"
else
    echo "✗ rgbddepth/ not found in source"
    exit 1
fi

# Copy example data
echo ""
echo "2. Copying example data..."
if [ -d "$CDM_PATH/example_data" ]; then
    rsync -av "$CDM_PATH/example_data/" ./example_data/
    echo "✓ Example data copied"
else
    echo "⚠ No example_data/ found (optional)"
fi

# Copy documentation
echo ""
echo "3. Copying documentation..."
mkdir -p docs

for doc in OPTIMIZATIONS.md CHEATSHEET.md; do
    if [ -f "$CDM_PATH/$doc" ]; then
        cp "$CDM_PATH/$doc" ./docs/
        echo "✓ Copied $doc"
    else
        echo "⚠ $doc not found (optional)"
    fi
done

# Copy and adapt scripts
echo ""
echo "4. Copying scripts..."
mkdir -p scripts

# Copy infer.py if exists
if [ -f "$CDM_PATH/infer.py" ]; then
    cp "$CDM_PATH/infer.py" ./rgbddepth/infer.py
    echo "✓ Copied infer.py"

    # Add if __name__ == "__main__" wrapper if not present
    if ! grep -q "if __name__ == .__main__.:" ./rgbddepth/infer.py; then
        echo "" >> ./rgbddepth/infer.py
        echo "if __name__ == \"__main__\":" >> ./rgbddepth/infer.py
        echo "    run_inference()" >> ./rgbddepth/infer.py
        echo "✓ Added CLI wrapper to infer.py"
    fi
else
    echo "⚠ infer.py not found"
fi

# Copy other useful scripts
for script in example_usage.py verify_installation.py test_optimizations.py; do
    if [ -f "$CDM_PATH/$script" ]; then
        cp "$CDM_PATH/$script" ./scripts/
        echo "✓ Copied $script"
    fi
done

# Verify structure
echo ""
echo "5. Verifying structure..."
required_files=(
    "rgbddepth/__init__.py"
    "rgbddepth/dpt.py"
    "rgbddepth/attention.py"
    "rgbddepth/optimization_config.py"
    "pyproject.toml"
    "README.md"
)

all_good=true
for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file"
    else
        echo "✗ $file (missing!)"
        all_good=false
    fi
done

# Update CLI to use migrated infer.py
echo ""
echo "6. Updating CLI integration..."
if [ -f "rgbddepth/cli.py" ] && [ -f "rgbddepth/infer.py" ]; then
    # Update the import in cli.py to point to the migrated infer
    sed -i.bak 's/from rgbddepth.infer import run_inference/from rgbddepth.infer import main as run_inference/' rgbddepth/cli.py 2>/dev/null || true
    echo "✓ CLI updated"
fi

echo ""
if [ "$all_good" = true ]; then
    echo "=========================================="
    echo "✓ Migration complete!"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "  1. Install package: pip install -e .[dev]"
    echo "  2. Run tests: pytest tests/ -v"
    echo "  3. Test CLI: cdm-infer --help"
    echo "  4. Initialize git: git init && git add . && git commit -m 'Initial commit'"
    echo ""
else
    echo "=========================================="
    echo "⚠ Migration completed with warnings"
    echo "=========================================="
    echo "Some files are missing. Check the output above."
fi
