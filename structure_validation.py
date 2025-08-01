#!/usr/bin/env python3
"""
Simple validation for repository reorganization.
Tests directory structure and basic organization.
"""

import os
from pathlib import Path

def test_directory_structure():
    """Test that the new directory structure exists."""
    print("📁 Testing directory structure...")
    
    required_structure = {
        "Production Code": [
            "src/agent_factory",
            "src/agent_factory/agents", 
            "src/agent_factory/pipelines",
            "src/agent_factory/core",
            "src/agent_factory/config",
            "src/agent_factory/api"
        ],
        "Research & Experiments": [
            "research/experiments",
            "research/notebooks",
            "research/research_configs",
            "research/experiments/ingestion",
            "research/experiments/agents",
            "research/experiments/models",
            "research/experiments/evaluations"
        ],
        "Data Management": [
            "data/datasets",
            "data/knowledge_bases",
            "data/test_data"
        ],
        "Outputs & Results": [
            "outputs",
            "outputs/models",
            "outputs/reports", 
            "outputs/visualizations",
            "outputs/experiments_results"
        ],
        "Testing": [
            "tests",
            "tests/unit",
            "tests/integration",
            "tests/e2e"
        ]
    }
    
    all_good = True
    for category, dirs in required_structure.items():
        print(f"\n  {category}:")
        for dir_path in dirs:
            if os.path.exists(dir_path):
                print(f"    ✅ {dir_path}")
            else:
                print(f"    ❌ {dir_path} - Missing!")
                all_good = False
    
    return all_good

def test_key_files():
    """Test that key files are in place."""
    print("\n📄 Testing key files...")
    
    key_files = [
        ("README.md", "Updated documentation"),
        ("research/README.md", "Research guidelines"),
        ("src/agent_factory/__init__.py", "Main package init"),
        ("src/agent_factory/pipelines/__init__.py", "Pipelines init"),
        ("src/agent_factory/agents/__init__.py", "Agents init"),
        ("research/experiments/ingestion/ing_pip_test.ipynb", "Moved test notebook"),
        ("research/research_configs/chunk_size_optimization.json", "Research config example")
    ]
    
    all_exist = True
    for file_path, description in key_files:
        if os.path.exists(file_path):
            print(f"  ✅ {file_path} ({description})")
        else:
            print(f"  ❌ {file_path} - Missing!")
            all_exist = False
    
    return all_exist

def test_file_moves():
    """Test that files were properly moved from old locations."""
    print("\n📦 Testing file migrations...")
    
    moved_files = [
        ("src/pipelines/ing_pip_test.ipynb", "research/experiments/ingestion/ing_pip_test.ipynb"),
        ("src/agents/agent_0.py", "src/agent_factory/agents/agent_0.py"),
        ("src/pipelines/ingestion_pipeline_2.py", "src/agent_factory/pipelines/ingestion_pipeline_2.py"),
        ("src/utils/llmp_utils.py", "src/agent_factory/core/llmp_utils.py")
    ]
    
    migration_ok = True
    for old_path, new_path in moved_files:
        old_exists = os.path.exists(old_path)
        new_exists = os.path.exists(new_path)
        
        if not old_exists and new_exists:
            print(f"  ✅ Moved: {old_path} → {new_path}")
        elif old_exists and new_exists:
            print(f"  ⚠️  Both exist: {old_path} and {new_path}")
        elif old_exists and not new_exists:
            print(f"  ❌ Not moved: {old_path} still exists, {new_path} missing")
            migration_ok = False
        else:
            print(f"  ❓ Unknown: Neither {old_path} nor {new_path} exists")
    
    return migration_ok

def main():
    """Run all validation tests."""
    print("🔍 Repository Reorganization Validation")
    print("=" * 50)
    
    # Run tests
    structure_ok = test_directory_structure()
    files_ok = test_key_files()
    migration_ok = test_file_moves()
    
    # Summary
    print("\n📊 Validation Summary:")
    print("=" * 30)
    
    if structure_ok and files_ok and migration_ok:
        print("🎉 Repository reorganization successful!")
        print("\n✅ The structure separates:")
        print("  • Production code (src/agent_factory/)")
        print("  • Research & experiments (research/)")
        print("  • Data management (data/)")
        print("  • Results & outputs (outputs/)")
        print("  • Testing (tests/)")
        print("\n🚀 Ready for development with clear separation of concerns!")
        return 0
    else:
        print("❌ Some structural issues found. Check the details above.")
        return 1

if __name__ == "__main__":
    exit(main())
