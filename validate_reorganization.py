#!/usr/bin/env python3
"""
Validation script for the reorganized repository structure.
Tests that imports and basic functionality work correctly.
"""

import sys
import os
from pathlib import Path

def test_production_imports():
    """Test that production package imports work correctly."""
    print("🧪 Testing production package imports...")
    
    try:
        # Add src to path
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        
        # Test main package import
        import agent_factory
        print("  ✅ agent_factory package imported successfully")
        
        # Test pipelines import (most important)
        from agent_factory.pipelines import IngPipeline
        print("  ✅ IngPipeline imported from production package")
        
        # Test individual agent imports
        try:
            from agent_factory.agents import agent_0
            print("  ✅ Agent modules imported successfully")
        except ImportError as e:
            print(f"  ⚠️  Agent import warning: {e}")
        
        # Test core utilities (may have optional dependencies)
        try:
            from agent_factory.core import llmp_utils
            print("  ✅ Core utilities imported successfully")
        except ImportError as e:
            print(f"  ⚠️  Core utilities warning: {e}")
        
        print("  ✅ Essential imports working - reorganization successful!")
        return True
        
    except ImportError as e:
        print(f"  ❌ Critical import error: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")
        return False

def test_directory_structure():
    """Test that the new directory structure exists."""
    print("\n📁 Testing directory structure...")
    
    required_dirs = [
        "src/agent_factory",
        "src/agent_factory/agents", 
        "src/agent_factory/pipelines",
        "src/agent_factory/core",
        "research/experiments",
        "research/notebooks",
        "research/research_configs",
        "data/knowledge_bases",
        "data/test_data",
        "outputs",
        "tests"
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"  ✅ {dir_path}")
        else:
            print(f"  ❌ {dir_path} - Missing!")
            all_exist = False
    
    return all_exist

def test_research_structure():
    """Test research area organization."""
    print("\n🧪 Testing research structure...")
    
    research_files = [
        "research/README.md",
        "research/experiments/ingestion/README.md", 
        "research/experiments/ingestion/ing_pip_test.ipynb",
        "research/research_configs/chunk_size_optimization.json"
    ]
    
    all_exist = True
    for file_path in research_files:
        if os.path.exists(file_path):
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} - Missing!")
            all_exist = False
    
    return all_exist

def main():
    """Run all validation tests."""
    print("🔍 Validating Repository Reorganization")
    print("=" * 50)
    
    # Run tests
    imports_ok = test_production_imports()
    structure_ok = test_directory_structure() 
    research_ok = test_research_structure()
    
    # Summary
    print("\n📊 Validation Summary:")
    print("=" * 30)
    
    if imports_ok and structure_ok and research_ok:
        print("🎉 All tests passed! Repository reorganization successful.")
        print("\n✅ Next steps:")
        print("  1. Update notebook imports to use new structure")
        print("  2. Run research experiments to validate functionality") 
        print("  3. Add production tests in tests/ directory")
        return 0
    else:
        print("❌ Some tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    exit(main())
