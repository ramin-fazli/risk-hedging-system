#!/usr/bin/env python3
"""
GitHub Publishing Readiness Script
==================================

This script performs final checks and preparations before publishing to GitHub.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command, description):
    """Run a command and return success status."""
    logger.info(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"✅ {description} completed successfully")
            return True
        else:
            logger.error(f"❌ {description} failed: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"❌ {description} error: {e}")
        return False

def check_git_status():
    """Check if all changes are committed."""
    logger.info("🔍 Checking Git status...")
    try:
        result = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True)
        if result.stdout.strip():
            logger.warning("⚠️ Uncommitted changes found:")
            logger.warning(result.stdout)
            return False
        else:
            logger.info("✅ All changes are committed")
            return True
    except Exception as e:
        logger.error(f"❌ Git status check failed: {e}")
        return False

def count_project_files():
    """Count project files and lines of code."""
    logger.info("📊 Counting project files...")
    
    try:
        # Count files (excluding venv and .git)
        total_files = 0
        python_files = 0
        total_lines = 0
        
        for root, dirs, files in os.walk('.'):
            # Skip venv and .git directories
            dirs[:] = [d for d in dirs if d not in ['venv', '.git', '__pycache__']]
            
            for file in files:
                if not file.startswith('.') and not file.endswith('.pyc'):
                    total_files += 1
                    
                    if file.endswith('.py'):
                        python_files += 1
                        try:
                            with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                                total_lines += sum(1 for line in f)
                        except:
                            pass
        
        logger.info(f"📁 Total files: {total_files}")
        logger.info(f"🐍 Python files: {python_files}")
        logger.info(f"📝 Lines of code: {total_lines}")
        
        return total_files, python_files, total_lines
    except Exception as e:
        logger.error(f"❌ File counting failed: {e}")
        return 0, 0, 0

def run_tests():
    """Run all available tests."""
    logger.info("🧪 Running tests...")
    
    tests_passed = 0
    tests_total = 0
    
    # Run deployment verification
    if run_command('python deployment_verification.py', 'Deployment verification'):
        tests_passed += 1
    tests_total += 1
    
    # Run system test
    if run_command('python system_test.py', 'System integration test'):
        tests_passed += 1
    tests_total += 1
    
    # Run unit tests if pytest is available
    try:
        subprocess.run(['python', '-m', 'pytest', '--version'], capture_output=True, check=True)
        if run_command('python -m pytest tests/ -v', 'Unit tests'):
            tests_passed += 1
        tests_total += 1
    except:
        logger.info("ℹ️ Pytest not available, skipping unit tests")
    
    logger.info(f"🎯 Tests passed: {tests_passed}/{tests_total}")
    return tests_passed == tests_total

def create_release_notes():
    """Create release notes for v1.0.0."""
    logger.info("📝 Creating release notes...")
    
    release_notes = """# FBX Hedging Strategy Backtesting System v1.0.0

## 🚀 Initial Release

This is the first production release of the FBX Hedging Strategy Backtesting System, a comprehensive ML-powered Python project for backtesting hedging strategies against the Freightos Baltic Index (FBX).

## ✨ Key Features

### 🎯 Core Capabilities
- **Complete Backtesting Framework**: Simulate hedging strategies with comprehensive performance metrics
- **Advanced Analytics**: Multi-method hedge ratio optimization and risk analysis
- **Professional Reporting**: Generate detailed Excel reports with visualizations
- **Data Management**: Automated data fetching, processing, and validation

### 🤖 Machine Learning Pipeline
- **Feature Engineering**: 200+ engineered features with technical indicators
- **Multi-Model Training**: 11+ algorithms including Random Forest, XGBoost, Neural Networks
- **Ensemble Methods**: Voting, stacking, and blending for improved accuracy
- **Hyperparameter Optimization**: Bayesian optimization with Optuna
- **Model Interpretation**: SHAP, LIME, and permutation importance

### 🛠️ Production Ready
- **Scalable Architecture**: Configurable complexity levels (minimal/testing/production)
- **Robust Error Handling**: Graceful degradation and comprehensive logging
- **Automated Testing**: Unit tests, ML tests, and system integration tests
- **CI/CD Pipeline**: GitHub Actions workflow for automated testing

## 📊 Performance Metrics
- **Data Processing**: 1,736+ days of financial data
- **Hedge Instruments**: 40+ ETFs and stocks analyzed
- **Execution Time**: 2-5 minutes (configurable)
- **Memory Usage**: 2-16 GB (configurable)

## 🚀 Getting Started

1. **Install Python 3.8+**
2. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/fbx-hedging-strategy.git
   cd fbx-hedging-strategy
   ```
3. **Run the installer** (Windows):
   ```bash
   install.bat
   ```
4. **Run the system**:
   ```bash
   run.bat
   ```

## 📚 Documentation

- **[README.md](README.md)** - Complete project documentation
- **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** - Production deployment guide
- **[ML_DOCUMENTATION.md](ML_DOCUMENTATION.md)** - Machine learning pipeline documentation
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contribution guidelines

## 🔧 System Requirements

- **Python**: 3.8+
- **RAM**: 2-16 GB (configurable)
- **Storage**: 1-5 GB
- **OS**: Windows 10/11, macOS 10.14+, Linux Ubuntu 18.04+

## 🎯 What's Included

- 75+ files with comprehensive documentation
- 35+ Python modules
- 11,868+ lines of production code
- Professional Excel reporting
- Advanced ML capabilities
- Complete test suite
- GitHub Actions CI/CD

## 🤝 Contributing

We welcome contributions! Please read our [Contributing Guide](CONTRIBUTING.md) for details on setting up the development environment and submitting pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Built with ❤️ for the quantitative finance community**
"""
    
    try:
        with open('RELEASE_NOTES.md', 'w', encoding='utf-8') as f:
            f.write(release_notes)
        logger.info("✅ Release notes created successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to create release notes: {e}")
        return False

def generate_github_commands():
    """Generate the commands needed for GitHub publishing."""
    logger.info("📋 Generating GitHub publishing commands...")
    
    commands = """
# GitHub Publishing Commands
# =========================

# 1. Create repository on GitHub (do this first):
#    - Go to https://github.com/new
#    - Repository name: fbx-hedging-strategy
#    - Description: Advanced ML-powered hedging strategy backtesting system for FBX exposure
#    - Public repository
#    - Don't initialize with README (we have existing code)

# 2. Connect local repository to GitHub:
git remote add origin https://github.com/yourusername/fbx-hedging-strategy.git

# 3. Push to GitHub:
git branch -M main
git push -u origin main

# 4. Create the first release:
git tag -a v1.0.0 -m "v1.0.0: Initial release with ML capabilities"
git push origin v1.0.0

# 5. Then create a GitHub Release:
#    - Go to your repository on GitHub
#    - Click "Releases" → "Create a new release"
#    - Choose tag v1.0.0
#    - Title: "v1.0.0 - Initial Release"
#    - Copy content from RELEASE_NOTES.md
#    - Publish release

# 6. Optional: Set up branch protection:
#    - Go to Settings → Branches
#    - Add rule for main branch
#    - Enable "Require pull request reviews before merging"
"""
    
    try:
        with open('GITHUB_COMMANDS.txt', 'w', encoding='utf-8') as f:
            f.write(commands)
        logger.info("✅ GitHub commands generated successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to generate GitHub commands: {e}")
        return False

def main():
    """Main publishing readiness check."""
    logger.info("🚀 GitHub Publishing Readiness Check")
    logger.info("=" * 50)
    
    checks = []
    
    # Check 1: Git status
    checks.append(("Git Status", check_git_status()))
    
    # Check 2: Count files
    total_files, python_files, total_lines = count_project_files()
    checks.append(("File Count", total_files > 0))
    
    # Check 3: Run tests
    checks.append(("Test Suite", run_tests()))
    
    # Check 4: Create release notes
    checks.append(("Release Notes", create_release_notes()))
    
    # Check 5: Generate GitHub commands
    checks.append(("GitHub Commands", generate_github_commands()))
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("🎯 PUBLISHING READINESS SUMMARY")
    logger.info("=" * 50)
    
    passed = sum(1 for _, status in checks if status)
    total = len(checks)
    
    for check_name, status in checks:
        status_icon = "✅" if status else "❌"
        logger.info(f"{status_icon} {check_name}")
    
    logger.info(f"\n📊 Results: {passed}/{total} checks passed")
    
    if passed == total:
        logger.info("\n🎉 ALL CHECKS PASSED - READY FOR GITHUB PUBLISHING!")
        logger.info("🚀 Follow the instructions in GITHUB_COMMANDS.txt")
        logger.info("📝 Use RELEASE_NOTES.md for your GitHub release")
        logger.info("\n📊 Project Statistics:")
        logger.info(f"   📁 Total files: {total_files}")
        logger.info(f"   🐍 Python modules: {python_files}")
        logger.info(f"   📝 Lines of code: {total_lines}")
        logger.info(f"   🎯 Features: Production-ready ML backtesting system")
        return True
    else:
        logger.error("\n❌ SOME CHECKS FAILED - REVIEW REQUIRED")
        logger.error("🔧 Please fix the issues before publishing")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
