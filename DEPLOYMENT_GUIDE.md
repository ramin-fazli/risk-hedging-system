"""
FBX Hedging Strategy Backtesting System - Deployment Guide
==========================================================

🚀 QUICK START GUIDE

1. SYSTEM REQUIREMENTS
   - Python 3.8+ (recommended: 3.12+)
   - Windows 10/11 (tested) or macOS/Linux
   - 4-16 GB RAM (depends on ML configuration)
   - 1-2 GB disk space

2. INSTALLATION
   ```bash
   # Clone or extract project
   cd shipping_project
   
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # macOS/Linux
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. CONFIGURATION
   Edit config/settings.py:
   ```python
   # Basic settings
   START_DATE = datetime(2020, 1, 1)
   END_DATE = datetime(2024, 12, 31)
   
   # ML settings
   ML_ENABLED = True
   ML_MODE = "minimal"  # Options: minimal, testing, production
   ```

4. EXECUTION
   ```bash
   # Quick run
   python main.py
   
   # Or use batch script (Windows)
   run.bat
   ```

📊 CONFIGURATION MODES

MINIMAL MODE (Recommended for first run)
- Memory: 2-4 GB
- Time: 5-15 minutes
- Models: Random Forest, Ridge Regression
- Features: Basic + Technical Indicators

TESTING MODE (Development)
- Memory: 4-8 GB
- Time: 20-60 minutes
- Models: 6+ algorithms
- Features: Full feature engineering

PRODUCTION MODE (Full capabilities)
- Memory: 8-16 GB
- Time: 1-4 hours
- Models: All 11 algorithms + ensembles
- Features: Complete ML pipeline

🔧 TROUBLESHOOTING

Common Issues and Solutions:

1. "No module named 'talib'" (Optional dependency)
   - Solution: System works without TA-Lib
   - Alternative: pip install TA-Lib (requires additional setup)

2. "ImportError: cannot import name '_lazywhere'"
   - Solution: Already fixed in requirements.txt
   - Alternative: pip install scipy==1.10.0

3. Memory issues with ML
   - Solution: Set ML_MODE = "minimal"
   - Alternative: Increase virtual memory

4. Slow execution
   - Solution: Reduce date range or disable ML
   - Alternative: Use more powerful hardware

🎯 OUTPUT FILES

Generated in reports/ directory:
- FBX_Hedging_Report_YYYYMMDD_HHMMSS.xlsx
  * Executive Summary
  * Data Analysis
  * Exposure Analysis
  * Hedge Ratios
  * Backtesting Results
  * Risk Analysis
  * ML Analysis (if enabled)
  * Charts and Visualizations

📈 PERFORMANCE EXPECTATIONS

Typical Results:
- Data Processing: 1,500-2,000 days
- Hedge Instruments: 30-50 instruments
- Execution Time: 2-5 minutes (minimal mode)
- Report Size: 2-10 MB Excel file
- Memory Usage: 2-4 GB (minimal mode)

🔍 VALIDATION

System Health Check:
```bash
python system_test.py
```

Expected Output:
✅ 6/7 components pass (ML optional)
✅ Data generation works
✅ System ready for production

🚀 DEPLOYMENT CHECKLIST

□ Python 3.8+ installed
□ Virtual environment created
□ Dependencies installed
□ Configuration customized
□ Test run successful
□ Output directory accessible
□ System test passed

🎖️ PRODUCTION RECOMMENDATIONS

1. Resource Allocation:
   - CPU: 4+ cores recommended
   - RAM: 8+ GB for production mode
   - Storage: SSD for better performance

2. Configuration:
   - Start with minimal mode
   - Gradually increase complexity
   - Monitor memory usage

3. Monitoring:
   - Check logs in logs/ directory
   - Monitor execution time
   - Validate output reports

4. Maintenance:
   - Update dependencies quarterly
   - Review configuration annually
   - Archive old reports

🔒 SECURITY CONSIDERATIONS

- No sensitive data stored
- All data processing local
- No external API keys required
- Virtual environment isolation

📞 SUPPORT

For issues:
1. Check logs in logs/ directory
2. Review error messages
3. Try minimal configuration
4. Refer to documentation

🎯 SUCCESS METRICS

Your deployment is successful if:
✅ System runs without errors
✅ Excel report generated
✅ Data processing completes
✅ Charts and visualizations appear
✅ ML analysis included (if enabled)

STATUS: ✅ PRODUCTION READY
Last Updated: July 2025
Version: 1.0.0
==========================================================
