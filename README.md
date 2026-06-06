# US Stock Scanner & VCP Pattern Detector

A comprehensive Python-based stock scanning system featuring:
1. **Traditional Stock Scanner**: Analyzes 1000+ US stocks and 50 ETFs using technical indicators
2. **VCP Pattern Detector**: Advanced Volatility Contraction Pattern detection based on Mark Minervini's methodology

## Features

### Traditional Stock Scanner
- **1000 US Stocks**: Comprehensive coverage across all major sectors
- **50 ETFs**: Major market ETFs including SPY, QQQ, IWM, and sector-specific ETFs
- **Technical Analysis**: RSI, MACD, Moving Averages (20-day, 50-day), Volume analysis
- **Scoring System**: 100-point algorithm across 4 categories (Trend, Capital Flow, Valuation, Volatility)
- **Excel Output**: Multi-sheet reports with detailed analysis and ETF overview

### VCP Pattern Detector
- **1,243 US Stocks**: Enhanced stock universe with comprehensive sector coverage
- **Mark Minervini's 10-Point Trend Template**: Complete implementation of Stage 2 identification
- **Advanced VCP Detection**: 30-point scoring system with breakout readiness analysis
- **Stage 2 Identification**: Automatic detection of stocks in Stage 2 uptrend
- **Professional Reports**: Automated markdown report generation with detailed analysis
- **Multi-timeframe Analysis**: Daily and weekly data integration for comprehensive pattern recognition

## Installation & Setup

### 1. Clone Repository
```bash
git clone <repository-url>
cd stock_scan
```

### 2. VCP Pattern Detector Setup (Recommended)

#### Create VCP Virtual Environment
```bash
# Create virtual environment for VCP
python3 -m venv vcp_env

# Activate virtual environment
source vcp_env/bin/activate  # On Windows: vcp_env\Scripts\activate

# Install VCP dependencies
pip install -r requirements.txt
```

#### Verify VCP Installation
```bash
# Test VCP dependencies
python -c "import yfinance, pandas, numpy, ta; print('VCP dependencies installed successfully!')"

# Test VCP imports
python -c "from stock_symbols_1243 import STOCK_SYMBOLS; import vcp_enhanced_criteria; print(f'VCP ready! Stock symbols: {len(STOCK_SYMBOLS)}')"
```

### 3. Traditional Scanner Setup (Alternative)

#### Create Standard Virtual Environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Verify Standard Installation
```bash
python3 -c "import yfinance, pandas, numpy, ta; print('All dependencies installed successfully!')"
```

## Usage

### VCP Pattern Detector (Primary Tool)

#### Activate VCP Environment
```bash
# Always activate VCP environment first
source vcp_env/bin/activate
```

#### Run VCP Scanner
```bash
# Interactive mode with options
python vcp_enhanced_criteria.py

# Available options:
# 1. Test mode (50 stocks) - Quick testing
# 2. Full scan (1,243 stocks) - Complete analysis
# 3. Custom stock list - Specific symbols
```

#### VCP Scanner Features
- **Mark Minervini's 10-Point Trend Template**: Complete Stage 2 identification
- **Enhanced VCP Scoring**: 30-point system (Trend 10pts + Breakout 6pts + Higher Lows 3pts + Volume 6pts + Stage 2 bonus)
- **Selective Detail Display**: Full breakdown only for top candidates (21+ points)
- **Professional Reports**: Auto-generated markdown reports in `results/YYYYMMDD/` folder
- **Stage 2 Detection**: Automatic identification of stocks meeting all 6 Stage 2 criteria

#### VCP Expected Runtime
- **Test Mode (50 stocks)**: 2-3 minutes
- **Full Scan (1,243 stocks)**: 45-60 minutes
- **Processing Speed**: ~23 stocks/minute
- **Real-time Progress**: Updates every 25 stocks with ETA

### Traditional Stock Scanner (Legacy)

#### Activate Standard Environment
```bash
# Activate standard environment
source venv/bin/activate
```

#### Run Traditional Scanner
```bash
python3 us_stock_etf_scan.py
```

#### Traditional Scanner Features
1. **Downloads 3 months of daily data** for 1050 symbols (1000 stocks + 50 ETFs)
2. **Calculates technical indicators** for each symbol
3. **Applies scoring algorithm** to identify strong opportunities
4. **Generates Excel report** with multiple sheets:
   - Stock ⭐Strong Buy (Score ≥ 85)
   - Stock ✅Buy (Score 70-84)
   - ETF ⭐Strong Buy (Score ≥ 85)
   - ETF ✅Buy (Score 70-84)
   - Industry Summary
   - ETF Overview (all ETFs regardless of score)

#### Traditional Expected Runtime
- **Estimated time**: 15-20 minutes for full scan
- **Progress updates**: Every 50 symbols processed
- **Real-time notifications**: High-scoring stocks displayed immediately

## File Structure

```
stock_scan/
├── vcp_enhanced_criteria.py     # VCP Pattern Detector (Primary)
├── stock_symbols_1243.py        # 1,243 stock symbols for VCP analysis
├── us_stock_etf_scan.py         # Traditional stock scanner (Legacy)
├── stock_symbols.py             # 1,000 stock symbols for traditional scanner
├── requirements.txt             # Python dependencies
├── README.md                    # This documentation
├── vcp_env/                     # VCP virtual environment
├── venv/                        # Traditional scanner virtual environment
└── results/                     # VCP analysis reports
    └── YYYYMMDD/
        └── YYYYMMDD-vcp.md      # Daily VCP analysis report
```

## Scoring Algorithms

### VCP Enhanced Scoring (30-Point System)

#### Mark Minervini's 10-Point Trend Template
1. **Price above 50-day MA** (1 pt)
2. **Price above 150-day MA** (1 pt)
3. **Price above 200-day MA** (1 pt)
4. **50-day > 150-day > 200-day MA** (2 pts)
5. **200-day MA rising** (1 pt)
6. **Within 25% of 52-week high** (1 pt)
7. **Above 30% of 52-week low** (1 pt)
8. **Positive relative strength** (1 pt)

#### Breakout Readiness Analysis (6 points)
- Near 100-day high within 10 candles (2 pts)
- Within 7% of daily 100-day high (2 pts)
- Within 20% of weekly 100-day high (1 pt)
- Below resistance (not broken out yet) (1 pt)

#### Higher Lows Pattern (3 points)
- 10-day higher lows (1 pt)
- 20-day higher lows (1 pt)
- 30-day higher lows (1 pt)

#### Volume Contraction (6 points)
- Volume contracting across 6 timeframes (5/10/15/20/25/30 days)
- Points awarded based on number of contracting signals

#### Stage 2 Identification
- **Stage 2 Bonus**: Additional categorization for stocks meeting all 6 core Stage 2 criteria
- **Perfect Candidates**: 24+ points with Stage 2 status

### Traditional Scoring (100-Point System)

#### 1. Trend Momentum (40 points)
- Price above 20-day MA (10 pts)
- Price above 50-day MA (10 pts)
- RSI ≥ 55 (10 pts)
- MACD > Signal line (10 pts)

#### 2. Capital Flow (20 points)
- Volume > 1.2x average (10 pts)
- Price + MACD momentum alignment (10 pts)

#### 3. Valuation & Quality (20 points)
- RSI between 5-75 (not overbought/oversold) (10 pts)
- Price within 20% of 50-day MA (10 pts)

#### 4. Volatility/Sentiment (20 points)
- Price within 10% of 20-day MA (10 pts)
- Volume ratio < 3x (not excessive) (10 pts)

## Stock Universe

### VCP Pattern Detector (1,243 Stocks)
Comprehensive coverage across all major sectors:
- **Technology**: 200 stocks (AAPL, MSFT, GOOGL, NVDA, etc.)
- **Healthcare & Biotech**: 300 stocks (JNJ, PFE, UNH, ABBV, etc.)
- **Financial Services**: 250 stocks (JPM, BAC, BRK.B, GS, etc.)
- **Consumer Discretionary**: 250 stocks (AMZN, TSLA, HD, MCD, etc.)
- **Consumer Staples**: 150 stocks (PG, KO, WMT, COST, etc.)
- **Energy**: 150 stocks (XOM, CVX, COP, EOG, etc.)
- **Materials & Industrials**: 250 stocks (CAT, DE, HON, LIN, etc.)
- **Real Estate & REITs**: 200 stocks (AMT, PLD, O, EQIX, etc.)
- **Utilities**: 100 stocks (NEE, DUK, SO, D, etc.)
- **Communication Services**: 150 stocks (META, DIS, NFLX, GOOGL, etc.)
- **Additional Stocks**: International ADRs, Growth stocks, Russell 2000 components

### Traditional Scanner (1,000 Stocks + 50 ETFs)
- **1000 stocks** across major sectors (100 per sector)
- **50 ETFs** covering:
  - Broad market (SPY, QQQ, IWM)
  - Sectors (XLF, XLE, XLK, etc.)
  - International (VEA, VWO)
  - Bonds (AGG, BND, LQD)
  - Commodities (GLD, SLV, USO)

## Output Formats

### VCP Pattern Detector Output
**Markdown Reports** (Auto-generated in `results/YYYYMMDD/`):
- **Professional Analysis**: Top 20 VCP candidates with detailed breakdowns
- **Stage 2 Identification**: Stocks meeting all 6 Stage 2 criteria marked
- **Comprehensive Methodology**: Complete explanation of Mark Minervini's approach
- **Detailed Statistics**: Pass rates for each criterion across the universe
- **Perfect Candidates**: 24+ point stocks with Stage 2 status highlighted

**Console Output**:
- **Real-time Progress**: Updates every 25 stocks with ETA
- **Selective Details**: Full breakdown only for top candidates (21+ points)
- **Stage 2 Indicators**: Immediate identification of Stage 2 stocks
- **Performance Stats**: Processing speed and discovery rates

### Traditional Scanner Output
**Excel Reports** (`US_StrongBuy_Scan_YYYYMMDD.xlsx`):
- **Timestamp**: File named with current date
- **Multiple sheets**: Organized by category and rating
- **Key metrics**: Price, change %, RSI, volume ratio, score
- **ETF Overview**: Technical snapshot of all ETFs

## Quick Start Guide

### For VCP Pattern Detection (Recommended)
```bash
# 1. Activate VCP environment
source vcp_env/bin/activate

# 2. Run VCP scanner
python vcp_enhanced_criteria.py

# 3. Select option 2 for full scan
# 4. Check results in results/YYYYMMDD/ folder
```

### For Traditional Scanning
```bash
# 1. Activate standard environment
source venv/bin/activate

# 2. Run traditional scanner
python3 us_stock_etf_scan.py
```

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**:
   - Ensure virtual environment is activated
   - Install dependencies: `pip install -r requirements.txt`

2. **Python path errors**:
   - Use direct path: `vcp_env/bin/python vcp_enhanced_criteria.py`
   - Check Python version: `python --version`

3. **Slow performance**:
   - VCP: Normal for 1,243 symbols, expect 45-60 minutes
   - Traditional: Normal for 1000+ symbols, expect 15-20 minutes

4. **Data errors**: Some symbols may have insufficient data (automatically skipped)

5. **Memory usage**: Large dataset may require 2-4GB RAM

### Performance Tips

- **Run during market hours** for most current data
- **Stable internet connection** recommended
- **Close other applications** to free up memory
- **Check progress messages** for real-time status
- **Use test mode first** to verify setup before full scans

## Customization

### VCP Pattern Detector
**Modify Stock Universe**:
- Edit `stock_symbols_1243.py` to add/remove stocks from sector lists
- Adjust `ENHANCED_VCP_CONFIG` in `vcp_enhanced_criteria.py` for criteria thresholds

**Adjust VCP Scoring**:
- Modify `check_trend_template()` for Mark Minervini criteria
- Adjust `check_uptrend_nearing_breakout()` for breakout thresholds
- Customize `check_volume_contracting()` for volume analysis periods

**Output Customization**:
- Edit `save_vcp_results_to_markdown()` for report format
- Modify minimum score thresholds for detailed output
- Adjust Stage 2 identification criteria

### Traditional Scanner
**Modify Stock List**:
- Edit `stock_symbols.py` to add/remove stocks from sector lists
- Adjust ETF selection and total number of symbols

**Adjust Scoring**:
- Edit `score_stock()` function in `us_stock_etf_scan.py`
- Modify point allocations and add new technical indicators
- Change scoring thresholds

**Output Format**:
- Modify Excel output sections for new sheets and column layouts
- Include additional metrics and analysis

## License

This project is for educational and research purposes. Please ensure compliance with data provider terms of service.
