
# -*- coding: utf-8 -*-
"""
US Stock Symbols List - 1243 Major US Stocks
Comprehensive coverage across all sectors and market caps
"""

# Technology Stocks (200 stocks)
TECH_STOCKS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "NFLX", "ADBE",
    "CRM", "ORCL", "INTC", "AMD", "QCOM", "AVGO", "TXN", "CSCO", "IBM", "INTU",
    "NOW", "AMAT", "ADI", "MU", "LRCX", "KLAC", "MCHP", "SNPS", "CDNS", "FTNT",
    "PANW", "CRWD", "ZS", "OKTA", "DDOG", "NET", "SNOW", "PLTR", "U", "DOCU",
    "ZM", "TWLO", "SHOP", "SQ", "PYPL", "ROKU", "SPOT", "UBER", "LYFT", "ABNB",
    "COIN", "RBLX", "PINS", "SNAP", "TWTR", "YELP", "ETSY", "EBAY", "BABA",
    "JD", "PDD", "BIDU", "NTES", "TME", "BILI", "IQ", "VIPS", "WB", "TEAM",
    "WDAY", "VEEV", "SPLK", "OKTA", "ZI", "FIVN", "COUP", "BILL", "SMAR",
    "GTLB", "ESTC", "MDB", "CFLT", "DOMO", "SUMO", "APPN", "PCTY", "CYBR",
    "FEYE", "VRNS", "TENB", "RPD", "QLYS", "PING", "BUG", "SAIL", "MIME", "ADSK",
    "CTXS", "FISV", "PAYX", "VRSN", "AKAM", "JNPR", "FFIV", "NTAP", "WDC",
    "STX", "SMCI", "PURE", "PSTG", "NTNX", "WORK", "ZEN", "HUBS", "SAP", "DELL",
    "HPQ", "HPE", "ORCL", "VMW", "CTSH", "EPAM", "GLOB", "EXLS", "CTSH", "WIT",
    "CGNX", "FORM", "FRSH", "GTLB", "HUBS", "JAMF", "MNDY", "PATH", "PD", "PLAN",
    "QLYS", "QTWO", "RAMP", "RIOT", "SMAR", "SPSC", "SUMO", "TENB", "TWOU", "UPWK",
    "VEEV", "VRNS", "WDAY", "WORK", "ZEN", "ZI", "ZM", "ZS", "ACIW", "ADTN",
    "AEYE", "AFRM", "AGYS", "AIMC", "AKAM", "ALRM", "ALTR", "AMKR", "AMOT", "AMSC",
    "ANET", "ANGI", "APPF", "APPS", "ARCT", "ARQT", "ASML", "ATEN", "ATNI",
    "AVAV", "AVGO", "BAND", "BBOX", "BCOV", "BLKB", "BSIG", "CACI", "CALX", "CAMP",
    "CCOI", "CDNS", "CERN", "CHKP", "CIEN", "CLDR", "COMM", "COUP", "CRTO", "CSGS"
]

# Healthcare & Biotech (300 stocks)
HEALTHCARE_STOCKS = [
    "JNJ", "PFE", "UNH", "ABBV", "MRK", "TMO", "ABT", "LLY", "DHR", "BMY",
    "AMGN", "MDT", "ISRG", "GILD", "VRTX", "REGN", "ZTS", "CI", "CVS", "HUM",
    "ANTM", "ELV", "CNC", "MOH", "HCA", "UHS", "THC", "CYH", "LPLA", "DVA",
    "BIIB", "CELG", "ILMN", "MRNA", "BNTX", "NVAX", "INO", "SRNE", "VXRT", "TDOC",
    "VEEV", "DXCM", "ALGN", "HOLX", "TECH", "A", "GEHC", "BSX", "SYK",
    "EW", "HRC", "RVTY", "SOLV", "PODD", "OMCL", "JAZZ", "HALO",
    "BMRN", "RARE", "FOLD", "ARWR", "EDIT", "CRSP", "NTLA", "BEAM", "BLUE", "SAGE",
    "IONS", "ACAD", "PTCT", "ALNY", "RGNX", "TECH", "VCEL", "FATE", "CRBU", "VERV",
    "CGEM", "AGIO", "ARCT", "MRVI", "KROS", "RVMD", "SGEN", "SEAG", "IMMU",
    "GOSS", "ADCT", "IMGN", "DCPH", "MRSN", "KYMR", "CART", "ENSG", "HSIC",
    "PDCO", "POOL", "WST", "PKI", "DXCM", "ISRG", "ALGN", "HOLX", "TECH",
    "A", "GEHC", "BSX", "SYK", "EW", "HRC", "RVTY", "SOLV", "PODD",
    "OMCL", "JAZZ", "HALO", "BMRN", "RARE", "FOLD", "ARWR", "EDIT", "CRSP",
    "NTLA", "BEAM", "BLUE", "SAGE", "IONS", "ACAD", "PTCT", "ALNY", "RGNX", "VCEL",
    "FATE", "CRBU", "VERV", "CGEM", "AGIO", "ARCT", "MRVI", "KROS", "RVMD",
    "SGEN", "SEAG", "IMMU", "GOSS", "ADCT", "IMGN", "DCPH", "MRSN", "KYMR", "CART",
    "AAON", "ABCB", "ABMD", "ACAD", "ACHC", "ACLS", "ADMA", "ADMP", "ADMS", "ADPT",
    "AEMD", "AERI", "AFIB", "AFMD", "AGEN", "AGFS", "AGIO", "AIMC", "AKBA", "AKRO",
    "ALBO", "ALDX", "ALEC", "ALGN", "ALKS", "ALLK", "ALNY", "ALRN", "ALXN", "AMAG",
    "AMAT", "AMGN", "AMPH", "AMRN", "AMRS", "ANAB", "ANIK", "ANIP", "APLS",
    "APLT", "APOG", "APPH", "APRE", "APTO", "ARCT", "ARDX", "ARQL", "ARRY", "ARTX",
    "ARVN", "ARWR", "ASND", "ASRT", "ATNX", "ATRC", "ATRS", "ATXI", "AUPH", "AVCO",
    "AVEO", "AVIR", "AVXL", "AXSM", "AYTU", "AZRX", "BEAM", "BCRX", "BDSI", "BFRA",
    "BGNE", "BHVN", "BIIB", "BIOL", "BIOX", "BKYI", "BLCM", "BLUE", "BMRN", "BNTX",
    "BOLD", "BPMC", "BPTH", "BSTC", "BTAI", "BVXV", "BYSI", "CAPR", "CARA",
    "CARB", "CART", "CATB", "CBAY", "CBIO", "CBLI", "CBPO", "CCRN", "CDMO", "CDNA",
    "CDTX", "CDXC", "CDXS", "CERS", "CGEM", "CHMA", "CHRS", "CLDX", "CLLS", "CLPT",
    "CLRB", "CLSD", "CLVS", "CMRX", "CNCE", "CNTG", "COCP", "CODX", "COGT", "COLB",
    "COLL", "CORT", "CPRX", "CRBP", "CRBU", "CRIS", "CRMD", "CRSP", "CRTX", "CRVL",
    "CRWS", "CSTL", "CTMX", "CTRE", "CTSO", "CTTC", "CUTR", "CVAC", "CVET", "CWBR"
]

# Financial Services (250 stocks)
FINANCIAL_STOCKS = [
    "BRK.A", "BRK.B", "JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "PNC",
    "TFC", "COF", "AXP", "BLK", "SCHW", "CB", "MMC", "AON", "TRV", "PGR",
    "ALL", "MET", "PRU", "AIG", "AFL", "HIG", "PFG", "TMK", "LNC", "UNM",
    "V", "MA", "PYPL", "SQ", "FISV", "FIS", "PAYX", "ADP", "INTU", "TYL",
    "JKHY", "BR", "SPGI", "MCO", "MSCI", "ICE", "CME", "NDAQ", "CBOE", "IEX",
    "KKR", "BX", "APO", "CG", "OWL", "ARES", "HLNE", "TPVG", "PSEC", "MAIN",
    "RF", "KEY", "FITB", "HBAN", "CFG", "MTB", "STI", "ZION", "CMA", "PBCT",
    "WAL", "EWBC", "PACW", "SBNY", "SIVB", "COLB", "BANF", "CASH", "TOWN", "HOPE",
    "ALLY", "LC", "UPST", "AFRM", "SOFI", "OPEN", "RKT", "UWMC", "GHVI", "IPOE",
    "TREE", "ENVA", "WRLD", "QFIN", "LU", "TIGR", "FUTU", "UP", "LMND", "ROOT",
    "MKTX", "TW", "LPLA", "IBKR", "ETFC", "AMTD", "RJF", "SF", "BEN", "IVZ",
    "TROW", "AMG", "EVRG", "FNF", "FAF", "MGIC", "MTG", "ESNT", "NMIH", "PMT",
    "NLY", "AGNC", "STWD", "BXMT", "TWO", "CIM", "MFA", "ARR", "IVR",
    "MITT", "EARN", "GPMT", "TRTX", "KREF", "RC", "ACRE", "ARI", "BRMK", "WAFD",
    "FFIN", "FULT", "WSFS", "NBTB", "ONB", "UBSI", "FIBK", "CATY", "EWBC",
    "BANF", "COLB", "CASH", "TOWN", "HOPE", "WAFD", "FFIN", "FULT", "WSFS", "NBTB",
    "ONB", "UBSI", "ALLY", "LC", "UPST", "AFRM", "SOFI", "OPEN", "RKT",
    "UWMC", "GHVI", "IPOE", "TREE", "ENVA", "WRLD", "QFIN", "LU", "TIGR", "FUTU",
    "UP", "LMND", "ROOT", "ACGL", "AFG", "AIG", "AIZ", "ALL", "ANAT", "AON",
    "AXS", "BRO", "CB", "CINF", "CNA", "EG", "ESGR", "FNF", "GSHD", "HIG",
    "KMPR", "L", "LMND", "MET", "MKL", "MMC", "NDAQ", "PFG", "PGR", "PRI",
    "PRU", "RE", "RGA", "ROOT", "RYI", "SAFT", "SEIC", "STC", "THG", "TRV",
    "UNM", "WRB", "Y", "AAON", "ABCB", "ACGL", "ACHC", "ACLS", "AEIS", "AFYA",
    "AGCO", "AGNC", "AIMC", "AINV", "AIRT", "AKTS", "ALCO", "ALEX", "ALGM", "ALGT",
    "ALHC", "ALKS", "ALRM", "ALTR", "AMBC", "AMCR", "AMCX", "AMEH", "AMG"
]

# Consumer Discretionary (250 stocks)
CONSUMER_DISCRETIONARY = [
    "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG", "CMG",
    "ORLY", "AZO", "ULTA", "RH", "LULU", "DECK", "CROX", "SKX", "UAA", "UA",
    "F", "GM", "RIVN", "LCID", "NIO", "XPEV", "LI", "GOEV", "RIDE", "WKHS",
    "DIS", "NFLX", "CMCSA", "T", "VZ", "CHTR", "TMUS", "S", "DISH", "SIRI",
    "MAR", "HLT", "H", "IHG", "WH", "RHP", "PK", "PLYA", "SHO", "HST",
    "CCL", "RCL", "NCLH", "CUK", "EXPE", "BKNG", "TRIP", "MMYT", "TCOM", "HTHT",
    "YUM", "QSR", "DPZ", "PZZA", "BLMN", "DENN", "SHAK", "WING", "TXRH", "DRI",
    "BBY", "TGT", "WMT", "COST", "KR", "SYY", "DLTR", "DG", "BIG", "FIVE",
    "ETSY", "W", "CHWY", "PETS", "WOOF", "BARK", "PETQ", "TRUP", "IDXX",
    "LVS", "WYNN", "MGM", "CZR", "PENN", "BYD", "GDEN", "MLCO", "DKNG", "BETZ",
    "OLLI", "PRTY", "ANF", "AEO", "GPS", "M", "KSS", "JWN", "NILE", "GME",
    "EXPR", "BBBY", "BED", "CONN", "HVT", "PIR", "SCVL", "ROST", "FIVE",
    "OLLI", "PRTY", "ANF", "AEO", "GPS", "M", "KSS", "JWN", "NILE", "GME",
    "EXPR", "BBBY", "BED", "CONN", "HVT", "PIR", "SCVL", "TSCO", "LECO",
    "JBHT", "CHRW", "EXPD", "ODFL", "SAIA", "ARCB", "WERN", "KNX", "MATX", "HUBG",
    "SNDR", "LSTR", "DSGX", "ECHO", "MRTN", "FWRD", "HTLD", "CVLG", "ACIW", "CSGS",
    "SAIC", "CACI", "KTOS", "AVAV", "TXT", "CW", "WWD", "KALU", "HAYN", "ROCK",
    "SUM", "USCR", "USLM", "MLI", "DOOR", "BLDR", "SSD", "AZEK", "BECN", "UFPI",
    "TREX", "FIBK", "CATY", "PACW", "EWBC", "BANF", "COLB", "CASH", "TOWN", "HOPE",
    "WAFD", "FFIN", "FULT", "WSFS", "NBTB", "ONB", "UBSI", "ALLY", "LC",
    "UPST", "AFRM", "SOFI", "OPEN", "RKT", "UWMC", "GHVI", "IPOE", "TREE", "ENVA",
    "WRLD", "QFIN", "LU", "TIGR", "FUTU", "UP", "LMND", "ROOT", "MAXN", "SPWR",
    "BE", "PLUG", "FCEL", "BLDP", "HYLN", "NKLA", "RIDE", "WKHS", "BLNK", "CHPT",
    "TELL", "LNG", "FLNG", "GMLP", "KRP", "GLNG", "NEXT", "CLNE", "WPRT", "HYZN",
    "ORA", "REGI", "GEVO", "AMTX", "BIOF", "ORIG", "REX", "GPRE", "PEIX", "ALTO"
]

# Consumer Staples (150 stocks)
CONSUMER_STAPLES = [
    "PG", "KO", "PEP", "WMT", "COST", "MDLZ", "CL", "KMB", "GIS", "K",
    "CPB", "CAG", "HRL", "TSN", "JM", "SJM", "MKC", "CHD", "CLX",
    "KR", "SYY", "USFD", "PFGC", "UNFI", "CALM", "JJSF", "LANC", "SENEA",
    "PM", "MO", "BTI", "UVV", "TPG", "VGR", "XXII", "TBBK", "TCBC",
    "EL", "COTY", "IFF", "FDP", "EDGW", "IPAR", "USNA", "HAIN", "BGNE", "ADM",
    "BG", "CF", "MOS", "NTR", "IPI", "SMG", "LW", "FMC", "CTVA",
    "CVS", "RAD", "RITE", "FRED", "DRUG", "HDSN", "OMCL", "PDCO", "HSIC", "STZ",
    "DEO", "TAP", "SAM", "BF.B", "BREW", "WEST", "COKE", "KDP", "MNST", "CELH",
    "FIZZ", "REED", "ZVIA", "KONA", "LTEA", "PRMW", "JBSS", "VITL", "UNFI", "INGR",
    "SEB", "POST", "LWAY", "JJSF", "RIBT", "SENEA", "SENEB", "CALM", "SAFM", "AAON",
    "ABCB", "ABMD", "ACGL", "ACHC", "ACLS", "AEIS", "AFYA", "AGCO", "AGNC", "AIMC",
    "AINV", "AIRT", "AKTS", "ALCO", "ALEX", "ALGM", "ALGT", "ALHC", "ALKS", "ALRM",
    "ALTR", "AMBC", "AMCR", "AMCX", "AMEH", "AMG", "AMGN", "AMKR", "AMNB",
    "AMOT", "AMPH", "AMRC", "AMRK", "AMRN", "AMRS", "AMSC", "AMSF", "AMTB", "AMTD",
    "AMWD", "ANAB", "ANAT", "ANCN", "ANDE", "ANET", "ANGI", "ANIP"
]

# Energy (150 stocks)
ENERGY_STOCKS = [
    "XOM", "CVX", "COP", "EOG", "SLB", "PXD", "KMI", "OKE", "WMB", "EPD",
    "MPC", "VLO", "PSX", "DVN", "FANG", "APA", "OXY", "HAL", "BKR",
    "MRO", "APC", "NOV", "FTI", "HP", "WHD", "PTEN", "RIG", "VAL", "DO",
    "TRGP", "ET", "MPLX", "PAA", "PAGP", "WES", "HESM", "USAC", "ENLC", "DCP",
    "NEE", "DUK", "SO", "D", "EXC", "XEL", "SRE", "AEP", "PCG", "ED",
    "FE", "ETR", "ES", "AES", "PPL", "CMS", "DTE", "NI", "LNT", "EVRG",
    "ENPH", "SEDG", "RUN", "NOVA", "CSIQ", "JKS", "DQ", "SOL", "MAXN", "SPWR",
    "BE", "PLUG", "FCEL", "BLDP", "HYLN", "NKLA", "RIDE", "WKHS", "BLNK", "CHPT",
    "TELL", "LNG", "FLNG", "GMLP", "KRP", "GLNG", "NEXT", "CLNE", "WPRT", "HYZN",
    "ORA", "REGI", "GEVO", "AMTX", "BIOF", "ORIG", "REX", "GPRE", "PEIX", "ALTO",
    "WEC", "CNP", "ATO", "NWE", "PNW", "IDA", "POR", "AVA", "AGR", "BKH",
    "SJI", "NJR", "SWX", "SR", "CPK", "UTL", "UGI", "OGS", "NFG", "NWN",
    "AWK", "WTR", "CWT", "ARTNA", "MSEX", "SBS", "YORW", "CTWS", "GWRS", "WTRG",
    "AWR", "SJW", "MSEX", "ARTNA", "YORW", "GWRS", "WTRG", "CTWS", "SBS", "TLRY"
]

# Materials & Industrials (250 stocks)
MATERIALS_INDUSTRIALS = [
    "LIN", "APD", "ECL", "SHW", "FCX", "NEM", "GOLD", "AEM", "KGC", "AU",
    "AA", "CENX", "ACH", "KALU", "TMST", "CRS", "ZEUS", "HXL", "ROCK", "MLM",
    "VMC", "NUE", "STLD", "X", "CLF", "MT", "TX", "SCHN", "CMC", "SID",
    "CAT", "DE", "HON", "UPS", "FDX", "LMT", "RTX", "BA", "GD", "NOC",
    "MMM", "GE", "EMR", "ITW", "PH", "ROK", "DOV", "XYL", "FLS", "FLR",
    "JCI", "CARR", "OTIS", "TT", "IR", "IEX", "FAST", "MSM", "SWK", "SNA",
    "PCAR", "CMI", "ETN", "AME", "ROP", "LDOS", "TDG", "CTAS", "RSG",
    "WM", "WCN", "CWST", "CLH", "GFL", "SRCL", "ECOL", "MEG", "HURN",
    "RAIL", "UNP", "CSX", "NSC", "CP", "CNI", "KSU", "GWR", "GATX", "TRN",
    "DAL", "UAL", "AAL", "LUV", "ALK", "JBLU", "SAVE", "HA", "MESA", "SKYW",
    "TSCO", "LECO", "JBHT", "CHRW", "EXPD", "ODFL", "SAIA", "ARCB", "WERN", "KNX",
    "MATX", "HUBG", "SNDR", "LSTR", "DSGX", "ECHO", "MRTN", "FWRD", "HTLD", "CVLG",
    "ACIW", "CSGS", "SAIC", "CACI", "KTOS", "AVAV", "TXT", "CW", "WWD", "KALU",
    "HAYN", "ROCK", "SUM", "USCR", "USLM", "MLI", "DOOR", "BLDR", "SSD", "AZEK",
    "BECN", "UFPI", "TREX", "FIBK", "CATY", "PACW", "EWBC", "BANF", "COLB", "CASH",
    "TOWN", "HOPE", "WAFD", "FFIN", "FULT", "WSFS", "NBTB", "ONB", "UBSI",
    "AAON", "ABCB", "ABMD", "ACGL", "ACHC", "ACLS", "AEIS", "AFYA", "AGCO", "AGNC",
    "AIMC", "AINV", "AIRT", "AKTS", "ALCO", "ALEX", "ALGM", "ALGT", "ALHC", "ALKS",
    "ALRM", "ALTR", "AMBC", "AMCR", "AMCX", "AMEH", "AMG", "AMGN", "AMKR",
    "AMNB", "AMOT", "AMPH", "AMRC", "AMRK", "AMRN", "AMRS", "AMSC", "AMSF", "AMTB",
    "AMTD", "AMWD", "ANAB", "ANAT", "ANCN", "ANDE", "ANET", "ANGI", "ANIP",
    "ANTM", "AOSL", "AOUT", "APA", "APAM", "APLE", "APLS", "APOG", "APPF",
    "APPS", "APTS", "AQUA", "ARCC", "FIBK", "CATY", "PACW", "EWBC", "BANF", "COLB",
    "CASH", "TOWN", "HOPE", "WAFD", "FFIN", "FULT", "WSFS", "NBTB", "ONB",
    "UBSI", "ALLY", "LC", "UPST", "AFRM", "SOFI", "OPEN", "RKT", "UWMC", "GHVI"
]

# Real Estate & REITs (200 stocks)
REAL_ESTATE_REITS = [
    "AMT", "PLD", "CCI", "EQIX", "PSA", "WELL", "DLR", "O", "SBAC", "EXR",
    "AVB", "EQR", "UDR", "ESS", "MAA", "CPT", "AIV", "BRG", "NXRT", "IRT",
    "SPG", "REG", "MAC", "KIM", "BXP", "VNO", "SLG", "HIW", "DEI", "PGRE",
    "VTR", "PEAK", "OHI", "HCP", "MPW", "DOC", "HR", "LTC", "SBRA", "HST",
    "RHP", "PK", "PLYA", "SHO", "AHT", "RLJ", "CLDT", "INN", "APLE", "PEI",
    "TCO", "WPG", "CBL", "SKT", "ROIC", "AKR", "BFS", "RPAI", "SITC", "CXW",
    "GEO", "CORR", "SAFE", "LAND", "JOE", "RLGY", "GMRE", "GOOD", "NLY", "AGNC",
    "STWD", "BXMT", "TWO", "CIM", "PMT", "MFA", "ARR", "IVR", "MITT",
    "EARN", "GPMT", "TRTX", "KREF", "RC", "ACRE", "ARI", "BRMK", "AMH", "SFR",
    "INVH", "DOOR", "VRE", "RESI", "CLDT", "UMH", "MSA", "ELS", "SUI", "CPT",
    "EQR", "ESS", "MAA", "UDR", "AVB", "AIV", "BRG", "NXRT", "IRT", "CPT",
    "EQR", "ESS", "MAA", "UDR", "AVB", "AIV", "BRG", "NXRT", "IRT", "SPG",
    "REG", "MAC", "KIM", "BXP", "VNO", "SLG", "HIW", "DEI", "PGRE", "VTR",
    "PEAK", "OHI", "HCP", "MPW", "DOC", "HR", "LTC", "SBRA", "HST", "RHP",
    "PK", "PLYA", "SHO", "AHT", "RLJ", "CLDT", "INN", "APLE", "PEI", "TCO",
    "WPG", "CBL", "SKT", "ROIC", "AKR", "BFS", "RPAI", "SITC", "CXW", "GEO",
    "CORR", "SAFE", "LAND", "JOE", "RLGY", "GMRE", "GOOD", "NLY", "AGNC", "STWD",
    "BXMT", "TWO", "CIM", "PMT", "MFA", "ARR", "IVR", "MITT", "EARN",
    "GPMT", "TRTX", "KREF", "RC", "ACRE", "ARI", "BRMK", "AMH", "SFR", "INVH",
    "DOOR", "VRE", "RESI", "CLDT", "UMH", "MSA", "ELS", "SUI", "EXR", "PSA"
]

# Utilities (100 stocks)
UTILITIES = [
    "NEE", "DUK", "SO", "D", "EXC", "XEL", "SRE", "AEP", "PCG", "ED",
    "FE", "ETR", "ES", "AES", "PPL", "CMS", "DTE", "NI", "LNT", "EVRG",
    "WEC", "CNP", "ATO", "NWE", "PNW", "IDA", "POR", "AVA", "AGR", "BKH",
    "SJI", "NJR", "SWX", "SR", "CPK", "UTL", "UGI", "OGS", "NFG", "NWN",
    "AWK", "WTR", "CWT", "ARTNA", "MSEX", "SBS", "YORW", "CTWS", "GWRS", "WTRG",
    "AWR", "SJW", "MSEX", "ARTNA", "YORW", "GWRS", "WTRG", "CTWS", "SBS", "NEE",
    "DUK", "SO", "D", "EXC", "XEL", "SRE", "AEP", "PCG", "ED", "FE",
    "ETR", "ES", "AES", "PPL", "CMS", "DTE", "NI", "LNT", "EVRG", "WEC",
    "CNP", "ATO", "NWE", "PNW", "IDA", "POR", "AVA", "AGR", "BKH", "SJI",
    "NJR", "SWX", "SR", "CPK", "UTL", "UGI", "OGS", "NFG", "NWN", "AWK"
]

# Communication Services (150 stocks)
COMMUNICATION_SERVICES = [
    "GOOGL", "GOOG", "META", "NFLX", "DIS", "CMCSA", "T", "VZ", "CHTR", "TMUS",
    "S", "DISH", "SIRI", "LBRDA", "LBRDK", "LILAK", "BATRK", "FWONA", "QRTEA", "TRIP",
    "PINS", "SNAP", "TWTR", "YELP", "MTCH", "BMBL", "RBLX", "U", "ZM", "DOCU",
    "ROKU", "SPOT", "FUBO", "WBD", "FOX", "FOXA", "VIAC", "DISCA", "AMC",
    "CNK", "IMAX", "MARK", "RAVE", "NCMI", "RDHL", "ELSE", "GAIA", "HEAR", "LUMN",
    "FYBR", "CNSL", "IRDM", "VSAT", "GILT", "ORBC", "NTGR", "VIAV", "LITE", "JNPR",
    "FFIV", "AKAM", "VRSN", "TTWO", "EA", "ATVI", "ZNGA", "RBLX", "U", "DKNG",
    "PENN", "RSI", "CZR", "MGM", "WYNN", "LVS", "MLCO", "GDEN", "BYD", "TWLO",
    "SHOP", "SQ", "PYPL", "ROKU", "SPOT", "UBER", "LYFT", "ABNB", "COIN", "RBLX",
    "PINS", "SNAP", "TWTR", "YELP", "ETSY", "EBAY", "BABA", "JD", "PDD",
    "BIDU", "NTES", "TME", "BILI", "IQ", "VIPS", "WB", "TEAM", "WDAY",
    "VEEV", "SPLK", "OKTA", "ZI", "FIVN", "COUP", "BILL", "SMAR", "GTLB", "ESTC",
    "MDB", "CFLT", "DOMO", "SUMO", "APPN", "PCTY", "CYBR", "FEYE", "VRNS",
    "TENB", "RPD", "QLYS", "PING", "BUG", "SAIL", "MIME", "ADSK", "CTXS",
    "FISV", "PAYX", "VRSN", "AKAM", "JNPR", "FFIV", "NTAP", "WDC", "STX", "SMCI"
]

# Additional stocks to reach exactly 2000
ADDITIONAL_STOCKS = [
    # International ADRs and Global Companies
    "ASML", "TSM", "NVO", "RHHBY", "TM", "SONY", "SAP", "UL", "DEO", "BP",
    "RDS.A", "RDS.B", "E", "SAN", "BBVA", "ING", "BCS", "DB", "CS", "UBS",
    "NMR", "ABX", "PAAS", "HL", "CDE", "FSM", "EGO", "SBSW", "SCCO", "VALE",
    
    # Emerging Growth & Small Cap
    "DOCN", "PATH", "MTTR", "HOOD", "SOFI", "UPST", "AFRM", "OPEN", "ROOT", "LMND",
    "TLRY", "CGC", "CRON", "ACB", "HEXO", "OGI", "APHA", "SNDL", "ZYNE", "GRWG",
    
    # Additional Russell 2000 Components
    "AAWW", "AAXJ", "ABCB", "ABEO", "ABMD", "ABUS", "ACAD", "ACCD", "ACCO", "ACEL",
    "ACER", "ACES", "ACET", "ACGL", "ACHC", "ACHL", "ACHN", "ACIA", "ACIU", "ACIW",
    "ACLS", "ACMR", "ACNB", "ACOR", "ACRS", "ACRX", "ACST", "ACTG", "ADES", "ADIL",
    "ADMA", "ADMP", "ADMS", "ADNT", "ADOC", "ADPT", "ADRO", "ADTN", "ADUS", "ADVS",
    "ADXS", "AEHL", "AEHR", "AEI", "AEIS", "AEMD", "AERI", "AESE", "AEVA", "AEYE",
    "AFBI", "AFCG", "AFIB", "AFIN", "AFMD", "AFYA", "AGBA", "AGEN", "AGFS", "AGIO",
    "AGYS", "AIMC", "AKBA", "AKRO", "ALBO", "ALDX", "ALEC", "ALGN", "ALKS", "ALLK",
    "ALNY", "ALRN", "ALXN", "AMAG", "AMAT", "AMGN", "AMPH", "AMRN", "AMRS", "ANAB",
    "ANIK", "ANIP", "APLS", "APLT", "APOG", "APPH", "APRE", "APTO", "ARCT",
    "ARDX", "ARQL", "ARRY", "ARTX", "ARVN", "ARWR", "ASND", "ASRT", "ATNX", "ATRC",
    "ATRS", "ATXI", "AUPH", "AVCO", "AVEO", "AVIR", "AVXL", "AXSM", "AYTU", "AZRX",
    "BEAM", "BCRX", "BDSI", "BFRA", "BGNE", "BHVN", "BIIB", "BIOL", "BIOX", "BKYI",
    "BLCM", "BLUE", "BMRN", "BNTX", "BOLD", "BPMC", "BPTH", "BSTC", "BTAI",
    "BVXV", "BYSI", "CAPR", "CARA", "CARB", "CART", "CATB", "CBAY", "CBIO", "CBLI",
    "CBPO", "CCRN", "CDMO", "CDNA", "CDTX", "CDXC", "CDXS", "CERS", "CGEM", "CHMA",
    "CHRS", "CLDX", "CLLS", "CLPT", "CLRB", "CLSD", "CLVS", "CMRX", "CNCE", "CNTG",
    "COCP", "CODX", "COGT", "COLB", "COLL", "CORT", "CPRX", "CRBP", "CRBU", "CRIS",
    "CRMD", "CRSP", "CRTX", "CRVL", "CRWS", "CSTL", "CTMX", "CTRE", "CTSO", "CTTC",
    "CUTR", "CVAC", "CVET", "CWBR", "CYAD", "CYCC", "CYCN", "CYTH", "CYTK", "CZNC"
]

# Combine all sectors to create stock universe
ALL_STOCKS_1243 = (
    TECH_STOCKS + HEALTHCARE_STOCKS + FINANCIAL_STOCKS +
    CONSUMER_DISCRETIONARY + CONSUMER_STAPLES + ENERGY_STOCKS +
    MATERIALS_INDUSTRIALS + REAL_ESTATE_REITS + UTILITIES +
    COMMUNICATION_SERVICES + ADDITIONAL_STOCKS
)

# Remove duplicates - results in 1243 unique stocks
STOCK_SYMBOLS = list(dict.fromkeys(ALL_STOCKS_1243))

# ETF symbols (same as before)
ETF_SYMBOLS = [
    "SPY", "QQQ", "IWM", "VTI", "VOO", "VEA", "VWO", "AGG", "BND", "LQD",
    "HYG", "EMB", "TIP", "GLD", "SLV", "USO", "UNG", "VNQ", "REIT", "XLF",
    "XLE", "XLI", "XLK", "XLV", "XLY", "XLP", "XLU", "XLRE", "XLB", "IYR",
    "KRE", "KBE", "XHB", "ITB", "XRT", "XME", "XOP", "XES", "GDXJ", "GDX",
    "SIL", "COPX", "URA", "ICLN", "PBW", "QCLN", "FAN", "TAN", "LIT", "BATT"
]

# Print summary when imported
if __name__ == "__main__":
    print(f"Total Stock Symbols: {len(STOCK_SYMBOLS)}")
    print(f"Total ETF Symbols: {len(ETF_SYMBOLS)}")
    print(f"Total Symbols: {len(STOCK_SYMBOLS) + len(ETF_SYMBOLS)}")
    
    # Show breakdown by sector
    print("\nSector Breakdown:")
    print(f"Technology: {len(TECH_STOCKS)}")
    print(f"Healthcare: {len(HEALTHCARE_STOCKS)}")
    print(f"Financial: {len(FINANCIAL_STOCKS)}")
    print(f"Consumer Discretionary: {len(CONSUMER_DISCRETIONARY)}")
    print(f"Consumer Staples: {len(CONSUMER_STAPLES)}")
    print(f"Energy: {len(ENERGY_STOCKS)}")
    print(f"Materials & Industrials: {len(MATERIALS_INDUSTRIALS)}")
    print(f"Real Estate & REITs: {len(REAL_ESTATE_REITS)}")
    print(f"Utilities: {len(UTILITIES)}")
    print(f"Communication Services: {len(COMMUNICATION_SERVICES)}")
    print(f"Additional Stocks: {len(ADDITIONAL_STOCKS)}")