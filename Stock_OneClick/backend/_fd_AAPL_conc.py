import importlib.util
spec=importlib.util.spec_from_file_location("p","_fd_AAPL_parse.py"); P=importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
for lbl,path in [("10K FY2025","_fd_AAPL_10K_FY2025_inst.xml"),("10Q FY26Q3","_fd_AAPL_10Q_FY26Q3_inst.xml")]:
    F=P.parse(path)
    print("="*90); print(lbl)
    for f in F:
        d=f["dims"]
        if "MajorCustomersAxis" in d or "ConcentrationRiskByBenchmarkAxis" in d:
            print(f" {f['tag']:55s} {d} {f['per']} = {f['val']}")
    print("-- geographic (StatementGeographicalAxis, long-lived / rev):")
    for f in F:
        if f["dims"].get("StatementGeographicalAxis"):
            print(f" {f['tag']:55s} {f['dims']} {f['per']} = {f['val']/1e6:,.0f}M")
