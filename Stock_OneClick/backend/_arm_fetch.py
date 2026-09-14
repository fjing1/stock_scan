import sys, os, time, requests
UA = {"User-Agent": "independent research fj@example.com"}
def get(url, out):
    if os.path.exists(out) and os.path.getsize(out) > 1000:
        print("cached", out, os.path.getsize(out)); return
    r = requests.get(url, headers=UA, timeout=60)
    print(r.status_code, len(r.content), url, "->", out)
    if r.status_code == 200:
        open(out, "wb").write(r.content)
    time.sleep(0.4)

CIK = "1973239"
def doc(acc, fn, out):
    a = acc.replace("-", "")
    get(f"https://www.sec.gov/Archives/edgar/data/{CIK}/{a}/{fn}", out)

if __name__ == "__main__":
    jobs = [
      ("0001973239-26-000097","arm-20260331.htm","_arm_20f_fy2026.htm"),
      ("0001973239-25-000016","arm-20250331.htm","_arm_20f_fy2025.htm"),
      ("0001973239-24-000012","arm-20240331.htm","_arm_20f_fy2024.htm"),
    ]
    for a,f,o in jobs: doc(a,f,o)
    get("https://www.sec.gov/Archives/edgar/data/1973239/000119312523235320/d550931d424b4.htm","_arm_ipo_prospectus_424b4.htm")
