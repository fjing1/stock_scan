import sys, re
pat=sys.argv[1]; files=sys.argv[2:]
pre=int(__import__('os').environ.get('PRE','200')); post=int(__import__('os').environ.get('POST','400'))
n=int(__import__('os').environ.get('N','8'))
for f in files:
    t=open(f, errors='replace').read()
    ms=list(re.finditer(pat,t,re.I))
    print(f'##### {f}: {len(ms)} hits')
    for m in ms[:n]:
        print('   ...'+re.sub(r'\s+',' ',t[max(0,m.start()-pre):m.end()+post])+'...')
        print()
