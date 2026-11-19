import csv,sys,re,difflib,json
from collections import Counter
rows=list(csv.DictReader(open(sys.argv[1],encoding="utf-8"),delimiter="\t"))
O_ROW="オコゴソゾトドノホボポモヨロョ"; E_ROW="エケゲセゼテデネヘベペメレ"
PAIRS=[("ヴァ","バ"),("ヴィ","ビ"),("ヴェ","ベ"),("ヴォ","ボ"),("ヴ","ブ"),("ヂ","ジ"),("ヅ","ズ"),("ヲ","オ"),("イェ","イエ"),("トユウ","トイウ")]
def norm(s):
    for a,b in PAIRS: s=s.replace(a,b)
    s=re.sub(f"([{O_ROW}])ウ",r"\1オ",s); s=re.sub(f"([{E_ROW}])イ",r"\1エ",s)
    return s
RARE=re.compile(r"[ァィゥェォヴヂヅヲ]|[テデ][ャュョ]")
DIGLAT={"ROHAN4600_0198","ROHAN4600_1079","ROHAN4600_4537","ROHAN4600_1203","ROHAN4600_4223"}
cat=Counter(); ucat={}; rest=[]
for r in rows:
    ref,pl=r["reference"],r["pseudo_katakana"]
    for op,i1,i2,j1,j2 in difflib.SequenceMatcher(None,ref,pl,autojunk=False).get_opcodes():
        if op=="equal": continue
        cost=max(i2-i1,j2-j1); a=ref[max(0,i1-1):i2+1]; b=pl[max(0,j1-1):j2+1]
        if norm(a)==norm(b): k="1 spelling-only"
        elif r["utt_id"] in DIGLAT: k="3 digits/Latin reading"
        elif RARE.search(ref[max(0,i1-1):i2+1]): k="2 rare mora in name"
        else:
            k="4 other"; rest.append(dict(utt=r["utt_id"],cref=ref[max(0,i1-3):i2+3],cpl=pl[max(0,j1-3):j2+3],asr=r["asr_text"],cost=cost))
        cat[k]+=cost
json.dump(rest,open("rest2.json","w",encoding="utf-8"),ensure_ascii=False,indent=0)
tot=sum(cat.values()); print("span cost total",tot,"(Levenshtein total 590)")
for k in sorted(cat): print(f"  {k:28s} {cat[k]:4d}  {100*cat[k]/tot:4.1f}%")
for i,x in enumerate(rest): print(i,x['utt'][-4:],x['cost'],x['cref'],'|',x['cpl'],'|',x['asr'])
