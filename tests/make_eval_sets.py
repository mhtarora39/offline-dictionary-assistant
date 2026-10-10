#!/usr/bin/env python3
"""
Generate dictionary-query eval sets with human-like typos.

For each split (dev, test) it writes TWO files so the model never sees labels:
  queries_<split>.jsonl   {"id", "query"}
  gold_<split>.jsonl      {"id", "type", "phrasing", "typo_kind", "expect",
                           "gold_surface", "gold_headwords", "zipf"}

  gold_surface   the term exactly as it appears in the query (typo included, case kept).
                 This is what an EXTRACTOR should return. null for no-term queries.
  gold_headwords headword(s) a correct LOOKUP should resolve to ([] = not found).
  expect         found | not_found | no_term

Dev and test use disjoint words and disjoint phrasing templates. Tune prompts and
rules on dev only. Never tune on test.

Usage
  pip install wordfreq
  python build_dictionary_db.py --out dictionary_eval.db --holdout 0.05 --seed 42
  python make_eval_sets.py --db dictionary_eval.db \
         --holdout-file dictionary_eval_holdout_words.txt --outdir eval_data
"""
import argparse
import json
import random
import re
import sqlite3
from collections import Counter
from pathlib import Path

from wordfreq import zipf_frequency

# --------------------------------------------------------------------------- config
SPEC = {
    # type: (test_count, dev_count)
    "found_clean": (150, 45),
    "found_typo": (140, 40),
    "inflected": (40, 12),
    "multiword": (30, 8),
    "noisy": (30, 10),
    "not_found_holdout": (40, 12),
    "not_found_pseudo": (20, 8),
    "no_term": (30, 10),
    "adversarial": (20, 5),
}

# Templates. {w} = the term. Category = phrasing style.
TEMPLATES_TEST = {
    "clean": [
        "what is the meaning of {w}", "what does {w} mean", "define {w}", "{w} meaning",
        "meaning of {w}", "what is {w}", "what's the definition of {w}", "{w} definition",
        "what does the word {w} mean", "can you tell me what {w} means",
        "tell me the meaning of {w}", "explain the word {w}", "explain {w} to me",
        "i want to know the meaning of {w}", "help me understand what {w} means",
        "meaning of the word {w}", "please define {w}", "what is meant by {w}",
        "dictionary meaning of {w}", "{w} meaning in english",
    ],
    "ungrammatical": [
        "what is {w} mean", "what {w} mean", "{w} mean what", "meaning {w}",
        "tell what means {w}", "what means {w}", "{w} is meaning", "say meaning of {w}",
        "tell me {w} meaning", "i wanna know what {w} means",
    ],
    "chatty": [
        "so like what does {w} actually mean", "hey quick question what does {w} mean",
        "um what is the meaning of {w} lol", "random but what does {w} mean",
        "ok so i keep seeing {w} what is it", "idk what {w} means can you help",
        "sorry stupid question but what does {w} mean", "just wondering what {w} means",
    ],
    "quoted": [
        "what does '{w}' mean", 'meaning of "{w}"', 'what is the meaning of "{w}"',
        "define '{w}'",
    ],
}
CAT_WEIGHTS_TEST = {"clean": 0.35, "ungrammatical": 0.2, "chatty": 0.25, "quoted": 0.2}

TEMPLATES_DEV = {
    "clean": ["what is the definition of {w}", "tell me about the word {w}", "{w} meaning please"],
    "ungrammatical": ["what meaning {w}", "{w} what is mean"],
    "chatty": ["hey what does {w} mean exactly", "can u tell me what {w} is"],
    "quoted": ["what does the word '{w}' mean"],
}
CAT_WEIGHTS_DEV = CAT_WEIGHTS_TEST

NOISY_TEST = [
    "wat does {w} mean", "whats {w} mean", "meaning of {w} plz", "pls tell meaning of {w}",
    "{w} meaning plz", "wat is {w}", "wht is the meening of {w}", "tel me {w} meaning",
    "whts the meaning of {w}", "wat is the maening of {w}", "{w} mening", "defn of {w}",
]
NOISY_DEV = ["wat is the meaning of {w}", "whats the meening of {w}", "{w} meaning pls"]

INTENT_TYPOS = {
    "what": ["wat", "wht", "waht", "whta"], "what's": ["whats", "wat's", "whts"],
    "whats": ["wats", "whts"], "meaning": ["meening", "maening", "meanin", "meaing", "menaing"],
    "mean": ["mena", "meen", "mean"], "means": ["mens", "meens"],
    "definition": ["definiton", "defenition", "definatoin"], "define": ["defien", "defin", "defne"],
    "explain": ["explian", "explane", "expain"], "the": ["teh", "th", "hte"],
    "does": ["dose", "des"], "tell": ["tel", "tlel"], "is": ["iz", "si"], "of": ["ofr", "fo"],
    "please": ["plese", "pls", "plz"], "word": ["wrod", "wrd"], "me": ["em"],
}

NO_TERM = [
    "is it going to rain tomorrow", "hello", "hi there", "tell me a joke",
    "how are you doing", "what time is it", "thanks a lot", "who is the prime minister of india",
    "play some music", "what is 2 plus 2", "set an alarm for 7am", "good morning",
    "how do i reset my password", "translate this to french", "can you help me",
    "what can you do", "how far is the moon", "remind me to call mom", "i am hungry",
    "open the settings", "tell me something interesting", "whats up", "ok", "cool", "lol",
    "capital of france", "convert 10 km to miles", "write me a poem", "who are you", "bye",
    "how tall is mount everest", "what day is today", "turn off the lights", "good night",
    "send a message to dad", "how do i boil an egg", "take a photo", "what's your name",
    "call a taxi", "sing me a song",
]

# (template, term). Terms that are themselves intent/stop words.
ADVERSARIAL = [
    ("what does {w} mean", "mean"), ("meaning of {w}", "meaning"),
    ("what is the meaning of the word {w}", "word"), ("what does the word {w} mean", "the"),
    ("what is the meaning of {w}", "is"), ("what does {w} mean", "what"),
    ("definition of {w}", "definition"), ("what does {w} mean", "of"),
    ("meaning of {w}", "does"), ("what is the meaning of {w}", "explain"),
    ("what does the word {w} mean", "please"), ("meaning of {w}", "tell"),
    ("what does {w} mean", "define"), ("what does {w} mean", "me"),
    ("what does {w} mean", "and"), ("what does {w} mean", "a"),
    ("tell me what {w} means", "tell"), ("explain the word {w}", "explain"),
    ("what is {w}", "define"), ("meaning of the word {w}", "meaning"),
    ("what does {w} mean", "is"), ("define {w}", "the"), ("{w} meaning", "what"),
    ("what is the definition of {w}", "mean"), ("what does the word {w} mean", "does"),
]

# Common human misspellings: (misspelling, correct). Invalid pairs are filtered by DB checks.
COMMON_MISSPELLINGS = """definately definitely|seperate separate|recieve receive|occured occurred|
accomodate accommodate|untill until|wierd weird|neccessary necessary|embarass embarrass|
goverment government|tommorow tomorrow|calender calendar|beleive believe|existance existence|
independant independent|millenium millennium|noticable noticeable|occassion occasion|
publically publicly|refrence reference|succesful successful|supercede supersede|tendancy tendency|
truely truly|aquire acquire|adress address|agressive aggressive|apparant apparent|
arguement argument|basicly basically|begining beginning|buisness business|catagory category|
cemetary cemetery|collegue colleague|comming coming|commitee committee|concious conscious|
curiousity curiosity|dissapear disappear|dilemna dilemma|enviroment environment|
exagerate exaggerate|familar familiar|finaly finally|foriegn foreign|fourty forty|freind friend|
gaurd guard|grammer grammar|happend happened|harrass harass|humourous humorous|
immediatly immediately|intresting interesting|knowlege knowledge|liason liaison|lisence license|
maintainance maintenance|mischievious mischievous|neice niece|occurence occurrence|
parliment parliament|persistant persistent|posession possession|prefered preferred|
privelege privilege|pronounciation pronunciation|questionaire questionnaire|rythm rhythm|
sargent sergeant|sieze seize|speach speech|strenght strength|suprise surprise|thier their|
threshhold threshold|tounge tongue|twelth twelfth|unforseen unforeseen|vaccum vacuum|
vegtable vegetable|writting writing|yeild yield|wich which|alot allot|recomend recommend|
embarassed embarrassed|managment management|medecine medicine|temperture temperature|
tomatoe tomato|accross across|adolesent adolescent|athiest atheist|bizzare bizarre|
bouyant buoyant|camoflage camouflage|conscence conscience|desparate desperate|
dictionnary dictionary|equiptment equipment|exellent excellent|facinate fascinate|
fluorescant fluorescent|guarentee guarantee|hieght height|hierarchy hierachy|ignorence ignorance|
innoculate inoculate|jewelery jewellery|judgement judgment|konw know|lenght length|
mathematic mathematics|miniscule minuscule|ocassion occasion|paralel parallel|
playwrite playwright|potatoe potato|relevent relevant|religous religious|resistence resistance|
restaraunt restaurant|schedual schedule|sentance sentence|sucess success|tecnology technology""".replace("\n", "")

# Random dictionary sampling can surface vulgar or slur terms; keep them out of shared test files.
BLOCK = {"dyke", "douche", "porn", "pornography", "pornographic", "scrotal", "skin", "fag", "faggot",
         "nigger", "negro", "slut", "whore", "bitch", "bastard", "cunt", "dick", "cock", "pussy",
         "tit", "tits", "dildo", "rape", "rapist", "kike", "spic", "wop", "gook", "chink", "jap",
         "retard", "retarded", "cripple", "tranny", "shemale", "hooker", "prostitute", "orgasm",
         "masturbate", "penis", "vagina", "anus", "rectum", "semen", "sperm", "suicide", "genocide",
         "nazi", "heroin", "cocaine", "crack"}


def ok(word):
    return not any(t in BLOCK for t in word.lower().split())


QWERTY_ROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
PHONETIC = [("ie", "ei"), ("ei", "ie"), ("ph", "f"), ("f", "ph"), ("tion", "sion"),
            ("sion", "tion"), ("able", "ible"), ("ible", "able"), ("ent", "ant"), ("ant", "ent"),
            ("ence", "ance"), ("ance", "ence"), ("ck", "k"), ("c", "s"), ("s", "c"),
            ("ll", "l"), ("ss", "s"), ("tt", "t"), ("mm", "m"), ("nn", "n"), ("rr", "r"),
            ("cc", "c"), ("pp", "p"), ("l", "ll"), ("t", "tt"), ("s", "ss")]


def neighbors():
    pos = {}
    for r, row in enumerate(QWERTY_ROWS):
        for c, ch in enumerate(row):
            pos[ch] = (r, c)
    out = {}
    for ch, (r, c) in pos.items():
        ns = []
        for dr, dc in [(0, -1), (0, 1), (-1, 0), (-1, 1), (1, -1), (1, 0)]:
            rr, cc = r + dr, c + dc
            if 0 <= rr < 3 and 0 <= cc < len(QWERTY_ROWS[rr]):
                ns.append(QWERTY_ROWS[rr][cc])
        out[ch] = ns
    return out


NEIGH = neighbors()


def osa(a, b):
    """Optimal string alignment distance (adjacent transposition = 1)."""
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        d[i][0] = i
    for j in range(len(b) + 1):
        d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[-1][-1]


# ------------------------------------------------------------------------ typo ops
def t_transpose(w, rng):
    idx = [i for i in range(1, len(w) - 1) if w[i] != w[i + 1]]
    if not idx:
        return None
    i = rng.choice(idx)
    return w[:i] + w[i + 1] + w[i] + w[i + 2:]


def t_omit(w, rng):
    i = rng.randrange(1, len(w))
    return w[:i] + w[i + 1:]


def t_dup(w, rng):
    i = rng.randrange(1, len(w))
    return w[:i + 1] + w[i] + w[i + 1:]


def t_adjacent(w, rng):
    i = rng.randrange(1, len(w))
    ns = NEIGH.get(w[i])
    if not ns:
        return None
    return w[:i] + rng.choice(ns) + w[i + 1:]


def t_vowel(w, rng):
    idx = [i for i in range(1, len(w) - 1) if w[i] in "aeiou"]
    if not idx:
        return None
    i = rng.choice(idx)
    return w[:i] + rng.choice([v for v in "aeiou" if v != w[i]]) + w[i + 1:]


def t_phonetic(w, rng):
    opts = [(a, b, m.start()) for a, b in PHONETIC for m in re.finditer(a, w) if m.start() > 0]
    if not opts:
        return None
    a, b, i = rng.choice(opts)
    return w[:i] + b + w[i + len(a):]


TYPO_OPS = [("transpose", t_transpose, 0.2), ("omit", t_omit, 0.2), ("dup", t_dup, 0.1),
            ("adjacent_key", t_adjacent, 0.15), ("vowel", t_vowel, 0.15),
            ("phonetic", t_phonetic, 0.2)]


def edits1(w):
    """All strings one edit (insert, delete, substitute, adjacent swap) away."""
    letters = "abcdefghijklmnopqrstuvwxyz"
    splits = [(w[:i], w[i:]) for i in range(len(w) + 1)]
    out = {a + b[1:] for a, b in splits if b}
    out |= {a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1}
    out |= {a + c + b[1:] for a, b in splits if b for c in letters}
    out |= {a + c + b for a, b in splits for c in letters}
    return out


def make_typo(w, rng, valid_forms):
    for _ in range(40):
        name, fn, _ = rng.choices(TYPO_OPS, weights=[o[2] for o in TYPO_OPS])[0]
        t = fn(w, rng)
        if t and t != w and t.isalpha() and t not in valid_forms and osa(t, w) <= 2:
            # reject if another headword is also one edit away (gold would be ambiguous)
            others = {h for e in edits1(t) for h in valid_forms.get(e, ()) if h != w}
            if others:
                continue
            return t, name
    return None, None


def mutate_token(tok, rng):
    core = tok.lower().strip("?.,!'\"")
    if core in INTENT_TYPOS:
        return tok.lower().replace(core, rng.choice(INTENT_TYPOS[core]))
    if len(core) >= 4 and core.isalpha():
        t = (t_transpose(core, rng) if rng.random() < 0.5 else t_omit(core, rng)) or core
        return tok.lower().replace(core, t)
    return tok


# ------------------------------------------------------------------- query builder
def build_query(template, term, rng, p_intent_typo=0.0, p_join=0.0, p_upper=0.03, p_capfirst=0.04):
    """Returns (query, surface). surface = term slice exactly as it appears in query."""
    toks = template.split()
    ti = next(i for i, t in enumerate(toks) if "{w}" in t)
    # typos in non-term tokens
    for i, t in enumerate(toks):
        if i != ti and rng.random() < p_intent_typo:
            toks[i] = mutate_token(t, rng)
    # missing space between two non-term tokens
    i = 0
    while i < len(toks) - 1:
        if i != ti and i + 1 != ti and rng.random() < p_join:
            toks[i] += toks[i + 1]
            del toks[i + 1]
            if ti > i:
                ti -= 1
        i += 1
    toks[ti] = toks[ti].replace("{w}", term)
    start = sum(len(t) + 1 for t in toks[:ti]) + toks[ti].find(term)
    q = " ".join(toks)
    r = rng.random()
    if r < p_upper:
        q = q.upper()
    elif r < p_upper + p_capfirst:
        q = q[0].upper() + q[1:]
    if not q.endswith(("?", "'", '"')) and rng.random() < 0.4:
        q += rng.choice(["?", "?", "?", "??", ".", "!"])
    if rng.random() < 0.02:  # stray double space after first token
        q = q.replace(" ", "  ", 1)
        if start >= len(toks[0]) + 1:
            start += 1
    surface = q[start:start + len(term)]
    assert surface.lower() == term.lower(), (q, term, surface)
    return q, surface


def pick_template(rng, templates, weights):
    cat = rng.choices(list(weights), weights=list(weights.values()))[0]
    return cat, rng.choice(templates[cat])


# --------------------------------------------------------------------------- data
class Dict:
    def __init__(self, db_path):
        db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        self.forms = {}
        for form, head in db.execute(
                "SELECT f.form, e.headword FROM forms f JOIN entries e ON e.id=f.entry_id"):
            self.forms.setdefault(form, set()).add(head)
        self.heads = {h for h in (e[0] for e in db.execute("SELECT DISTINCT headword FROM entries"))}
        db.close()

    def lookup(self, term):
        return sorted(self.forms.get(term.lower(), []))


def pseudo_word(rng):
    on = ["bl", "fl", "gr", "pr", "zor", "kr", "sn", "vl", "dr", "qu", "th", "z", "v", "gl"]
    nu = ["a", "e", "i", "o", "u", "ai", "ou", "ea"]
    co = ["n", "x", "rp", "lt", "mb", "sk", "nd", "gg", "st", "ft", "rn"]
    return "".join(rng.choice(on) + rng.choice(nu) + rng.choice(co) for _ in range(rng.choice([2, 3])))


def generate(split, counts, d, holdout_words, used, rng):
    T = TEMPLATES_TEST if split == "test" else TEMPLATES_DEV
    W = CAT_WEIGHTS_TEST
    NOISY = NOISY_TEST if split == "test" else NOISY_DEV
    rows = []
    seen_q = set()

    # word pools (single alphabetic headwords), bucketed by zipf
    single = [h for h in d.heads if re.fullmatch(r"[a-z]{4,}", h) and h not in used and ok(h)]
    z = {h: zipf_frequency(h, "en") for h in single}
    pool = {"common": [h for h in single if z[h] >= 4.5],
            "mid": [h for h in single if 3.2 <= z[h] < 4.5],
            "rare": [h for h in single if 2.0 <= z[h] < 3.2]}
    for v in pool.values():
        v.sort()
        rng.shuffle(v)

    def take(bucket):
        while pool[bucket]:
            w = pool[bucket].pop()
            if w not in used and w not in ("black", "weather"):
                used.add(w)
                return w
        raise RuntimeError("pool exhausted: " + bucket)

    def add(typ, query, surface, expect, heads, phrasing, kind=None, zp=None):
        if query.lower() in seen_q:
            return False
        seen_q.add(query.lower())
        rows.append({"type": typ, "phrasing": phrasing, "typo_kind": kind, "expect": expect,
                     "query": query, "gold_surface": surface, "gold_headwords": heads,
                     "zipf": None if zp is None else round(zp, 2)})
        return True

    def emit(typ, term, heads, expect, phrasing_templates=None, p_typo=0.1, p_join=0.02, kind=None, zp=None):
        for _ in range(20):
            if phrasing_templates is not None:
                tpl, cat = rng.choice(phrasing_templates), "noisy"
            else:
                cat, tpl = pick_template(rng, T, W)
            q, surf = build_query(tpl, term, rng, p_intent_typo=p_typo, p_join=p_join)
            if add(typ, q, surf, expect, heads, cat, kind, zp):
                return
        raise RuntimeError("could not emit unique query for " + term)

    n = lambda t: counts[t][0 if split == "test" else 1]

    # found_clean: 1/3 each bucket
    k = n("found_clean")
    for i in range(k):
        b = ["common", "mid", "rare"][i % 3]
        w = take(b)
        emit("found_clean", w, [w], "found", p_typo=0.0, p_join=0.0, zp=z[w])

    # found_typo: common misspellings first, then algorithmic
    cm = []
    for pair in COMMON_MISSPELLINGS.split("|"):
        pair = pair.strip().split()
        if len(pair) == 2:
            bad, good = pair
            if good in d.heads and bad not in d.forms and good not in used and ok(good):
                cm.append((bad, good))
    rng.shuffle(cm)
    share = 50 if split == "test" else 12
    ktypo = n("found_typo")
    for bad, good in cm[:min(share, ktypo)]:
        used.add(good)
        emit("found_typo", bad, [good], "found", p_typo=0.05, kind="common_misspelling", zp=zipf_frequency(good, "en"))
    while sum(1 for r in rows if r["type"] == "found_typo") < ktypo:
        b = rng.choice(["common", "mid", "mid", "rare"])
        w = take(b)
        if len(w) < 5:
            continue
        t, kind = make_typo(w, rng, d.forms)
        if not t:
            continue
        emit("found_typo", t, [w], "found", p_typo=0.05, kind=kind, zp=z[w])

    # inflected: half irregular-looking, half regular; keep forms mapping to one headword
    infl = [(f, next(iter(h))) for f, h in d.forms.items()
            if len(h) == 1 and re.fullmatch(r"[a-z]{3,}", f) and f not in d.heads and f not in used
            and next(iter(h)) not in used and ok(f) and ok(next(iter(h))) and zipf_frequency(f, "en") >= 3.0]
    irregular = [(f, h) for f, h in infl if not f.startswith(h[:-1])]
    regular = [(f, h) for f, h in infl if f.startswith(h[:-1])]
    for lst in (irregular, regular):
        lst.sort()
        rng.shuffle(lst)
    k = n("inflected")
    for i in range(k):
        lst = irregular if i % 2 == 0 else regular
        f, h = lst.pop()
        used.update([f, h])
        emit("inflected", f, [h], "found", p_typo=0.03, kind="irregular" if lst is irregular else "regular",
             zp=zipf_frequency(f, "en"))

    # multiword headwords
    mw = [h for h in d.heads if 1 < len(h.split()) <= 3 and re.fullmatch(r"[a-z]+( [a-z]+)+", h)
          and all(zipf_frequency(t, "en") >= 3.0 for t in h.split()) and h != "kick the bucket" and h not in used and ok(h)]
    mw.sort()
    rng.shuffle(mw)
    for _ in range(n("multiword")):
        h = mw.pop()
        used.add(h)
        emit("multiword", h, [h], "found", p_typo=0.03)

    # noisy phrasing on clean words
    for i in range(n("noisy")):
        w = take(["common", "mid"][i % 2])
        emit("noisy", w, [w], "found", phrasing_templates=NOISY, p_typo=0.5, p_join=0.15, zp=z[w])

    # not found: real hidden words and invented words
    hw = [w for w in holdout_words if re.fullmatch(r"[a-z]{5,}", w) and zipf_frequency(w, "en") >= 2.0
          and w not in d.forms and w not in used and ok(w)]
    hw.sort()
    rng.shuffle(hw)
    for _ in range(n("not_found_holdout")):
        w = hw.pop()
        used.add(w)
        emit("not_found_holdout", w, [], "not_found", p_typo=0.03, zp=zipf_frequency(w, "en"))
    for _ in range(n("not_found_pseudo")):
        while True:
            w = pseudo_word(rng)
            if w not in d.forms and zipf_frequency(w, "en") == 0 and w not in used:
                break
        used.add(w)
        emit("not_found_pseudo", w, [], "not_found", p_typo=0.03)

    # no-term queries (disjoint between splits by slicing)
    nt = NO_TERM[:30] if split == "test" else NO_TERM[30:]
    for q in nt[:n("no_term")]:
        add("no_term", q, None, "no_term", [], "no_term")

    # adversarial: the term is itself an intent or function word
    adv = ADVERSARIAL[:20] if split == "test" else ADVERSARIAL[20:]
    for tpl, term in adv[:n("adversarial")]:
        heads = d.lookup(term)
        q, surf = build_query(tpl, term, rng)
        add("adversarial", q, surf, "found" if heads else "not_found", heads, "adversarial")

    rng.shuffle(rows)
    for i, r in enumerate(rows, 1):
        r["id"] = f"{split[0]}{i:04d}"
    return rows


def write(rows, outdir, split):
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / f"queries_{split}.jsonl", "w", encoding="utf8") as fq, \
            open(outdir / f"gold_{split}.jsonl", "w", encoding="utf8") as fg:
        for r in rows:
            fq.write(json.dumps({"id": r["id"], "query": r["query"]}, ensure_ascii=False) + "\n")
            g = {k: r[k] for k in ("id", "type", "phrasing", "typo_kind", "expect",
                                   "gold_surface", "gold_headwords", "zipf")}
            fg.write(json.dumps(g, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="dictionary_eval.db (built with --holdout)")
    ap.add_argument("--holdout-file", required=True)
    ap.add_argument("--outdir", default="eval_data")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    d = Dict(a.db)
    holdout = [l.strip() for l in open(a.holdout_file, encoding="utf8") if l.strip()]
    used = set()
    out = Path(a.outdir)
    for split, seed_off in (("dev", 1), ("test", 2)):  # dev first so test avoids its words
        rng = random.Random(a.seed * 1000 + seed_off)
        rows = generate(split, SPEC, d, holdout, used, rng)
        write(rows, out, split)
        c = Counter(r["type"] for r in rows)
        print(f"{split}: {len(rows)} rows  ", dict(c))

        # sanity checks
        for r in rows:
            if r["expect"] == "found" and r["typo_kind"] is None:
                assert d.lookup(r["gold_surface"]), ("surface not in dict", r)
            if r["expect"] == "not_found":
                assert not r["gold_headwords"], r
            if r["typo_kind"] and r["typo_kind"] != "irregular" and r["typo_kind"] != "regular":
                assert not d.lookup(r["gold_surface"]), ("typo is a valid form", r)
        assert len({r["query"].lower() for r in rows}) == len(rows)
    print("wrote to", out.resolve())


if __name__ == "__main__":
    main()
