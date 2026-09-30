"""
Text normalization for business names and addresses.

Everything that touches raw strings lives here. Heavy lifting is done with
vectorized polars string expressions; only non-ASCII tokens (Indic scripts,
accented Latin) go through a cached Python transliteration function.

Transliteration of Indic-script tokens uses a dictionary *learned from the
training ground truth* (native token -> Latin token, via positional alignment
of matched name pairs), with a unidecode-based fallback for unseen tokens.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

import polars as pl
from unidecode import unidecode

# ---------------------------------------------------------------------------
# Vocabularies
# ---------------------------------------------------------------------------

# Legal forms, honorifics and filler words. Removed from the "core" name used
# for blocking and core-name similarity (kept in the full name).
NAME_STOPWORDS = {
    # English / US
    "llc", "inc", "incorporated", "corp", "corporation", "co", "company", "ltd",
    "limited", "lp", "llp", "pllc", "pc", "plc", "the", "of", "and", "a", "an",
    "dba", "fka", "aka", "l", "p", "c",
    # India
    "pvt", "private", "pvtltd", "ms", "m", "s", "smt", "shri", "sri", "shree",
    "opc", "firm", "prop", "proprietor",
    # France
    "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "scp", "ets",
    "ei", "eirl", "selarl", "gie", "sarlu", "earl",
    "etablissements", "du", "de", "des", "la", "le", "les", "et", "d",
    # web
    "com", "www", "net", "org", "in", "fr",
}

LEGAL_CANON = {
    "incorporated": "inc", "corporation": "corp", "company": "co",
    "limited": "ltd", "private": "pvt",
    # French equivalents of "& Co" (a legal-style suffix, not a business word)
    "cie": "co", "compagnie": "co",
}

# variant -> canonical (short) form
ADDR_ABBREV = {
    # US street types
    "street": "st", "str": "st", "saint": "st", "road": "rd", "avenue": "ave", "av": "ave",
    "boulevard": "blvd", "bd": "blvd", "drive": "dr", "lane": "ln", "court": "ct",
    "highway": "hwy", "parkway": "pkwy", "place": "pl", "circle": "cir", "terrace": "ter",
    "trail": "trl", "square": "sq", "crescent": "cres", "mount": "mt", "fort": "ft",
    "apartment": "apt", "suite": "ste", "floor": "fl", "flr": "fl", "building": "bldg",
    "route": "rte", "rt": "rte", "heights": "hts", "junction": "jct", "expressway": "expy",
    "freeway": "fwy", "turnpike": "tpke", "cove": "cv", "crossing": "xing", "point": "pt",
    "north": "n", "south": "s", "east": "e", "west": "w",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
    "township": "twp", "county": "cnty", "city": "",
    # India
    "near": "nr", "opposite": "opp", "market": "mkt", "colony": "clny", "district": "dist",
    "taluk": "tq", "tal": "tq", "sector": "sec", "phase": "ph", "number": "no",
    # France
    "r": "rue", "imp": "impasse", "che": "chemin", "ch": "chemin", "allee": "allee",
    "all": "allee", "fbg": "faubourg", "qu": "quai", "crs": "cours",
    "res": "residence", "zi": "zone", "za": "zone",
}

ADDR_STOPWORDS = {"na", "po", "box", "no", "unit", "ste", "fl", "the", "of", "and", "nr", "opp",
                  "du", "des", "la", "le", "les", "d", "l", "et", "chez", "mme", "m"}

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi",
    "wyoming": "wy", "district of columbia": "dc",
}

ORDINAL_WORDS = {
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5", "sixth": "6",
    "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10", "eleventh": "11",
    "twelfth": "12", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
}

_INDIC_RE = re.compile(r"[ऀ-෿]")
_TOKEN_STRIP = re.compile(r"^[^\wऀ-෿]+|[^\wऀ-෿]+$")


# ---------------------------------------------------------------------------
# Transliteration
# ---------------------------------------------------------------------------

class Transliterator:
    """Native-script token -> Latin token mapping, learned + unidecode fallback."""

    def __init__(self, mapping: Optional[Dict[str, str]] = None):
        self.mapping = mapping or {}
        self._cache: Dict[str, str] = {}

    def token(self, tok: str) -> str:
        hit = self._cache.get(tok)
        if hit is not None:
            return hit
        core = _TOKEN_STRIP.sub("", tok)
        if core in self.mapping:
            out = self.mapping[core]
        elif _INDIC_RE.search(tok):
            out = _phonetic_fallback(unidecode(core))
        else:
            out = unidecode(tok)
        if len(self._cache) < 2_000_000:
            self._cache[tok] = out
        return out

    def text(self, s: str) -> str:
        if s.isascii():
            return s
        return " ".join(self.token(t) for t in s.split())

    # -- persistence -------------------------------------------------------
    def save(self, path: Path):
        path.write_text(json.dumps(self.mapping, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Transliterator":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))


def _phonetic_fallback(s: str) -> str:
    """Clean unidecode output of Indic text towards English spelling."""
    s = s.lower()
    s = re.sub(r"[^a-z]", "", s)
    s = s.replace("aa", "a").replace("ii", "i").replace("uu", "u")
    s = re.sub(r"(.)\1+", r"\1", s)
    return s


def learn_translit(s1: pl.DataFrame, targets: pl.DataFrame, gt: pl.DataFrame,
                   min_count: int = 2) -> Transliterator:
    """Learn native->Latin token map from matched pairs with equal token counts."""
    pairs = (
        gt.select(pl.col("source1_entity_id").alias("s1"),
                  pl.col("matched_entity_ids").str.split(",").alias("t"))
        .explode("t").filter(pl.col("t") != "")
        .join(targets.select(pl.col("entity_id").alias("t"), pl.col("business_name").alias("tn")), on="t")
        .filter(pl.col("tn").str.contains(r"[ऀ-෿]"))
        .join(s1.select(pl.col("entity_id").alias("s1"), pl.col("business_name").alias("sn")), on="s1")
    )
    return Transliterator(_align(pairs.select("sn", "tn").iter_rows(), min_count, 0.5))


def _align(name_pairs, min_count: int, min_share: float) -> Dict[str, str]:
    """Positional alignment of (latin_name, native_name) pairs with equal token counts."""
    counts: Dict[str, Counter] = defaultdict(Counter)
    for sn, tn in name_pairs:
        lt = [t for t in re.sub(r"[^a-z0-9 ]", " ", sn.lower().replace("&", " and ")).split()]
        nt = [c for c in (_TOKEN_STRIP.sub("", t) for t in tn.split()) if c]
        if len(lt) != len(nt):
            continue
        for a, b in zip(nt, lt):
            if _INDIC_RE.search(a):
                counts[a][b] += 1
    mapping = {}
    for tok, c in counts.items():
        best, n = c.most_common(1)[0]
        if n >= min_count and n >= min_share * sum(c.values()):
            mapping[tok] = best
    return mapping


# ---------------------------------------------------------------------------
# Vectorized normalization
# ---------------------------------------------------------------------------

def _translit_col(col: pl.Series, tr: Transliterator) -> pl.Series:
    return col.map_elements(tr.text, return_dtype=pl.String, skip_nulls=True)


def _replace_words(expr: pl.Expr, mapping: Dict[str, str]) -> pl.Expr:
    """Whole-word replacement on space-separated lowercase text."""
    multi = {k: v for k, v in mapping.items() if " " in k}
    for k, v in multi.items():
        expr = expr.str.replace_all(rf"\b{k}\b", v)
    single = {k: v for k, v in mapping.items() if " " not in k}
    keys = list(single)
    # double the spaces so adjacent words each own their delimiters
    padded = pl.lit(" ") + expr.str.replace_all(" ", "  ") + pl.lit(" ")
    out = padded.str.replace_many([f" {k} " for k in keys], [f" {single[k]} " for k in keys])
    return out.str.replace_all(r"\s+", " ").str.strip_chars()


def normalize_names(names: pl.Series, tr: Transliterator) -> pl.DataFrame:
    """Return DataFrame(name_full, name_core) for a Series of raw names."""
    s = _translit_col(names, tr)
    df = pl.DataFrame({"s": s}).lazy()
    e = pl.col("s").str.to_lowercase()
    e = e.str.replace_all(r"https?://|www\.", " ")
    e = e.str.replace_all(r"\.(com|net|org|in|co|fr|biz|info)\b", " ")
    e = e.str.replace_all(r"'", "")
    e = e.str.replace_all("&", " and ").str.replace_all(r"\+", " and ")
    # OCR-like digit substitutions inside words
    for _ in range(2):
        e = e.str.replace_all(r"([a-z])0", "${1}o").str.replace_all(r"0([a-z])", "o${1}")
        e = e.str.replace_all(r"([a-z])1", "${1}l").str.replace_all(r"1([a-z])", "l${1}")
    e = e.str.replace_all(r"[^a-z0-9]+", " ").str.strip_chars()
    df = df.with_columns(e.alias("name_full"))
    # canonical legal forms then strip stopwords for core
    full = _replace_words(pl.col("name_full"), LEGAL_CANON)
    df = df.with_columns(full.alias("name_full"))
    stop = sorted(NAME_STOPWORDS | set(LEGAL_CANON.values()))
    core = (pl.col("name_full").str.split(" ")
            .list.eval(pl.element().filter(~pl.element().is_in(stop) & (pl.element() != "")))
            .list.join(" "))
    df = df.with_columns(core.alias("name_core"))
    # if everything was a stopword keep full name
    df = df.with_columns(pl.when(pl.col("name_core") == "").then(pl.col("name_full"))
                         .otherwise(pl.col("name_core")).alias("name_core"))
    return df.select("name_full", "name_core").collect()


# Country subdivisions: S1 writes India states in English ("Maharashtra") and
# France regions ("Nouvelle-Aquitaine"); S2/S3 often write the state code
# ("MH"), a native-script name, or the French department ("Gironde").
IN_STATES = {
    "maharashtra": "st_mh", "mh": "st_mh", "delhi": "st_dl", "dl": "st_dl", "uttar pradesh": "st_up",
    "up": "st_up", "karnataka": "st_ka", "ka": "st_ka", "tamil nadu": "st_tn", "tn": "st_tn",
    "west bengal": "st_wb", "wb": "st_wb", "gujarat": "st_gj", "gj": "st_gj", "telangana": "st_tg",
    "tg": "st_tg", "ts": "st_tg", "haryana": "st_hr", "hr": "st_hr", "rajasthan": "st_rj", "rj": "st_rj",
    "kerala": "st_kl", "keralam": "st_kl", "kl": "st_kl", "madhya pradesh": "st_mp", "mp": "st_mp",
    "bihar": "st_br", "br": "st_br", "andhra pradesh": "st_ap", "ap": "st_ap", "punjab": "st_pb",
    "pb": "st_pb", "odisha": "st_od", "orissa": "st_od", "od": "st_od", "or": "st_od", "assam": "st_as",
    "as": "st_as", "jharkhand": "st_jh", "jh": "st_jh", "chandigarh": "st_ch", "ch": "st_ch", "goa": "st_ga",
    "ga": "st_ga", "himachal pradesh": "st_hp", "hp": "st_hp", "uttarakhand": "st_uk", "uttaranchal": "st_uk",
    "uk": "st_uk", "jammu and kashmir": "st_jk", "jk": "st_jk", "chhattisgarh": "st_cg", "cg": "st_cg",
    "ct": "st_cg", "puducherry": "st_py", "pondicherry": "st_py", "py": "st_py",
}
# Historical / alternate Indian city names -> one canonical form (the generator
# swaps e.g. Mumbai -> Bombay in S2/S3 addresses).  India rows only.
IN_CITY_ALIASES = {
    "bombay": "mumbai", "calcutta": "kolkata", "madras": "chennai", "bangalore": "bengaluru",
    "gurgaon": "gurugram", "poona": "pune", "baroda": "vadodara", "trivandrum": "thiruvananthapuram",
    "cochin": "kochi", "mysore": "mysuru", "benares": "varanasi", "banaras": "varanasi",
    "allahabad": "prayagraj", "calicut": "kozhikode", "mangalore": "mangaluru", "belgaum": "belagavi",
    "cawnpore": "kanpur", "simla": "shimla", "hubli": "hubballi", "vizag": "visakhapatnam",
    "waltair": "visakhapatnam", "gauhati": "guwahati", "jubbulpore": "jabalpur", "ootacamund": "ooty",
}
FR_REGIONS = {
    "hauts de france": "rg_hdf", "nord": "rg_hdf", "pas de calais": "rg_hdf",
    "nouvelle aquitaine": "rg_naq", "gironde": "rg_naq",
    "pays de la loire": "rg_pdl", "loire atlantique": "rg_pdl",
}
NATIVE_STATES: Dict[str, str] = {}      # learned: native-script state name -> English (see learn_native_states)


def learn_native_states(pairs_last_components, min_count: int = 20, min_share: float = 0.6) -> Dict[str, str]:
    """(target last address component, S1 last address component) of true pairs -> native->English map."""
    counts: Dict[str, Counter] = defaultdict(Counter)
    for tgt, s1c in pairs_last_components:
        tgt = tgt.strip()
        if tgt and _INDIC_RE.search(tgt):
            counts[tgt][s1c.strip()] += 1
    out = {}
    for k, c in counts.items():
        best, n = c.most_common(1)[0]
        if n >= min_count and n >= min_share * sum(c.values()):
            out[k] = best
    return out


def normalize_addresses(addrs: pl.Series, tr: Transliterator, country: pl.Series | None = None) -> pl.Series:
    if NATIVE_STATES:
        keys = sorted(NATIVE_STATES, key=len, reverse=True)
        addrs = addrs.str.replace_many(keys, [NATIVE_STATES[k] for k in keys])
    s = _translit_col(addrs, tr)
    cty = country if country is not None else pl.Series([""] * len(s))
    df = pl.DataFrame({"s": s, "cty": cty}).lazy()
    e = pl.col("s").str.to_lowercase()
    e = e.str.replace_all(r"'", " ")
    e = e.str.replace_all(r"[^a-z0-9/\-]+", " ")
    e = e.str.replace_all(r"(\d+)(st|nd|rd|th)\b", "${1}")
    e = e.str.replace_all(r"\b0+(\d)", "${1}")                # 0098 -> 98
    e = e.str.replace_all(r"(\d)[/\-](\d)", "${1}_${2}")      # keep 13/410 together
    e = e.str.replace_all(r"[/\-]", " ")
    e = e.str.replace_all(r"_", "/")
    e = e.str.replace_all(r"\s+", " ").str.strip_chars()
    df = df.with_columns(e.alias("a"))
    df = df.with_columns(pl.when(pl.col("cty") == "India").then(_replace_words(pl.col("a"), {**IN_STATES, **IN_CITY_ALIASES}))
                         .when(pl.col("cty") == "France").then(_replace_words(
                             pl.col("a").str.replace_all(r"\b(n\s?deg|ndeg|numero|bis|ter|quater)\b", " ")
                             .str.replace_all(r"\s+", " ").str.strip_chars(), FR_REGIONS))
                         .otherwise(pl.col("a")).alias("a"))
    df = df.with_columns(_replace_words(pl.col("a"), US_STATES).alias("a"))
    df = df.with_columns(_replace_words(pl.col("a"), ORDINAL_WORDS).alias("a"))
    df = df.with_columns(_replace_words(pl.col("a"), ADDR_ABBREV).alias("a"))
    df = df.with_columns(pl.col("a").str.replace_all("_", ""))
    return df.collect()["a"]


def normalize_frame(df: pl.DataFrame, tr: Transliterator) -> pl.DataFrame:
    """Add name_full, name_core, addr columns to a raw source frame."""
    names = normalize_names(df["business_name"], tr)
    addr = normalize_addresses(df["business_address"], tr, df["country"])
    return df.with_columns(names["name_full"], names["name_core"], addr.alias("addr"))
