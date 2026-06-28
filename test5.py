import json
import re
import requests
import sys

from rdflib import Graph
from transformers import AutoTokenizer, AutoModelForCausalLM

##############################################################################
# CONFIG
##############################################################################

UAT_FILE = "aas_the-unified-astronomy-thesaurus_6-0-0.ttl"
MAPPING_FILE = "uat_wikidata_mapping_updated.ttl"

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"

##############################################################################
# WIKIDATA PROPERTY CATALOGUE
# The LLM chooses from this set – keeps hallucinated P-IDs out.
##############################################################################

KNOWN_WD_PROPERTIES = {
    # Astronomical relations
    "P397": "parent astronomical body (e.g. planet a moon orbits)",
    "P398": "child astronomical body (moons of a planet)",
    "P31":  "instance of (what class/type an item is)",
    "P279": "subclass of (broader class)",
    "P361": "part of (larger system)",
    "P527": "has part (components / members)",
    # Physical properties
    "P2067": "mass",
    "P2386": "diameter",
    "P2120": "radius",
    "P2243": "apoapsis",
    "P2244": "periapsis",
    "P2583": "distance from Earth",
    "P1096": "orbital eccentricity",
    "P2147": "orbital period",
    "P2583": "distance from Earth",
    # Discovery / classification
    "P575":  "time of discovery",
    "P61":   "discoverer",
    "P247":  "COSPAR ID",
    "P4743": "spectral class",
    # General
    "P18":  "image",
    "P856": "official website",
}

PROPERTY_CATALOGUE_TEXT = "\n".join(
    f"  {pid}: {desc}" for pid, desc in KNOWN_WD_PROPERTIES.items()
)

##############################################################################
# LOAD MODEL
##############################################################################

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="cpu",
    torch_dtype="auto"
)

##############################################################################
# LOAD RDF
##############################################################################

graph = Graph()

print("Loading UAT...")
graph.parse(UAT_FILE, format="ttl")

print("Loading mappings...")
graph.parse(MAPPING_FILE, format="ttl")

print("Triples:", len(graph))

##############################################################################
# LLM HELPER
##############################################################################

def llm_chat(prompt, max_tokens=256):

    messages = [
        {
            "role": "system",
            "content": (
                "You are a STRICT information extraction system. "
                "You MUST NOT use external knowledge. "
                "Only transform input to structured output."
            )
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=False
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

##############################################################################
# JSON PARSING (ROBUST)
##############################################################################

def extract_json(text):

    text = text.replace("```json", "").replace("```", "")

    split_by_assistant = text.split("assistant")

    if not split_by_assistant:
        raise ValueError("No assistant response found")

    last_assistant = split_by_assistant[-1]

    # Try to find the outermost JSON object (handles nested braces)
    depth = 0
    start = None
    for i, ch in enumerate(last_assistant):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                json_str = last_assistant[start:i+1]
                json_str = re.sub(r'[\n\r\t]+', ' ', json_str)
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = re.sub(r',\s*]', ']', json_str)
                return json.loads(json_str)

    raise ValueError("No valid JSON object found in assistant response")

##############################################################################
# STEP 1: RICH QUERY PLAN
#
# The LLM now produces a detailed execution plan that drives BOTH
#   – the UAT SKOS traversal depth / scope
#   – the exact Wikidata SPARQL queries (properties + limits + depth)
##############################################################################

QUERY_PLAN_SCHEMA = """
{
  "entities": ["<entity1>", ...],
  // ENTITY EXTRACTION RULES – follow strictly:
  // • Only include NAMED astronomical objects or well-defined concept classes.
  // • NEVER add generic nouns that describe the relation (e.g. "moon", "satellite",
  //   "planet", "star") when they are already implied by intent or a possessive
  //   phrase like "Jupiter's moons" or "moons of Saturn".
  //   ✗ BAD: entities: ["Jupiter", "moon"]
  //   ✓ GOOD: entities: ["Jupiter"]   (intent=satellites already covers moons)
  // • For comparisons include all named bodies: ["Mercury", "Venus"]
  // • For class-based list queries include the class name: ["dwarf planet"]
  "intent": "<one of the values below>",
  "uat_traversal": "<one of: definition | related | broader | narrower | full>",
  "wikidata_queries": [                  // list of query descriptors, one per need
    {
      "label": "<short human-readable label for this query>",
      "property": "<Wikidata P-ID or 'meta'>",
      // 'meta' = fetch label+description of the item itself (no traversal)
      "direction": "<outgoing | incoming>",
      // DIRECTION GUIDE:
      //   P397 (parent body): moon.P397 = planet  → use INCOMING to find moons
      //     incoming means: find all X where X.P397 = <our entity>
      //   P398 (child body):  planet.P398 = moon   → use OUTGOING to find moons
      //     outgoing means: find all X where <our entity>.P398 = X
      //   P31  (instance of): X.P31 = class        → INCOMING to list instances
      //   P279 (subclass of): X.P279 = class        → INCOMING to list subclasses
      //   Physical props (P2067 mass, P2386 diameter, P2147 period, …):
      //     these are OUTGOING (the entity has the property value)
      "limit": <integer 1-200>,
      "depth": <integer 1-3>
      // depth=1 → direct relations only
      // depth=2 → also fetch one hop further (e.g. moons of moons)
      // depth=3 → full transitive closure (use sparingly)
    }
  ]
}
"""

INTENT_VALUES = (
    "definition      – what is X? / explain X",
    "wikidata        – factual property lookup (size, mass, distance, …)",
    "related         – what is related to / connected to X?",
    "broader         – what is the parent/superclass of X?",
    "narrower        – what are sub-types / children of X?",
    "comparison      – compare two or more entities",
    "list            – enumerate all members / instances of a class",
    "satellites      – list moons / natural satellites of a body",
    "discovery       – when / by whom was X discovered?",
    "classification  – what type / class is X?",
)

PLAN_EXAMPLES = """
Question: What is a black hole?
{
  "entities": ["black hole"],
  "intent": "definition",
  "uat_traversal": "definition",
  "wikidata_queries": [
    {"label": "meta", "property": "meta", "direction": "outgoing", "limit": 1, "depth": 1}
  ]
}

Question: How many moons does Jupiter have?
{
  "entities": ["Jupiter"],
  "intent": "satellites",
  "uat_traversal": "definition",
  "wikidata_queries": [
    {"label": "moons of Jupiter", "property": "P397", "direction": "incoming", "limit": 100, "depth": 1}
  ]
}

Question: Name Jupiter's moons
{
  "entities": ["Jupiter"],
  "intent": "satellites",
  "uat_traversal": "definition",
  "wikidata_queries": [
    {"label": "moons of Jupiter (P397 incoming)", "property": "P397", "direction": "incoming", "limit": 100, "depth": 1},
    {"label": "moons of Jupiter (P398 outgoing)", "property": "P398", "direction": "outgoing", "limit": 100, "depth": 1}
  ]
}

Question: List the moons of Saturn
{
  "entities": ["Saturn"],
  "intent": "satellites",
  "uat_traversal": "definition",
  "wikidata_queries": [
    {"label": "moons of Saturn (P397 incoming)", "property": "P397", "direction": "incoming", "limit": 150, "depth": 1},
    {"label": "moons of Saturn (P398 outgoing)", "property": "P398", "direction": "outgoing", "limit": 150, "depth": 1}
  ]
}

Question: What are the sub-types of variable stars?
{
  "entities": ["variable star"],
  "intent": "narrower",
  "uat_traversal": "narrower",
  "wikidata_queries": [
    {"label": "subclasses", "property": "P279", "direction": "incoming", "limit": 50, "depth": 2},
    {"label": "instances",  "property": "P31",  "direction": "incoming", "limit": 50, "depth": 1}
  ]
}

Question: Size comparison between Mercury and Venus?
{
  "entities": ["Mercury", "Venus"],
  "intent": "comparison",
  "uat_traversal": "definition",
  "wikidata_queries": [
    {"label": "diameter", "property": "P2386", "direction": "outgoing", "limit": 1, "depth": 1},
    {"label": "mass",     "property": "P2067", "direction": "outgoing", "limit": 1, "depth": 1}
  ]
}

Question: What are all known dwarf planets in our solar system?
{
  "entities": ["dwarf planet"],
  "intent": "list",
  "uat_traversal": "definition",
  "wikidata_queries": [
    {"label": "instances of dwarf planet", "property": "P31",  "direction": "incoming", "limit": 100, "depth": 1},
    {"label": "subclasses",                "property": "P279", "direction": "incoming", "limit": 20,  "depth": 1}
  ]
}

Question: When was Neptune discovered and by whom?
{
  "entities": ["Neptune"],
  "intent": "discovery",
  "uat_traversal": "definition",
  "wikidata_queries": [
    {"label": "discoverer",       "property": "P61",  "direction": "outgoing", "limit": 5, "depth": 1},
    {"label": "time of discovery","property": "P575", "direction": "outgoing", "limit": 1, "depth": 1}
  ]
}

Question: What is the orbital period of Halley's Comet?
{
  "entities": ["Halley's Comet"],
  "intent": "wikidata",
  "uat_traversal": "definition",
  "wikidata_queries": [
    {"label": "orbital period", "property": "P2147", "direction": "outgoing", "limit": 1, "depth": 1}
  ]
}

Question: What concepts are related to pulsars?
{
  "entities": ["pulsar"],
  "intent": "related",
  "uat_traversal": "related",
  "wikidata_queries": [
    {"label": "meta",       "property": "meta", "direction": "outgoing", "limit": 1, "depth": 1},
    {"label": "instances",  "property": "P31",  "direction": "incoming", "limit": 30, "depth": 1},
    {"label": "subclasses", "property": "P279", "direction": "incoming", "limit": 20, "depth": 2}
  ]
}
"""

def question_to_plan(question):
    """
    Ask the LLM to produce a rich execution plan with per-query Wikidata specs.
    Falls back to a safe default plan on parse failure.
    """

    prompt = f"""
Return ONLY valid JSON – no prose, no markdown fences.

SCHEMA:
{QUERY_PLAN_SCHEMA}

INTENT VALUES:
{chr(10).join(INTENT_VALUES)}

AVAILABLE WIKIDATA PROPERTIES (use ONLY these P-IDs):
{PROPERTY_CATALOGUE_TEXT}

EXAMPLES:
{PLAN_EXAMPLES}

Now produce the plan for:
Question: {question}
"""

    text = llm_chat(prompt, max_tokens=400)

    print("\n===== RAW LLM OUTPUT (PLAN) =====")
    print(text)

    try:
        plan = extract_json(text)

        # --- normalise / validate ---
        if not isinstance(plan.get("entities"), list) or not plan["entities"]:
            plan["entities"] = [question]

        if plan.get("intent") not in [
            "definition", "wikidata", "related", "broader", "narrower",
            "comparison", "list", "satellites", "discovery", "classification"
        ]:
            plan["intent"] = "wikidata"

        if plan.get("uat_traversal") not in [
            "definition", "related", "broader", "narrower", "full"
        ]:
            plan["uat_traversal"] = "definition"

        # Sanitise wikidata_queries
        sanitised = []
        for q in plan.get("wikidata_queries", []):
            pid = q.get("property", "meta")
            if pid != "meta" and pid not in KNOWN_WD_PROPERTIES:
                print(f"[PLAN] Unknown property {pid} – dropping query")
                continue
            direction = q.get("direction", "incoming")
            if direction not in ("outgoing", "incoming"):
                direction = "incoming"
            limit = max(1, min(200, int(q.get("limit", 20))))
            depth  = max(1, min(3,   int(q.get("depth",  1))))
            sanitised.append({
                "label":     q.get("label", pid),
                "property":  pid,
                "direction": direction,
                "limit":     limit,
                "depth":     depth,
            })

        if not sanitised:
            sanitised = [{"label": "meta", "property": "meta",
                          "direction": "outgoing", "limit": 1, "depth": 1}]
        plan["wikidata_queries"] = sanitised

        print("\n===== PARSED PLAN =====")
        print(json.dumps(plan, indent=2))
        return plan

    except Exception as e:
        print("JSON ERROR:", e)
        return {
            "entities": [question],
            "intent": "wikidata",
            "uat_traversal": "definition",
            "wikidata_queries": [
                {"label": "meta", "property": "meta",
                 "direction": "outgoing", "limit": 1, "depth": 1}
            ],
        }

##############################################################################
# STEP 2: UAT SEARCH
##############################################################################

def find_uat_concept(entity):

    entity = entity.strip().lower()

    preferred_labels = [
        f"the {entity}",
        entity,
        f"{entity}s",
    ]

    results = []

    for search_term in preferred_labels:
        search_safe = search_term.replace('"', '\\"')

        query = f"""
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

SELECT DISTINCT ?concept ?label
WHERE {{
    ?concept skos:prefLabel|skos:altLabel ?label .
    FILTER(
        LCASE(STR(?label)) = LCASE("{search_safe}") ||
        LCASE(STR(?label)) = LCASE("The {search_safe}") ||
        LCASE(STR(?label)) = LCASE("{search_safe} (Solar system)") ||
        CONTAINS(LCASE(STR(?label)), LCASE("{search_safe}"))
    )
}}
LIMIT 30
"""

        for row in graph.query(query):
            r = {"concept": str(row.concept), "label": str(row.label)}
            if r not in results:
                results.append(r)

        exact_matches = [r for r in results if r["label"].lower() in preferred_labels]
        if exact_matches:
            break

    print("\n===== SPARQL UAT SEARCH =====")
    for r in results[:20]:
        print("HIT:", r)
    print("TOTAL:", len(results))
    return results

##############################################################################
# STEP 3: UAT CONTEXT (INTENT-AWARE)
##############################################################################

def get_uat_context(concept_uri, uat_traversal="definition"):

    dispatch = {
        "related":    get_uat_related_concepts,
        "broader":    get_uat_broader_concepts,
        "narrower":   get_uat_narrower_concepts,
        "full":       get_uat_full_context,
        "definition": get_uat_definition,
    }
    fn = dispatch.get(uat_traversal, get_uat_definition)
    return fn(concept_uri)

def get_uat_definition(concept_uri):
    query = f"""
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT ?pref ?alt ?definition
WHERE {{
    OPTIONAL {{ <{concept_uri}> skos:prefLabel ?pref }}
    OPTIONAL {{ <{concept_uri}> skos:altLabel ?alt }}
    OPTIONAL {{ <{concept_uri}> skos:definition ?definition }}
}}
"""
    data = {"pref": [], "alt": [], "definition": []}
    for row in graph.query(query):
        if row.pref:      data["pref"].append(str(row.pref))
        if row.alt:       data["alt"].append(str(row.alt))
        if row.definition:data["definition"].append(str(row.definition))
    print("\n===== UAT CONTEXT (DEFINITION) =====")
    print(json.dumps(data, indent=2))
    return data

def get_uat_related_concepts(concept_uri):
    query = f"""
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT ?related ?relLabel ?narrower ?narrowLabel ?broader ?broadLabel
WHERE {{
    OPTIONAL {{ <{concept_uri}> skos:related  ?related  . ?related  skos:prefLabel ?relLabel   }}
    OPTIONAL {{ <{concept_uri}> skos:narrower ?narrower . ?narrower skos:prefLabel ?narrowLabel }}
    OPTIONAL {{ <{concept_uri}> skos:broader  ?broader  . ?broader  skos:prefLabel ?broadLabel  }}
}}
"""
    data = {"related": [], "narrower": [], "broader": []}
    for row in graph.query(query):
        if row.related  and row.relLabel:   data["related"].append(  {"uri": str(row.related),  "label": str(row.relLabel)})
        if row.narrower and row.narrowLabel:data["narrower"].append({"uri": str(row.narrower),"label": str(row.narrowLabel)})
        if row.broader  and row.broadLabel: data["broader"].append(  {"uri": str(row.broader),  "label": str(row.broadLabel)})
    print("\n===== UAT CONTEXT (RELATED) =====")
    print(json.dumps(data, indent=2))
    return data

def get_uat_broader_concepts(concept_uri):
    query = f"""
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT ?broader ?broadLabel
WHERE {{
    <{concept_uri}> skos:broader* ?broader .
    ?broader skos:prefLabel ?broadLabel
}}
"""
    data = {"broader": []}
    for row in graph.query(query):
        data["broader"].append({"uri": str(row.broader), "label": str(row.broadLabel)})
    print("\n===== UAT CONTEXT (BROADER) =====")
    print(json.dumps(data, indent=2))
    return data

def get_uat_narrower_concepts(concept_uri):
    query = f"""
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT ?narrower ?narrowLabel
WHERE {{
    <{concept_uri}> skos:narrower+ ?narrower .
    ?narrower skos:prefLabel ?narrowLabel
}}
"""
    data = {"narrower": []}
    for row in graph.query(query):
        data["narrower"].append({"uri": str(row.narrower), "label": str(row.narrowLabel)})
    print("\n===== UAT CONTEXT (NARROWER) =====")
    print(json.dumps(data, indent=2))
    return data

def get_uat_full_context(concept_uri):
    """Merge definition + related + broader + narrower."""
    d   = get_uat_definition(concept_uri)
    rel = get_uat_related_concepts(concept_uri)
    d.update(rel)
    return d

##############################################################################
# STEP 4: WIKIDATA LINK
##############################################################################

def get_wikidata_uri(concept_uri):
    query = f"""
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT ?wd WHERE {{ <{concept_uri}> skos:closeMatch ?wd }}
"""
    for row in graph.query(query):
        print("WIKIDATA LINK:", row.wd)
        return str(row.wd)
    print("NO WIKIDATA LINK")
    return None

##############################################################################
# STEP 5: GENERIC WIKIDATA EXECUTOR
#
# Executes the query descriptors produced by the LLM plan.
##############################################################################

WD_HEADERS = {
    "Accept": "application/sparql-results+json",
    "User-Agent": "GraphRAG-UAT-Project/1.0 (student research)",
}

def _wd_sparql(query, timeout=30):
    """Run a SPARQL query against Wikidata and return bindings list."""
    try:
        r = requests.get(
            WIKIDATA_ENDPOINT,
            params={"query": query},
            headers=WD_HEADERS,
            timeout=timeout,
        )
        print("STATUS:", r.status_code)
        if r.status_code != 200:
            print("ERROR:", r.text[:300])
            return []
        return r.json().get("results", {}).get("bindings", [])
    except Exception as e:
        print("WIKIDATA ERROR:", e)
        return []

def execute_wikidata_query(wikidata_uri, query_desc):
    """
    Execute a single query descriptor against the given Wikidata entity URI.

    query_desc keys:
      label, property, direction, limit, depth
    """
    pid       = query_desc["property"]
    direction = query_desc["direction"]
    limit     = query_desc["limit"]
    depth     = query_desc["depth"]
    label     = query_desc["label"]

    print(f"\n===== WIKIDATA QUERY: {label} (P={pid}, dir={direction}, limit={limit}, depth={depth}) =====")

    # --- META: just fetch label + description ---
    if pid == "meta":
        sparql = f"""
SELECT ?label ?description WHERE {{
    BIND(<{wikidata_uri}> AS ?item)
    OPTIONAL {{ ?item rdfs:label ?label . FILTER(LANG(?label)="en") }}
    OPTIONAL {{ ?item schema:description ?description . FILTER(LANG(?description)="en") }}
}}
LIMIT 1
"""
        return _wd_sparql(sparql)

    # --- Build property path based on depth ---
    if depth == 1:
        prop_path = f"wdt:{pid}"
    elif depth == 2:
        prop_path = f"wdt:{pid}|wdt:{pid}/wdt:{pid}"
    else:
        # depth 3 → full transitive closure
        prop_path = f"wdt:{pid}+"

    # --- OUTGOING: <entity> → value ---
    if direction == "outgoing":
        sparql = f"""
SELECT ?value ?valueLabel ?description WHERE {{
    BIND(<{wikidata_uri}> AS ?item)
    ?item {prop_path} ?value .
    OPTIONAL {{ ?value rdfs:label ?valueLabel . FILTER(LANG(?valueLabel)="en") }}
    OPTIONAL {{ ?value schema:description ?description . FILTER(LANG(?description)="en") }}
}}
LIMIT {limit}
"""

    # --- INCOMING: value → <entity> ---
    else:
        sparql = f"""
SELECT ?item ?itemLabel ?description WHERE {{
    ?item {prop_path} <{wikidata_uri}> .
    OPTIONAL {{ ?item rdfs:label ?itemLabel . FILTER(LANG(?itemLabel)="en") }}
    OPTIONAL {{ ?item schema:description ?description . FILTER(LANG(?description)="en") }}
}}
LIMIT {limit}
"""

    bindings = _wd_sparql(sparql)
    print(f"  → {len(bindings)} result(s)")
    return bindings


def run_wikidata_queries(wikidata_uri, wikidata_queries):
    """
    Execute all query descriptors from the plan and collect results per label.
    Returns dict: { label → list[binding] }

    Special handling for satellite intents:
    - Merges results from P397-incoming and P398-outgoing queries (both label
      the same moon differently in Wikidata) and deduplicates by URI.
    - Filters out non-moon children (rings, orbits, missions, space stations).
    """
    # Non-moon noise keywords found in Wikidata descriptions
    NON_MOON_DESC_KEYWORDS = {
        "ring", "orbit", "mission", "spacecraft", "space station",
        "gateway", "synchronous", "probe", "lander", "orbiter",
    }

    raw = {}
    for qd in wikidata_queries:
        bindings = execute_wikidata_query(wikidata_uri, qd)
        raw[qd["label"]] = bindings

    # Detect satellite-related query labels
    satellite_labels = [
        lbl for lbl in raw
        if "moon" in lbl.lower() or "satellite" in lbl.lower()
    ]

    if len(satellite_labels) >= 2:
        # Merge both satellite result sets, deduplicate by item URI
        seen_uris = set()
        merged = []
        for lbl in satellite_labels:
            for b in raw[lbl]:
                # Normalise: P397-incoming has ?item; P398-outgoing has ?value
                uri_val = (
                    b.get("item", b.get("value", {})).get("value", "")
                )
                label_val = (
                    b.get("itemLabel", b.get("valueLabel", {})).get("value", "")
                )
                desc_val = b.get("description", {}).get("value", "").lower()

                if not uri_val or uri_val in seen_uris:
                    continue

                # Filter out non-moon noise
                if any(kw in desc_val for kw in NON_MOON_DESC_KEYWORDS):
                    continue

                seen_uris.add(uri_val)
                # Normalise to a consistent binding shape for the answer step
                merged.append({
                    "item":       {"value": uri_val},
                    "itemLabel":  {"value": label_val},
                    "description":{"value": b.get("description", {}).get("value", "")},
                })

        merged.sort(key=lambda b: b["itemLabel"]["value"])
        # Replace the individual satellite entries with one clean merged result
        for lbl in satellite_labels:
            del raw[lbl]
        raw["moons (merged & filtered)"] = merged
        print(f"[SATELLITE MERGE] {len(merged)} unique moons after dedup/filter")

    elif len(satellite_labels) == 1:
        # Single satellite query – just filter noise
        lbl = satellite_labels[0]
        filtered = []
        for b in raw[lbl]:
            desc_val = b.get("description", {}).get("value", "").lower()
            if not any(kw in desc_val for kw in NON_MOON_DESC_KEYWORDS):
                filtered.append(b)
        raw[lbl] = filtered
        print(f"[SATELLITE FILTER] {len(filtered)} moons after noise filter")

    return raw

##############################################################################
# STEP 6: ANSWER GENERATION
##############################################################################

def _has_useful_data(uat_context):
    return (
        (uat_context.get("definition") and any(uat_context.get("definition"))) or
        uat_context.get("related") or
        uat_context.get("narrower") or
        uat_context.get("broader") or
        uat_context.get("pref")
    )

def _format_wd_results(wd_results):
    """
    Convert the structured results dict into a clean, readable text block.

    Priority for human-readable name:
      itemLabel > valueLabel > item URI fragment > value URI fragment
    Physical / literal values (mass, diameter, …) are rendered as-is.
    """
    lines = []
    for section_label, bindings in wd_results.items():
        if not bindings:
            continue
        lines.append(f"[{section_label}]")
        for b in bindings:
            # --- Determine the primary human-readable label ---
            name = (
                b.get("itemLabel",  {}).get("value")
                or b.get("valueLabel", {}).get("value")
                or b.get("label",      {}).get("value")
            )
            # Fallback: extract the last path segment of a URI
            if not name:
                raw_uri = (
                    b.get("item",  {}).get("value")
                    or b.get("value", {}).get("value", "")
                )
                name = raw_uri.rstrip("/").rsplit("/", 1)[-1] if raw_uri else "?"

            # --- Optional description ---
            desc = b.get("description", {}).get("value", "")

            # --- Optional literal value (for physical properties) ---
            lit = b.get("value", {}).get("value", "")
            # Only include literal if it's NOT a Wikidata URI (i.e. it's a real value)
            if lit.startswith("http://www.wikidata.org"):
                lit = ""

            row = f"  • {name}"
            if lit:
                row += f": {lit}"
            if desc:
                row += f"  ({desc})"
            lines.append(row)

    return "\n".join(lines) if lines else "(no wikidata results)"


def create_answer(question, all_entity_data, intent="wikidata"):
    """
    Unified answer generator. all_entity_data is a dict:
      { entity_name: { "uat": {...}, "wikidata": {label: [bindings]} } }
    intent is passed through from the plan so the prompt can be tailored.
    """

    # Hard fail-closed
    if not all_entity_data:
        return "No answer found in knowledge base."

    has_any = any(_has_useful_data(v["uat"]) for v in all_entity_data.values())
    if not has_any:
        return "No answer found in knowledge base."

    # Build context block
    ctx_lines = []
    all_uat_text = ""

    for entity, data in all_entity_data.items():
        uat_ctx = data.get("uat", {})
        wd_res  = data.get("wikidata", {})

        ctx_lines.append(f"\n=== {entity} ===")
        ctx_lines.append("UAT:\n" + json.dumps(uat_ctx, indent=2))
        ctx_lines.append("WIKIDATA:\n" + _format_wd_results(wd_res))

        if uat_ctx.get("definition"):
            all_uat_text += " ".join(uat_ctx["definition"]) + " "

    context_text = "\n".join(ctx_lines)

    # --- Tailor the output instruction based on intent ---
    LIST_INTENTS = {"list", "satellites", "narrower", "broader", "related"}

    if intent in LIST_INTENTS:
        output_instruction = (
            "LIST every item that appears under the WIKIDATA bullet points above. "
            "Write each name on its own line, prefixed with a dash (-). "
            "Do NOT say 'see above', 'as listed', or refer to the data section. "
            "Do NOT omit any names. "
            "After the list, add one short sentence summarising the count."
        )
        max_tok = 600
    elif intent == "comparison":
        output_instruction = (
            "Compare the entities directly using ONLY the numbers/properties in the data. "
            "State values explicitly (e.g. 'X has diameter Y, while Z has diameter W'). "
            "Max 4 sentences."
        )
        max_tok = 200
    elif intent == "discovery":
        output_instruction = (
            "State the discoverer(s) and date of discovery as given in the data. "
            "Max 2 sentences."
        )
        max_tok = 100
    else:
        output_instruction = (
            "Answer in max 5 sentences using ONLY the data above. "
            "State facts directly – never refer to 'the data section' or 'the WIKIDATA section'."
        )
        max_tok = 250

    prompt = f"""
You are a STRICT grounded QA system.

ABSOLUTE RULES:
- Use ONLY the DATA PROVIDED below. Never add external knowledge.
- NEVER say "see above", "as listed in the data", "refer to WIKIDATA", or any phrase
  that points the user elsewhere. Always write out the information directly.
- Do not invent measurements, names, or properties absent from the data.
- If the data is insufficient, respond exactly: "No answer found in knowledge base."

QUESTION: {question}

DATA PROVIDED:
{context_text}

TASK: {output_instruction}
"""

    answer = llm_chat(prompt, max_tokens=max_tok)

    # --- Hallucination guard for size/distance claims ---
    if any(kw in question.lower() for kw in
           ["how big", "how far", "diameter", "size", "compar", "distance"]):

        size_in_data = any(
            ph in all_uat_text.lower()
            for ph in ["diameter", "radius", "distance", "km", "au", "light-year", "wide"]
        )
        size_in_answer = any(
            kw in answer.lower()
            for kw in ["diameter", "km", "kilometer", "across", "wide",
                       "light-year", "lightyear", "au", "mile"]
        )

        print(f"[HALLUCINATION CHECK] data_has_size={size_in_data} answer_has_size={size_in_answer}")
        if size_in_answer and not size_in_data:
            print("  DETECTED HALLUCINATION – returning 'No answer found'")
            return "No answer found in knowledge base."

    return answer

##############################################################################
# PIPELINE
##############################################################################

def ask(question):

    print("\n==============================")
    print("QUESTION:", question)
    print("==============================")

    # 1. Build rich execution plan
    plan            = question_to_plan(question)
    entities        = plan["entities"]
    intent          = plan["intent"]
    uat_traversal   = plan["uat_traversal"]
    wikidata_queries = plan["wikidata_queries"]

    print(f"\nENTITIES: {entities}")
    print(f"INTENT:   {intent}")
    print(f"UAT:      {uat_traversal}")
    print(f"WD QUERIES:")
    for qd in wikidata_queries:
        print(f"  {qd['label']}: P={qd['property']}, dir={qd['direction']}, "
              f"limit={qd['limit']}, depth={qd['depth']}")

    all_entity_data = {}

    for entity in entities:
        print(f"\n{'='*40}")
        print(f"  Processing: {entity}")
        print(f"{'='*40}")

        # 2. UAT lookup
        matches = find_uat_concept(entity)
        if not matches:
            print(f"NO UAT MATCH for '{entity}' – skipping")
            continue

        def score_match(m, ent):
            lbl = m["label"].lower()
            e   = ent.lower()
            if lbl == e:                return 100
            if lbl.startswith(e):       return 90
            if lbl.endswith(e):         return 80
            if e in lbl:                return 50
            return 1

        scored = sorted(
            ((m, score_match(m, entity)) for m in matches),
            key=lambda x: x[1],
            reverse=True
        )

        # Pick best UAT match that has data
        best_uri, best_context = None, None
        for m, score in scored[:10]:
            uri = m["concept"]
            ctx = get_uat_context(uri, uat_traversal)
            if _has_useful_data(ctx):
                best_uri, best_context = uri, ctx
                print(f"[UAT MATCH] {m['label']} (score={score})")
                break

        if not best_context:
            best_uri     = scored[0][0]["concept"]
            best_context = get_uat_context(best_uri, uat_traversal)
            print(f"[UAT MATCH] {scored[0][0]['label']} (no data)")

        # 3. Wikidata
        wikidata_uri = get_wikidata_uri(best_uri)
        wd_results   = {}

        if wikidata_uri:
            wd_results = run_wikidata_queries(wikidata_uri, wikidata_queries)
        else:
            print("No Wikidata URI – skipping WD queries")

        all_entity_data[entity] = {
            "uat":      best_context,
            "wikidata": wd_results,
        }

    if not all_entity_data:
        print("\nNO ENTITIES RESOLVED – FAIL CLOSED")
        return "No answer found in knowledge base."

    # 4. Generate answer
    answer = create_answer(question, all_entity_data, intent=intent)

    print("\n===== FINAL ANSWER =====")
    try:
        print(answer)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(answer.encode("utf-8") + b"\n")

    return answer

##############################################################################
# CLI
##############################################################################

if __name__ == "__main__":
    while True:
        q = input("\nUser: ")
        if q.lower() in ["exit", "quit"]:
            break
        ask(q)