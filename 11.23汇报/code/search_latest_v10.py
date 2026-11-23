import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET


QUERIES = {
    "map_prediction": 'all:"map prediction" AND all:navigation',
    "frontier_exploration": "all:frontier AND all:exploration",
    "zero_shot_vln": 'all:"zero-shot" AND (all:"vision-language navigation" OR all:VLN)',
    "rl_exploration": (
        'all:"reinforcement learning" AND all:exploration AND all:navigation'
    ),
    "uncertainty_planning": 'all:uncertainty AND all:"path planning"',
    "map_completion": 'all:"map completion" OR all:"map inpainting"',
}
NS = {"a": "http://www.w3.org/2005/Atom"}

for name, query in QUERIES.items():
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode({
        "search_query": query,
        "start": 0,
        "max_results": 12,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    })
    try:
        root = ET.fromstring(urllib.request.urlopen(url, timeout=60).read())
    except Exception as exc:
        print(name, exc)
        continue
    print(f"\n### {name}")
    for entry in root.findall("a:entry", NS)[:8]:
        title = " ".join(entry.find("a:title", NS).text.split())
        published = entry.find("a:published", NS).text[:10]
        link = entry.find("a:id", NS).text
        summary = " ".join(entry.find("a:summary", NS).text.split())
        print(f"{published} | {title} | {link}")
        print(summary[:420])
