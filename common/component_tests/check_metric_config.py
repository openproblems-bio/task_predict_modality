import yaml
import re
from typing import Dict, Optional

## VIASH START
meta = {
    "config" : "foo"
}
## VIASH END

NAME_MAXLEN = 50

SUMMARY_MAXLEN = 400

DESCRIPTION_MAXLEN = 5000

_MISSING_DOIS = ["vandermaaten2008visualizing", "hosmer2013applied"]

TIME_LABELS = ["lowtime", "midtime", "hightime"]
MEM_LABELS = ["lowmem", "midmem", "highmem"]
CPU_LABELS = ["lowcpu", "midcpu", "highcpu"]


def _load_bib():
    import os
    bib_path = meta["resources_dir"]+"/library.bib"
    if not os.path.exists(bib_path):
        return None
    with open(bib_path, "r") as file:
        return file.read()
    
def check_url(url):
    import requests
    from urllib3.util.retry import Retry
    from requests.adapters import HTTPAdapter

    # configure retry strategy
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    get = session.head(url)

    if get.ok or get.status_code == 429: # 429 rejected, too many requests
        return True
    else:
        return False


def check_reference(reference) -> Optional[str]:
    # reference is a doi
    if re.match(r"^10.\d{4,9}/[-._;()/:A-Za-z0-9]+$", reference):
        print(f"Checking DOI: {reference}", flush=True)
        url = f"https://doi.org/{reference}"
        if check_url(url):
            return None
        else:
            return f"DOI {reference} is not reachable"
    
    bib = _load_bib()
    if not bib:
        return f"Could not load bibtex file"
    
    entry_pattern =  r"(@\w+{[^}]*" + reference + r"[^}]*}(.|\n)*?)(?=@)"

    bib_entry = re.search(entry_pattern, bib)

    if not bib_entry:
        return f"reference {reference} not found in bibtex file"

    type_pattern = r"@(.*){" + reference
    doi_pattern = r"(?=[Dd][Oo][Ii]\s*=\s*{([^,}]+)})"

    entry_type = re.search(type_pattern, bib_entry.group(1))

    if not (entry_type.group(1) == "misc" or reference in _MISSING_DOIS):
        entry_doi = re.search(doi_pattern, bib_entry.group(1))
        if not entry_doi:
            return "No DOI found in reference"
        url = f"https://doi.org/{entry_doi.group(1)}"
        if not check_url(url):
            return f"doi {entry_doi.group(1)} is not reachable"

    return None

def check_metric(metric: Dict[str, str])  -> str:
    assert "name" in metric is not None, "name not a field or is empty"
    assert len(metric["name"]) <= NAME_MAXLEN, f"Component id (.info.metrics.metric.name) should not exceed {NAME_MAXLEN} characters."
    assert "label" in metric is not None, "label not a field in metric or is empty"
    assert "summary" in metric is not None, "summary not a field in metric or is empty"
    assert "FILL IN:" not in metric["summary"], "Summary not filled in"
    assert len(metric["summary"]) <= SUMMARY_MAXLEN, f"Component id (.info.metrics.metric.summary) should not exceed {SUMMARY_MAXLEN} characters."
    assert "description" in metric is not None, "description not a field in metric or is empty"
    assert len(metric["description"]) <= DESCRIPTION_MAXLEN, f"Component id (.info.metrics.metric.description) should not exceed {DESCRIPTION_MAXLEN} characters."
    assert "FILL IN:" not in metric["description"], "description not filled in"
    # assert "reference" in metric, "reference not a field in metric"
    if "reference" in metric:
        reference = metric["reference"]
        if not isinstance(reference, list):
            reference = [reference]
        for ref in reference:
            out = check_reference(ref)
            assert out is None, out
    # assert "documentation_url" in metric , "documentation_url not a field in metric"
    # assert "repository_url" in metric , "repository_url not a metric field"
    if "documentation_url" in metric:
        assert check_url(metric["documentation_url"]), f"{metric['documentation_url']} is not reachable"
    if "repository_url" in metric:
        assert check_url(metric["repository_url"]), f"{metric['repository_url']} is not reachable"
    assert "min" in metric is not None, f"min not a field in metric or is emtpy"
    assert "max" in metric is not None, f"max not a field in metric or is empty"
    assert "maximize" in metric is not None, f"maximize not a field in metric or is emtpy"
    assert isinstance(metric['min'], (int, str)), "not an int or string (-.inf)"
    assert isinstance(metric['max'], (int, str)), "not an int or string (+.inf)"
    assert isinstance(metric['maximize'], bool) or metric["maximize"] not in ["-inf", "+inf"], "not a bool"


print("Load config data", flush=True)
with open(meta["config"], "r") as file:
                config = yaml.safe_load(file)

print("check general fields", flush=True)
assert "name" in config is not None, "Name not a field or is empty"
assert len(config["name"]) <= NAME_MAXLEN, f"Component id (.name) should not exceed {NAME_MAXLEN} characters."
assert "namespace" in config is not None, "namespace not a field or is empty"


print("Check info fields", flush=True)
info = config["info"]
assert "type" in info, "type not an info field"
assert info["type"] == "metric" , f"got {info['type']} expected 'metric'"
assert "metrics" in info, "metrics not an info field"
for metric in info["metrics"]:
    check_metric(metric)

print("Check runner fields", flush=True)
runners = config["runners"]
for runner in runners:
    if not runner["type"] == "nextflow":
        continue
    nextflow= runner

assert nextflow, "nextflow not a runner"
assert nextflow["directives"], "directives not a field in nextflow runner"
assert nextflow["directives"]["label"], "label not a field in nextflow runner directives"

assert [i for i in nextflow["directives"]["label"] if i in TIME_LABELS], "time label not filled in"
assert [i for i in nextflow["directives"]["label"] if i in MEM_LABELS], "mem label not filled in"
assert [i for i in nextflow["directives"]["label"] if i in CPU_LABELS], "cpu label not filled in"

print("All checks succeeded!", flush=True)
