import yaml
import re
from typing import Optional

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
    session.mount("http://", adapter)
    session.mount("https://", adapter)

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

print("Load config data", flush=True)
with open(meta["config"], "r") as file:
    config = yaml.safe_load(file)

print("Check general fields", flush=True)
assert len(config["name"]) <= NAME_MAXLEN, f"Component id (.name) should not exceed {NAME_MAXLEN} characters."
assert "namespace" in config is not None, "namespace not a field or is empty"

print("Check info fields", flush=True)
info = config["info"]
assert "type" in info, "type not an info field"
info_types = ["method", "control_method"]
assert info["type"] in info_types , f"got {info['type']} expected one of {info_types}"
assert "label" in info is not None, "label not an info field or is empty"
assert "summary" in info is not None, "summary not an info field or is empty"
assert "FILL IN:" not in info["summary"], "Summary not filled in"
assert len(info["summary"]) <= SUMMARY_MAXLEN, f"Component id (.info.summary) should not exceed {SUMMARY_MAXLEN} characters."
assert "description" in info is not None, "description not an info field or is empty"
assert "FILL IN:" not in info["description"], "description not filled in"
assert len(info["description"]) <= DESCRIPTION_MAXLEN, f"Component id (.info.description) should not exceed {DESCRIPTION_MAXLEN} characters."
if info["type"] == "method":
    assert "reference" in info, "reference not an info field"
    bib = _load_bib()
    if info["reference"]:
        reference = info["reference"]
        if not isinstance(reference, list):
            reference = [reference]
        for ref in reference:
            out = check_reference(ref)
            assert out is None, out
    assert "documentation_url" in info is not None, "documentation_url not an info field or is empty"
    assert "repository_url" in info is not None, "repository_url not an info field or is empty"
    assert check_url(info["documentation_url"]), f"{info['documentation_url']} is not reachable"
    assert check_url(info["repository_url"]), f"{info['repository_url']} is not reachable"

for arg_grp in config["argument_groups"]:
    if arg_grp.get("name") == "Arguments":
        args = arg_grp["arguments"]

if "variants" in info:
    arg_names = [arg["name"].replace("--", "") for arg in args] + ["preferred_normalization"]

    for paramset_id, paramset in info["variants"].items():
        if paramset:
            for arg_id in paramset:
                assert arg_id in arg_names, f"Argument '{arg_id}' in `.info.variants['{paramset_id}']` is not an argument in `.arguments`."

if "preferred_normalization" in info:
    norm_methods = ["log_cpm", "log_cp10k", "counts", "log_scran_pooling", "sqrt_cpm", "sqrt_cp10k", "l1_sqrt"]
    assert info["preferred_normalization"] in norm_methods, "info['preferred_normalization'] not one of '" + "', '".join(norm_methods) + "'."

print("Check runners fields", flush=True)
runners = config["runners"]
for runner in runners:
    if not runner["type"] == "nextflow":
        continue
    nextflow = runner



assert nextflow, "nextflow not a runner"
assert nextflow["directives"], "directives not a field in nextflow runner"
assert nextflow["directives"]["label"], "label not a field in nextflow runner directives"

assert [i for i in nextflow["directives"]["label"] if i in TIME_LABELS], "time label not filled in"
assert [i for i in nextflow["directives"]["label"] if i in MEM_LABELS], "mem label not filled in"
assert [i for i in nextflow["directives"]["label"] if i in CPU_LABELS], "cpu label not filled in"

print("All checks succeeded!", flush=True)
