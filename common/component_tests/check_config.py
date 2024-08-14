import yaml
import re
from typing import Dict, List, Union

## VIASH START
meta = {
    "config" : "foo"
}
## VIASH END

NAME_MAXLEN = 50
LABEL_MAXLEN = 50
SUMMARY_MAXLEN = 400
DESCRIPTION_MAXLEN = 5000

TIME_LABELS = ["lowtime", "midtime", "hightime", "veryhightime"]
MEM_LABELS = ["lowmem", "midmem", "highmem", "veryhighmem"]
CPU_LABELS = ["lowcpu", "midcpu", "highcpu", "veryhighcpu"]

def check_url(url: str) -> bool:
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

def load_config() -> Dict:
    with open(meta["config"], "r") as file:
        config = yaml.safe_load(file)

    def process_argument(argument: dict) -> dict:
        argument["clean_name"] = argument["name"].lstrip("-")
        return argument

    config["all_arguments"] = [
        process_argument(arg)
        for arg_grp in config["argument_groups"]
        for arg in arg_grp["arguments"]
    ]

    return config

def check_references(references: Dict[str, Union[str, List[str]]]) -> None:
    doi = references.get("doi")
    bibtex = references.get("bibtex")

    assert doi or bibtex, "One of .references.doi or .references.bibtex should be defined"

    if doi:
        if not isinstance(doi, list):
            doi = [doi]
        for d in doi:
            assert re.match(r"^10.\d{4,9}/[-._;()/:A-Za-z0-9]+$", d), f"Invalid DOI format: {doi}"
            assert check_url(f"https://doi.org/{d}"), f"DOI '{d}' is not reachable"

    if bibtex:
        if not isinstance(bibtex, list):
            bibtex = [bibtex]
        for b in bibtex:
            assert re.match(r"^@.*{.*", b), f"Invalid bibtex format: {b}"

def check_info(this_info: Dict) -> None:
    metadata_field_lengths = {
        "label": LABEL_MAXLEN,
        "summary": SUMMARY_MAXLEN,
        "description": DESCRIPTION_MAXLEN
    }
    
    if comp_type == "metric":
        metadata_field_lengths["name"] = NAME_MAXLEN
    
    for field, max_length in metadata_field_lengths.items():
        value = this_info.get(field)
        if comp_type != "metric":
            value = config.get(field) or value
        assert value, f".info.{field} is not defined"
        assert "FILL IN:" not in value, f".info.{field} not filled in"
        assert len(value) <= max_length, f".info.{field} should not exceed {max_length} characters"

    documentation_url = this_info.get("documentation_url")
    if comp_type == "method" or documentation_url:
        assert documentation_url, ".info.documentation_url is not defined"
        assert check_url(documentation_url), f".info.documentation_url '{documentation_url}' is not reachable"

    repository_url = this_info.get("repository_url")
    if comp_type == "method" or repository_url:
        assert repository_url, ".info.repository_url is not defined"
        assert check_url(repository_url), f".info.repository_url '{repository_url}' is not reachable"

    references = this_info.get("references", {})
    if comp_type != "metric":
        references = config.get("references", {}) or references
    if comp_type != "control_method" or references:
        print("Check references fields", flush=True)
        check_references(references)

print("Load config data", flush=True)
config = load_config()

print("Check config name and namespace", flush=True)
assert len(config["name"]) <= NAME_MAXLEN, f".name should not exceed {NAME_MAXLEN} characters"
assert config.get("namespace"), ".namespace is not defined"

print("Check .info.type field", flush=True)
info = config.get("info", {})
comp_type = info.get("type")

expected_types = ["method", "control_method", "metric"]
assert comp_type in expected_types, ".info.type should be equal to 'method' or 'control_method'"

print("Check info metadata", flush=True)
if comp_type == "metric":
    metric_infos = info.get("metrics", [])
    assert metric_infos, ".info.metrics is not defined"

    for metric_info in metric_infos:
        check_info(metric_info)
else:
    check_info(info)

print("Processing arguments", flush=True)
if "variants" in info:
    arg_names = [arg["name"] for arg in config["all_arguments"]] + ["preferred_normalization"]

    for paramset_id, paramset in info["variants"].items():
        if paramset:
            for arg_id in paramset:
                assert arg_id in arg_names, f"Argument '{arg_id}' in `.info.variants['{paramset_id}']` is not an argument in `.arguments`."

if "preferred_normalization" in info:
    norm_methods = ["log_cpm", "log_cp10k", "counts", "log_scran_pooling", "sqrt_cpm", "sqrt_cp10k", "l1_sqrt"]
    assert info["preferred_normalization"] in norm_methods, ".info['preferred_normalization'] not one of '" + "', '".join(norm_methods) + "'."

print("Check runners fields", flush=True)
runners = config.get("runners", [])
for runner in runners:
    if runner["type"] == "nextflow":
        nextflow_runner = runner

assert nextflow_runner, ".runners does not contain a nextflow runner"
assert nextflow_runner.get("directives"), "directives not a field in nextflow runner"
nextflow_labels = nextflow_runner["directives"].get("label")
assert nextflow_labels, "label not a field in nextflow runner directives"

assert [label for label in nextflow_labels if label in TIME_LABELS], "time label not filled in"
assert [label for label in nextflow_labels if label in MEM_LABELS], "mem label not filled in"
assert [label for label in nextflow_labels if label in CPU_LABELS], "cpu label not filled in"

print("All checks succeeded!", flush=True)
