# %%
import sys
import glob
import json
sys.path.append("./scripts/")

import xml.etree.ElementTree as ET
from preprocess_content import preprocess

# %%
def xml_to_dict(element):
    node = {key: value for key, value in element.attrib.items()}
    if element.text and element.text.strip():
        node['text'] = element.text.strip()
    
    for child in element:
        child_dict = xml_to_dict(child)
        if child.tag not in node:
            node[child.tag] = child_dict
        else:
            if not isinstance(node[child.tag], list):
                node[child.tag] = [node[child.tag]]
            node[child.tag].append(child_dict)
    return node

# %%
files = glob.glob("data/Terveyskirjasto - Käyvän hoidon potilasversiot/*.xml", recursive=True)
texts = []

for file_path in files:
    with open(file_path, "rb") as file:
        xml_content = file.read()
    texts.append(xml_content)
    break

print(texts[0])

# %%
obj = preprocess(
    raw_content=texts, 
    model_name="TurkuNLP/sbert-cased-finnish-paraphrase", 
    embed_content=True
)

obj 

# %%
obj["data"][0]["page_content"]
