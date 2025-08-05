"""Module for preprocessing raw XML content for text embedding and optionally embedding the contents
with an embedding model of choice."""

import re
from functools import lru_cache
from typing import TypedDict, Optional
from typing_extensions import Unpack, NotRequired
from markdownify import markdownify
import tqdm
from bs4 import BeautifulSoup, NavigableString, ProcessingInstruction, Comment, Tag
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from transformers import AutoTokenizer, GPT2TokenizerFast
from dotenv import load_dotenv
from embedding import embed

load_dotenv()

embedding_dimension_map = {
    'embed-multilingual-v3.0': 1024,
    'text-embedding-ada-002': 1536,
    'text-embedding-3-large': 3072,
    'BAAI/bge-m3': 1024,
    'intfloat/multilingual-e5-large': 1024,
    'intfloat/multilingual-e5-large-instruct': 1024,
    'intfloat/multilingual-e5-base': 768,
    'intfloat/multilingual-e5-small': 384,
    'TurkuNLP/sbert-cased-finnish-paraphrase': 768
}

pooling_strategy = {
    'embed-multilingual-v3.0': None,
    'text-embedding-ada-002': None,
    'text-embedding-3-large': None,
    'BAAI/bge-m3': 'cls',
    'intfloat/multilingual-e5-large': 'mean',
    'intfloat/multilingual-e5-large-instruct': 'mean',
    'intfloat/multilingual-e5-base': 'mean',
    'intfloat/multilingual-e5-small': 'mean',
    'TurkuNLP/sbert-cased-finnish-paraphrase': 'mean',
}


def get_text_safely(element: Tag | NavigableString | None, tag_name: str):
    tag = element.find(tag_name) if element else None
    if isinstance(tag, Tag):
        return tag.text
    elif isinstance(tag, NavigableString):
        return str(tag)
    elif isinstance(tag, int):
        return str(tag)
    else:
        return None


@lru_cache(maxsize=None)
def get_tokenizer(model_name: str, model_path: Optional[str] = None):
    if model_name == 'embed-multilingual-v3.0':
        return AutoTokenizer.from_pretrained("Cohere/Cohere-embed-multilingual-v3.0")
    elif model_name in ['text-embedding-ada-002', 'text-embedding-3-large']:
        return GPT2TokenizerFast.from_pretrained("Xenova/text-embedding-ada-002")
    elif model_path:
        return AutoTokenizer.from_pretrained(model_path)
    else:
        return AutoTokenizer.from_pretrained(model_name)


class PreprocessParams(TypedDict):
    raw_content: str | list[str]
    strategy_name: NotRequired[str]
    model_name: NotRequired[str]
    model_path: NotRequired[str]
    model_max_length: NotRequired[int]
    embed_content: NotRequired[bool]


def preprocess(**kwargs: Unpack[PreprocessParams]) -> dict | None:
    """
    Preprocesses raw XML content for text embedding and optionally embed the contents with embedding
    model of choice.

    This function performs several preprocessing steps on raw XML content, including:
    - Tokenizing the content using different tokenizers based on the model name or path.
    - Parsing the XML content to extract metadata and article body.
    - Cleaning and modifying the HTML content within the article body.
    - Converting the cleaned HTML content to Markdown.
    - Splitting the Markdown content into chunks based on headers.
    - Further splitting the chunks if they exceed the maximum token length.
    - Embedding the content if specified.

    Parameters:
    - raw_content (str): List of strings or raw XML content to preprocess.
    - strategy_name (str): The name of the search strategy.
    - model_name (str): The name of the model (in Huggingface) to use for tokenization.
    - model_path (str): The path to the model to use for tokenization.
    - model_max_length (int): The maximum token length for the model.
    - embed_content (bool): Whether to embed the content after preprocessing.

    Returns:
    - dict or None: A list of dictionaries, each containing the preprocessed content and metadata,
                    or None if the function returns early, if the content doesn't have a title for
                    example .
    """
    strategy_name = kwargs.get('strategy_name') if kwargs.get(
        'strategy_name') is not None else 'e5-instruct'
    model_name = kwargs.get('model_name') if kwargs.get(
        'model_name') is not None else 'intfloat/multilingual-e5-large-instruct'
    model_path = kwargs.get('model_path')
    raw_content = kwargs.get('raw_content')
    model_max_length = kwargs.get('model_max_length')
    embed_content = kwargs.get('embed_content')

    if not raw_content:
        raise Exception("No raw content provided")

    # Get cached tokenizer
    tokenizer = get_tokenizer(model_name, model_path)

    # Max token length to be used, sometimes you might want to use smaller max_length than the
    # model's max_length
    max_length = model_max_length if model_max_length is not None else tokenizer.model_max_length

    # If max_length is longer than the model's max_length, throw an error
    if (max_length > tokenizer.model_max_length):
        raise Exception(
            f"Max length is too long: {model_max_length}, should be less than equal to {tokenizer.model_max_length}")

    # Allow max_length be longer than default so we can use it as a length function on text splitter
    # so it doesn't throw an error
    tokenizer.model_max_length = 99999

    # if raw content is a list of strings
    if isinstance(raw_content, str):
        raw_content = [raw_content]

    all_documents = []
    for raw in raw_content:
        # Initialize documents array
        documents = []

        # read raw xml content
        soup = BeautifulSoup(markup=raw, features="xml", from_encoding="ISO-8859-1")

        # get identifier
        identifier_tag = soup.find('identifier')

        # If identifier tag is found and it starts with duo, it is a magazine article
        if identifier_tag and identifier_tag.text.startswith('duo'):
            year = soup.select("meta_journal > year")
            if year is not None and len(year) > 0:
                year = year[0].text
                # Find tag inside <term> with content Katsaus inside <meta-index> tag
                is_katsaus = soup.find('term', string='Katsaus')
                if is_katsaus is None:
                    # print(f"Skipping file {identifier_tag.text} not a Katsaus")
                    return

        # If meta journal tag is found it is a magazine article, then collect all the tags and their
        # values inside the tag.
        # If meta_journal tags is found and the year is less than 2020 lets skip the file
        # Content looks something like this:
        # <meta_journal>
        #     <year>2021</year>
        #     <volume>137</volume>
        #     <issue>1</issue>
        #     <first_page>1</first_page>
        #     <last_page>8</last_page>
        #  </meta_journal>
        # If the year is 2020 or greater lets store the data
        meta_journal = soup.find('meta_journal')
        meta_journal_data = None
        if meta_journal is not None:
            year = get_text_safely(meta_journal, "year")
            if year is not None and int(year) < 2020:
                # print(f'------- year: {year} ------')
                # print(f"Skipping file {identifier_tag.text} with year {year}")
                return
            meta_journal_data = {}
            # If meta_journal tag is found, it is a magazine article, then collect all the tags and
            # their values inside the tag
            if isinstance(meta_journal, Tag):
                for child in meta_journal.children:
                    if isinstance(child, Tag) and child.name is not None:
                        # convert child.text to number if possible
                        meta_journal_data[child.name] = int(
                            child.text) if child.text.isdigit() else child.text

        # Collect dates
        meta_update = soup.find('meta_update')
        created_date = get_text_safely(meta_update, "created")
        updated_date = get_text_safely(meta_update, "updated")

        # Take body element, it contains the html
        article = soup.find('body')

        # If no body tag is found, skip the file
        if article is None or not isinstance(article, Tag):
            # print("Skipping: No body tag found")
            return

        # Create a map for h2 and h3 tags
        h2_id_map = {}
        # add ids to h2 elements
        h2_index = 1
        h3_index = 1
        curr_h2 = ""
        all_titles = article.find_all(
            ['h2', 'h3']) if isinstance(article, Tag) else []
        for index, tag in enumerate(all_titles):

            if tag.name == 'h2':
                if tag.get('id') is None:
                    h2_id = f"s{h2_index}"
                    h2_id_map[tag.text] = {'tag': h2_id, 'h3_tags': {}}
                    tag['id'] = h2_id
                    curr_h2 = tag.text
                    h2_index += 1
                else:
                    h2_id_map[tag.text] = {'tag': tag.get('id'), 'h3_tags': {}}
                    # curr_h2 = tag.text
                h3_index = 1
            if tag.name == 'h3' and tag.get('id') is None and curr_h2 != "":
                tag['id'] = f"{h2_id_map[curr_h2]['tag']}_{h3_index}"
                h2_id_map[curr_h2]['h3_tags'][tag.text] = tag['id']
                h3_index += 1
            # if tag.name != 'h3':
            # print(tag.name, tag.text, tag.get('id'))

        # Find all self closing figure tags
        if isinstance(article, Tag):
            for tag in article.find_all('figure'):
                # if figure doesn't have any children
                if tag.findChildren() is None or len(tag.findChildren()) == 0:
                    # if tag has href attribute, convert it to image tag with "remove:"" prefix
                    # so we can later remove it from the chunk and we can keep track of the images
                    # on the chunk level
                    if tag.get('href') is not None:
                        img_tag = soup.new_tag('img')
                        img_tag['src'] = 'remove:'+tag.get('href')
                        tag.insert(0, img_tag)

        # Remove comments and processing instructions from the html
        for e in article.find_all(string=lambda text:
                                  isinstance(text, (Comment, ProcessingInstruction))):
            if isinstance(e, ProcessingInstruction):
                e.extract()

        # Find all <a> tags with attribute type='reference'
        for tag in article.find_all('a'):
            if not tag.get_text(strip=True):
                # take tag title attribute and add it as a content of the tag
                if tag.get('title') is not None:

                    tag.string = tag.get('title')
                elif tag.get('href') is not None:
                    tag.string = tag.get('href')

            # Check if the tag is empty and the next sibling is a comma
            # if (not tag.get_text(strip=True)) and tag.next_sibling:
            #     next_sibling = tag.next_sibling
            #     if isinstance(next_sibling, str) and next_sibling.startswith(', '):
            #         # If the next sibling is a comma, remove it
            #         next_sibling.extract()

            # Remove the <a> tag
            # tag.decompose()

        # Remove reference tags
        for tag in article.find_all('reference'):
            tag.decompose()

        # Find "Kirjallisuusviite" title and remove everything after that from the html, this part
        # contains all the references. Kirjallisuusviite is usually after the article content so we
        # can quite safely remove everything after that and not lose any important information and
        # save space
        references_container = article.find(
            'h2', string='Kirjallisuusviite')
        if references_container is not None:
            if references_container.parent is not None:
                for e in references_container.parent.find_all_next():
                    if e is not None and isinstance(e, Tag):
                        e.decompose()

        # Find "Suomalaisen lÃ¤Ã¤kÃ¤riseuran Duodecimin" title and remove everything after that from
        # the html. This part contains all the writers and information related to them. Did this
        # because it is not relevant for the search index. Method is a bit hacky but it seems to
        # work, if there is a better way to do this, please let me know
        reference_block_container = article.find(
            'h2', string=lambda text: isinstance(text, str) and
            ("suomalaisen lÃ¤Ã¤kÃ¤riseuran duodecimin" in text.lower().replace('\u00A0', ' ')
             or "suomalaisen lÃ¤Ã¤kÃ¤riseura duodecimin" in text.lower().replace('\u00A0', ' '))
            and ("asettama tyÃ¶ryhmÃ¤" in text.lower().replace('\u00A0', ' ')
                 or "nimeÃ¤mÃ¤ tyÃ¶ryhmÃ¤" in text.lower().replace('\u00A0', ' ')))

        if reference_block_container is not None and reference_block_container.parent is not None:

            for e in reference_block_container.parent.find_all_next():
                if e is not None and isinstance(e, Tag):
                    e.decompose()

        # Remove all the exclude tags from the html
        exclude_tags = article.find_all('exclude')
        if exclude_tags is not None:
            for e in exclude_tags:
                e.decompose()

        # Remove all empty li tags, sometimes these are used as separators and contains no text
        for tag in article.find_all('li'):
            if not tag.get_text(strip=True):
                tag.decompose()

        # Convert html to markdown so we get "flat" structure
        # with headers and paragraphs and lists
        # This removes all unnecessary divs from the html

        # Replace markdown bullets with custom ones so we can add indentation back later on
        # Langchain text splitter removes indentation from the text which might have
        # significance in the text for language model
        md = markdownify(str(article), heading_style="ATX", escape_misc=False)

        # Split markdown text into chunks based on these headers
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
            ("#####", "Header 5"),
            ("######", "Header 6")
        ]
        # Find identifier, title and keywords from the xml
        identifier_tag = soup.find('identifier')
        identifier = identifier_tag.text if identifier_tag else None
        title_tag = article.find('h1')
        title = title_tag.text if title_tag else None
        if title is None:
            heading_tag = article.find('heading', {"class": "1"})
            title = heading_tag.text if heading_tag else None
            if title is None:
                # print("Skipping: No title found")
                return
        keywords = [element.get_text().strip()
                    for element in soup.find_all('term')]
        detailed_keywords = [{'value': element.get_text().strip(),
                              'term': element.get('term'),
                              'priority': element.get('priority'),
                              'thesaurus': element.get('thesaurus')}
                             for element in soup.find_all('term')]

        # Initialize splitter
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on, strip_headers=True)

        # Split text
        md_header_splits = markdown_splitter.split_text(md)

        # Add indentation back to the list items
        for index, dd in enumerate(md_header_splits):
            dd.page_content = dd.page_content.replace("<*>", "*")
            dd.page_content = dd.page_content.replace("<**>", "  *")
            dd.page_content = dd.page_content.replace("<***>", "    *")
            dd.page_content = dd.page_content.replace("<****>", "      *")
            dd.page_content = dd.page_content.replace("<*****>", "        *")
            dd.page_content = dd.page_content.replace(
                "<*******>", "          *")

            # Add metadata to each chunk
            dd.metadata['doc_id'] = identifier
            dd.metadata['keywords'] = keywords
            dd.metadata['detailed_keywords'] = detailed_keywords
            dd.metadata['title'] = title
            dd.metadata['doc_db'] = identifier[:3] if identifier else 'unk'  # pylint: disable=unsubscriptable-object
            dd.metadata["created"] = created_date
            dd.metadata["updated"] = updated_date
            if meta_journal_data is not None:
                dd.metadata['meta_journal'] = meta_journal_data
            dd.metadata['is_magazine_article'] = True if meta_journal_data is not None else False
            if 'Header 2' in dd.metadata and dd.metadata['Header 2'] in h2_id_map.keys():
                dd.metadata['h2_id'] = h2_id_map[dd.metadata['Header 2']]['tag']
                if 'Header 3' in dd.metadata and dd.metadata['Header 3'] in h2_id_map[dd.metadata['Header 2']]['h3_tags'].keys():
                    dd.metadata['h3_id'] = h2_id_map[dd.metadata['Header 2']
                                                     ]['h3_tags'][dd.metadata['Header 3']]

        # Add chunks to documents array
        documents.extend(md_header_splits)

        # Initialize chunked documents array
        chunked_docs = []

        # Length function for text splitter, uses tokenizer to get length of the text
        def length_function(text: str):
            return len(tokenizer.encode(text))

        # Split documents into further chunks if they are longer than max_length
        for index, doc in enumerate(tqdm.tqdm(documents, colour='yellow',
                                              desc="Splitting documents", position=1, leave=False)):

            # Concatenate headers to one string and add it to metadata, we use this later on
            # to add headers back to each chunk
            try:
                headers = []
                headers.append(doc.metadata['Header 1'])
                headers.append(doc.metadata['Header 2'])
                headers.append(doc.metadata['Header 3'])
                headers.append(doc.metadata['Header 4'])
                headers.append(doc.metadata['Header 5'])
                headers.append(doc.metadata['Header 6'])
                # if doc.metadata['Header 2'] != '':
                md_headers = []
                for index, header in enumerate(headers):
                    md_headers.append(f"{'#' * (index+1)} {header}")
                all_headers = "\n".join(md_headers)
                doc.metadata['headers'] = headers
                doc.metadata['all_headers'] = f"{all_headers}"
            except:
                try:
                    md_headers = []
                    for index, header in enumerate(headers):
                        md_headers.append(f"{'#' * (index+1)} {header}")
                    all_headers = "\n".join(md_headers)
                    doc.metadata['headers'] = headers
                    doc.metadata['all_headers'] = f"{all_headers}"

                except:
                    pass

            try:
                # remove headers from the metadata
                del doc.metadata['Header 1']
                del doc.metadata['Header 2']
                del doc.metadata['Header 3']
                del doc.metadata['Header 4']
                del doc.metadata['Header 5']
                del doc.metadata['Header 6']
            except:
                pass

            doc_header = f"{doc.metadata['all_headers']}\n---\n"
            encoded = tokenizer.encode(doc.page_content)
            headers_size = len(tokenizer.encode(doc_header))

            # Initialize text splitter
            # text_splitter = RecursiveCharacterTextSplitter.from_language(
            # language=Language.MARKDOWN, chunk_size=max_length-headers_size, chunk_overlap=0,
            # length_function=length_function)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=max_length-headers_size, chunk_overlap=0,
                length_function=length_function, keep_separator=True,
                separators=['\n\n', '\n', '. ', '.', ' ', '']
            )

            # If any of the chunks after "header splitting" is longer than
            # max_length - headers_size, split it further
            if len(encoded) > max_length-headers_size:
                texts = text_splitter.split_text(doc.page_content)

                for text in texts:
                    # Add headers back to each chunk and metadata
                    new_doc = Document(
                        # Lets make a copy of metadata since now the images could be different per
                        # chunk
                        page_content=f"{doc_header}{text}", metadata=doc.metadata.copy())

                    chunked_docs.append(new_doc)

            else:
                new_doc = Document(
                    # Lets make a copy of metadata since now the images could be different per chunk
                    page_content=f"{doc_header}{doc.page_content}", metadata=doc.metadata.copy())
                chunked_docs.append(new_doc)
        # Regular expression to match markdown image tags
        image_pattern = r'!\[(.*?)\]\((.*?)\)'

        for doc in chunked_docs:
            image_list = []

            def process_image(match):
                alt_text, url = match.groups()

                image_list.append(url.replace("remove:", "", 1))

                if url.startswith('remove:'):
                    new_url = url.replace("remove:", "", 1)
                    # Convert to link tag and remove 'remove:' prefix
                    return f'[{alt_text if alt_text else new_url}]({new_url})'
                else:
                    # Keep as image tag
                    return match.group(0)

            # Process the document to collect URLs and convert specific images to links
            cleaned_doc = re.sub(
                image_pattern, process_image, doc.page_content)

            doc.metadata['images'] = image_list
            doc.page_content = cleaned_doc

        # Convert documents to json
        chunked_docs = [{"page_content": doc.page_content,
                        "metadata": doc.metadata} for doc in chunked_docs]

        # Check that all chunks are less than max_length
        current_doc_id = ""
        index = 0
        for dd in chunked_docs:
            index += 1
            if current_doc_id == dd['metadata']['doc_id']:
                dd['id'] = f"{dd['metadata']['doc_id']}-{index}"
            else:
                index = 1
                dd['id'] = f"{dd['metadata']['doc_id']}-{index}"
                current_doc_id = dd['metadata']['doc_id']

            length = len(tokenizer.encode(dd['page_content']))
            if length > max_length:
                raise Exception(
                    (
                        f"Chunked document length is too long: {length} tokens, "
                        "should be less than {max_length} tokens"
                    ))

        tokenizer.model_max_length = max_length
        # print('here we are')
        # Embed chunks if --embed is set to True
        if model_name is not None and len(model_name) > 0 and embed_content is True:

            chunk_size = 16
            with tqdm.tqdm(total=len(chunked_docs), desc="Embedding chunk", colour="green",
                           position=1, leave=False) as pbar_chunk:
                for i in range(0, len(chunked_docs), chunk_size):
                    # if i > 1:
                    #     break
                    batch = chunked_docs[i:i + chunk_size]

                    # Prepare data for each embedding type
                    content = [doc['page_content'] for doc in batch]

                    # for embedModes in args.embed:
                    model = model_name
                    path = model_path
                    emb = embed(texts=content, model=model,
                                path=path, is_query=False)

                    # Assign embeddings back to the original documents
                    for j, doc in enumerate(batch):
                        if emb is not None:
                            doc['embedding'] = emb[j]

                    pbar_chunk.update(len(batch))
        all_documents.extend(chunked_docs)

    json_object = {
        "name": strategy_name,
        "model": model_name,
        "path": model_path,
        "max_tokens": max_length,
        "embedding_dimensions": embedding_dimension_map[model_name] if model_name else None,
        "embedding_pooling": pooling_strategy[model_name] if model_name else None,
        "data": all_documents
    }

    return json_object