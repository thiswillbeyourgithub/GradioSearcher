I want you to create gradioSearch, a python project organized in the gradioSearch folder, containing init main gradioSearch.py and a utils folder.
the goal of this project is to have a cli tool that takes as arg --db_path which leads to a langchain faiss db. That db contains langchain Documents made of content and metadata as well as embeddings.
it also takes as arg an embedding engine model name to pass to sbert to be used to embed the user query.
it also takes an arg "metadata_keys" which is a str for a comma separated list of dstaframe columns.
in utils/gui.py you create a gradio Block that contains a text bar for search queries at the top, and below on the left a datafram that will display the results of the search with select columns in the asked order (if missing a column we assume its empty). and to the right, we render/pretty print the content and metadata of the selected document.
we need an arg to specify the topk to retrieve by search too.

the search must occur when we press enter


