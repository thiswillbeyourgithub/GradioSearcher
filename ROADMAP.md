# gradioSearch Project Roadmap

## PROGRESS
- Overall completion: 0%
- TODOs remaining: 8
- Active issues: 0

## OBJECTIVES
- CREATE: CLI tool for searching FAISS vector databases with Gradio GUI
- SUPPORT: Langchain Documents with content, metadata, and embeddings
- PROVIDE: Interactive search interface with dataframe results and document preview
- ENABLE: Configurable embedding models, metadata columns, and top-k results

## COMPLETED
(none yet)

## IN_PROGRESS
- ADDED: CLI argument parsing in gradioSearch.py. REASON: foundation for all other functionality. STATUS: done

## TODO
- P0: Create utils folder with __init__.py
- P0: Create utils/gui.py with Gradio interface
- P1: Implement FAISS database loading and search functionality
- P1: Add document embedding with sentence-transformers
- P1: Create dataframe display with similarity scores and metadata
- P2: Add document content preview panel with pretty printing

## DECISIONS
- STRUCTURE: Following standard Python package layout with utils subfolder
- GUI: Using Gradio Blocks for flexible layout control
- SEARCH: Enter key trigger for search execution
- DISPLAY: Similarity score as first column, rounded to 0.01
