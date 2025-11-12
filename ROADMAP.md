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
- ADDED: utils folder with __init__.py. REASON: package structure for GUI components. STATUS: done

## IN_PROGRESS
- CREATE: utils/gui.py with Gradio interface. REASON: next step for interactive search GUI. STATUS: pending

## TODO
- P0: Implement FAISS database loading and search functionality in gradioSearch.py
- P1: Add document embedding with sentence-transformers integration
- P1: Connect GUI to main CLI entry point
- P2: Add error handling and validation

## DECISIONS
- STRUCTURE: Following standard Python package layout with utils subfolder
- GUI: Using Gradio Blocks for flexible layout control
- SEARCH: Enter key trigger for search execution
- DISPLAY: Similarity score as first column, rounded to 0.01
