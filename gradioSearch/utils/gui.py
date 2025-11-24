"""
Gradio GUI components for gradioSearch
"""

import gradio as gr
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import json


def create_search_interface(
    search_function, metadata_keys: List[str], topk: int = 50
) -> gr.Blocks:
    """
    Create Gradio interface for FAISS database search

    Args:
        search_function: Function that takes query string and topk, returns search results
        metadata_keys: List of metadata column names to display
        topk: Number of top results to retrieve

    Returns:
        Gradio Blocks interface
    """

    def format_search_results(
        results: List[Dict[str, Any]],
    ) -> Tuple[pd.DataFrame, str]:
        """
        Format search results into dataframe and detail view

        Args:
            results: List of search result dictionaries with 'content', 'metadata', 'similarity_score'

        Returns:
            Tuple of (dataframe, empty_detail_text)
        """
        if not results:
            empty_df = pd.DataFrame(
                columns=["Similarity"] + metadata_keys + ["Content"]
            )
            return empty_df, ""

        # Prepare dataframe data
        df_data = []
        for i, result in enumerate(results):
            row = {}

            # Add similarity score as first column (rounded to 0.01)
            row["Similarity"] = round(result.get("similarity_score", 0.0), 2)

            # Add metadata columns
            metadata = result.get("metadata", {})
            for key in metadata_keys:
                row[key] = metadata.get(key, "")

            # Add cropped content (limit to 100 chars)
            content = result.get("content", "")
            row["Content"] = content[:100] + "..." if len(content) > 100 else content

            df_data.append(row)

        df = pd.DataFrame(df_data)
        return df, ""

    def on_row_select(evt: gr.SelectData, results_state) -> Tuple[str, str]:
        """
        Handle row selection in dataframe to show document details

        Args:
            evt: Gradio SelectData event
            results_state: Current search results

        Returns:
            Tuple of (content_text, metadata_text)
        """
        if not results_state or evt.index[0] >= len(results_state):
            return "", ""

        selected_result = results_state[evt.index[0]]
        content = selected_result.get("content", "")
        metadata = selected_result.get("metadata", {})

        # Format the content
        content_text = f"**Document Content:**\n\n{content}"

        # Format metadata as bullet points with bold keys
        if metadata:
            metadata_lines = []
            for key, value in metadata.items():
                metadata_lines.append(f"- **{key}:** {value}")
            metadata_text = "\n".join(metadata_lines)
        else:
            metadata_text = "*No metadata available*"

        return content_text, metadata_text

    def perform_search(
        query: str, results_state
    ) -> Tuple[pd.DataFrame, str, str, List[Dict]]:
        """
        Perform search and update interface

        Args:
            query: Search query string
            results_state: Current results state

        Returns:
            Tuple of (updated_dataframe, empty_content, empty_metadata, new_results_state)
        """
        # Perform the search with topk parameter (empty queries return initial documents)
        results = search_function(query, topk)

        # Format results for display
        df, _ = format_search_results(results)

        return df, "", "", results

    # Get initial results to populate the dataframe on load
    initial_results = search_function("", topk)
    initial_df, _ = format_search_results(initial_results)

    # Create the Gradio interface
    with gr.Blocks(title="gradioSearch - FAISS Database Search") as interface:
        gr.Markdown("# gradioSearch - FAISS Database Search")

        # State to store current search results
        results_state = gr.State(initial_results)

        # Search input at the top
        search_input = gr.Textbox(
            label="Search Query",
            placeholder="Enter your search query and press Enter...",
            lines=1,
        )

        # Layout with dataframe on left, details on right
        with gr.Row():
            with gr.Column(scale=2):
                results_df = gr.Dataframe(
                    label="Search Results",
                    headers=["Similarity", "Content"] + metadata_keys,
                    datatype=["number"] + ["str"] * (len(metadata_keys) + 1),
                    interactive=False,
                    wrap=True,
                    row_count=(10, "dynamic"),
                    column_widths=["10%"] + ["15%"] * len(metadata_keys) + ["30%"],
                    value=initial_df,
                    max_height=10_000,
                    show_search="filter",
                    pinned_columns=2,
                    show_row_numbers=True,
                    #buttons=["fullscreen", "copy"],
                    line_breaks=True,
                )

            with gr.Column(scale=1):
                document_content = gr.Markdown(
                    label="Document Content",
                    value="Select a row from the search results to view document details.",
                )

                with gr.Accordion(label="Metadata", open=False):
                    document_metadata = gr.Markdown(value="*No metadata available*")

        # Event handlers
        search_input.submit(
            fn=perform_search,
            inputs=[search_input, results_state],
            outputs=[results_df, document_content, document_metadata, results_state],
        )

        results_df.select(
            fn=on_row_select,
            inputs=[results_state],
            outputs=[document_content, document_metadata],
        )

    return interface


def launch_gui(
    search_function,
    metadata_keys: List[str],
    topk: int = 50,
    share: bool = False,
    server_port: Optional[int] = None,
) -> None:
    """
    Launch the Gradio interface

    Args:
        search_function: Function that performs the search
        metadata_keys: List of metadata keys to display
        topk: Number of top results to retrieve
        share: Whether to create a public link
        server_port: Port to run the server on
    """
    interface = create_search_interface(search_function, metadata_keys, topk)
    interface.launch(share=share, server_port=server_port, server_name="0.0.0.0")
