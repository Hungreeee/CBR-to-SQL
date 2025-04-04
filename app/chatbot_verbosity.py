import sys
sys.dont_write_bytecode = True

import streamlit as st
import numpy as np

def render(document_list: list, time_elapsed: float):
    retriever_message = st.expander(f"Verbosity")
    message_map = {
        "retrieve_applicant_jd": "**A job description is detected**. The system defaults to using RAG.",
        "retrieve_applicant_id": "**Applicant IDs are provided**. The system defaults to using exact ID retrieval.",
        "no_retrieve": "**No retrieval is required for this task**. The system will utilize chat history to answer."
    }

    with retriever_message:
        st.markdown(f"Total time elapsed: {np.round(time_elapsed, 3)} seconds")
        st.markdown(f"Returning top 5 most relevant documents.")

        button_columns = st.columns([0.2, 0.2, 0.2, 0.2, 0.2], gap="small")
        for index, document in enumerate(document_list[:5]):
            with button_columns[index], st.popover(f"Resume {index + 1}"):
                st.markdown(document)

if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2])

