from core.schemas import WorkflowDefinition, WorkflowStep

ResearchWorkflow = WorkflowDefinition(
    name="research_to_pdf",
    description="Researches a topic and creates a PDF report",
    steps=[
        WorkflowStep(
            name="search",
            tool_name="duckduckgo_web_search",
            prompt="Search for python programming history",
            input_override={"search_key": "Python programming history", "max_results": 5},
            checkpoint=True,
        ),
        WorkflowStep(
            name="create_outline",
            tool_name=None,  # No tool — just LLM response
            prompt="""Based on search results, create an outline.
            Return JSON: {"sections": ["Section 1", "Section 2", ...]}""",
            checkpoint=True,
        ),
        WorkflowStep(
            name="write_content",
            tool_name=None,  # No tool — just LLM response
            prompt="""Write content for each section.
            Return JSON: {"sections": [{"heading": "...", "content": "..."}, ...]}""",
            checkpoint=True,
        ),
        WorkflowStep(
            name="create_pdf",
            tool_name="pdf_creator",
            prompt="Create PDF from the content above",
            input_override={
                "title": "Research Report",
                "output_path": "/tmp/research_output.pdf"
            },
        ),
    ]
)