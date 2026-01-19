# tools/domain/research/pdf_creator.py

from pydantic import BaseModel
from fpdf import FPDF
from core.base_tool import BaseTool
from core.schemas import ToolResult, ExecutionContext


class PDFCreatorInput(BaseModel):
    title: str
    sections: list[dict]  # [{"heading": "...", "content": "..."}, ...]
    output_path: str


class PDFCreatorTool(BaseTool):
    name = "pdf_creator"
    description = "Creates a PDF from structured content"
    input_model = PDFCreatorInput

    async def execute(self, input: PDFCreatorInput, context: ExecutionContext) -> ToolResult:
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 24)
            pdf.cell(0, 20, input.title, ln=True, align="C")
            pdf.ln(10)
            
            for section in input.sections:
                pdf.set_font("Helvetica", "B", 16)
                pdf.cell(0, 10, section["heading"], ln=True)
                pdf.set_font("Helvetica", "", 12)
                pdf.multi_cell(0, 7, section["content"])
                pdf.ln(5)
            
            pdf.output(input.output_path)
            
            return ToolResult(
                success=True,
                tool_name=self.name,
                input=input.model_dump(),
                data={"path": input.output_path, "pages": pdf.page_no()},
            )
        except Exception as e:
            return ToolResult(
                success=False,
                tool_name=self.name,
                input=input.model_dump(),
                error=str(e),
            )