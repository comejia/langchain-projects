import PyPDF2
from io import BytesIO

def pdf_to_text(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_file.read()))
        full_text = ""
        for page_number, page in enumerate(pdf_reader.pages, start=1):
            text_page = page.extract_text()
            if text_page.strip():
                full_text+= f"\n--- PÁGINA {page_number}---\n"
                full_text += text_page + "\n"

        full_text = full_text.strip()

        if not full_text:
            return "Error: el PDF parece estar vacío o contiene solo imagenes."

        return full_text

    except Exception as e:
        return f"Error al procesar el archivo PDF: {str(e)}"

