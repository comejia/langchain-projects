from pydantic import BaseModel, Field

class CVAnalyzer(BaseModel):
    """Modelo de datos para el análisis completo de un CV."""

    name: str = Field(description="Nombre completo del candidato extraído del CV.")
    experience_year: int = Field(description="Años de experiencia laboral relevante.")
    skills: list[str] = Field(description="Lista de las 5-7 habilidades más relevantes para el puesto.")
    education: str = Field(description="Nivel educativo más alto y especialización principal.")
    experience_key: str = Field(description="Resumen conciso de la experiencia más relevante para el puesto especifico.")
    strengths: list[str] = Field(description="3-5 principales fortalezas del candidato basadas en su perfil.")
    weaknesses: list[str] = Field(description="2-4 areas donde el candidato podría desarrollarse o mejorar.")
    adjustment_percentage: int = Field(description="Porcentaje de ajuste al puesto (0-100) basado en experiencia, habilidades y formación.", ge=0, le=100)

