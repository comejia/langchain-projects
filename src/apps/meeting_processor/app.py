from meeting_processor import create_workflow, State

# from tkinter import Tk, filedialog
from transcriber import transcribe_media


workflow = create_workflow()


def process_meeting_notes(notes: str, workflow):
    """Procesa una nota de reunión individual."""

    initial_state = {
        "notes": notes,
        "participants": [],
        "topics": [],
        "action_items": [],
        "minutes": "",
        "summary": "",
        "logs": [],
    }

    print("\n" + "=" * 60)
    print("🔄 Procesando nota de reunión...")
    print("=" * 60)

    result = workflow.invoke(initial_state)
    return result


def display_results(result: State, meeting_num: int):
    """Muestra los resultados de forma estructurada."""
    print(f"\n📋 RESULTADOS - REUNIÓN #{meeting_num}")
    print("-" * 60)

    print(f"\n👥 Participantes ({len(result['participants'])}):")
    for p in result["participants"]:
        print(f"   • {p}")

    print(f"\n📍 Temas tratados ({len(result['topics'])}):")
    for t in result["topics"]:
        print(f"   • {t}")

    print(f"\n✅ Acciones acordadas ({len(result['action_items'])}):")
    if result["action_items"]:
        for a in result["action_items"]:
            print(f"   • {a}")
    else:
        print("   • No se definieron acciones específicas")

    print("\n📄 MINUTA FORMAL:")
    print("-" * 40)
    print(result["minutes"])
    print("-" * 40)

    print("\n💡 RESUMEN EJECUTIVO:")
    print(f"   {result['summary']}")

    print("\n📝 LOGS DEL PROCESO:", result["logs"])

    print("\n" + "=" * 60)


if __name__ == "__main__":
    workflow = create_workflow()
    file_path = "/home/comejia/projects/langchain-project/src/apps/meeting_processor/Simulacion_reunion.mp4"

    notes = transcribe_media(file_path)

    result = process_meeting_notes(notes, workflow)
    display_results(result, 1)
