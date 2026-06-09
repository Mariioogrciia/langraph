from agente import app

# Generar la imagen del grafo usando la API de Mermaid
try:
    print("Generando imagen del grafo de LangGraph...")
    imagen = app.get_graph(xray=True).draw_mermaid_png()
    
    with open("grafo.png", "wb") as f:
        f.write(imagen)
        
    print("¡Éxito! Se ha guardado la imagen del grafo como 'grafo.png' en esta misma carpeta.")
except Exception as e:
    print(f"Error al generar la imagen: {e}")
