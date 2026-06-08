import os
import sqlite3
from typing import TypedDict, Optional
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

# =====================================================================
# 0. CONFIGURACIÓN INICIAL Y BASE DE DATOS DE PRUEBA (SQLite)
# =====================================================================
conn = sqlite3.connect(":memory:", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE productos (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nombre TEXT COLLATE NOCASE NOT NULL,
        precio REAL NOT NULL,
        stock INTEGER NOT NULL,
        caracteristicas TEXT
    )
""")
cursor.executemany(
    "INSERT INTO productos (nombre, precio, stock, caracteristicas) VALUES (?, ?, ?, ?)",
    [
        ("iPhone 15", 900.0, 15, "Chip A16 Bionic, pantalla Super Retina XDR OLED de 6.1 pulgadas, cámara principal de 48MP, Dynamic Island, conector USB-C, diseño de aluminio y vidrio tintado."),
        ("MacBook Air", 1200.0, 8, "Chip M3, pantalla Liquid Retina de 13.6 pulgadas, hasta 18 horas de autonomía, teclado Magic Keyboard retroiluminado, Touch ID, grosor ultradelgado."),
        ("AirPods Pro", 250.0, 40, "Cancelación Activa de Ruido, modo ambiente, Audio Espacial, estuche de carga MagSafe (USB-C), resistencia al agua y sudor, puntas de silicona."),
        ("iPad Pro", 1100.0, 5, "Chip M4, pantalla Ultra Retina XDR OLED de 11 pulgadas, soporte para Apple Pencil Pro, Magic Keyboard rediseñado, diseño de 5.3mm, cámaras Pro y escáner LiDAR.")
    ]
)
conn.commit()

# Configura tus variables de entorno para Azure OpenAI:
os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY", "tu_clave_aqui")
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT", "https://foundryasistente.openai.azure.com/")

model = AzureChatOpenAI(
    azure_deployment="gpt-4o-mini",
    api_version="2024-02-15-preview",
    temperature=0
)

# =====================================================================
# 1. DEFINICIÓN DEL ESTADO
# =====================================================================
class AgentState(TypedDict):
    pregunta_usuario: str
    sql_query: Optional[str]
    db_result: Optional[str]
    respuesta_final: Optional[str]

# =====================================================================
# 2. DEFINICIÓN DE LOS NODOS
# =====================================================================
def generar_query_node(state: AgentState):
    print("🧠 [Nodo: Agente AI] Generando la Query SQL...")
    prompt = f"""Eres un experto en SQL. Dada la siguiente pregunta del usuario, genera ÚNICAMENTE la consulta SQL SQLite para obtener la información solicitada. 
    La tabla se llama 'productos' y tiene las columnas (id, nombre, precio, stock, caracteristicas).
    NO des explicaciones, NO envuelvas en markdown, SOLO la query SQL cruda.
    
    Pregunta del usuario: {state['pregunta_usuario']}
    """
    
    response = model.invoke([HumanMessage(content=prompt)])
    query = response.content.strip().replace("```sql", "").replace("```", "").strip()
    return {"sql_query": query}

def ejecutar_db_node(state: AgentState):
    print(f"🛠️ [Nodo: Base de Datos] Ejecutando Query Segura: {state['sql_query']}")
    try:
        cursor.execute(state['sql_query'])
        resultados = cursor.fetchall()
        resultado_str = str(resultados) if resultados else "No se encontraron resultados."
    except Exception as e:
        resultado_str = f"Error al ejecutar SQL: {str(e)}"
    
    return {"db_result": resultado_str}

def generar_rechazo_node(state: AgentState):
    print("⚠️ [Nodo: Rechazo] Generando aviso de seguridad...")
    return {"db_result": "ACCESO DENEGADO: La consulta intentó modificar la base de datos o contenía palabras prohibidas."}

def generar_respuesta_node(state: AgentState):
    print("🗣️ [Nodo: Agente Respuesta] Traduciendo a lenguaje humano...")
    prompt = f"""Eres un asistente de inventario. El usuario hizo una pregunta. Se ejecutó una consulta (o se bloqueó por seguridad) y obtuvimos este resultado.
    
    Pregunta original: {state['pregunta_usuario']}
    Consulta SQL intentada: {state['sql_query']}
    Resultado obtenido: {state['db_result']}
    
    Responde al usuario de forma natural, clara y amigable usando la información del resultado obtenido. Si hubo un error o bloqueo, explícaselo amablemente.
    """
    
    response = model.invoke([HumanMessage(content=prompt)])
    return {"respuesta_final": response.content}

# =====================================================================
# 3. LÓGICA CONDICIONAL (Edges)
# =====================================================================
def validar_seguridad(state: AgentState):
    print(f"🔍 [Condición] Validando la consulta: {state['sql_query']}")
    query_upper = state['sql_query'].upper()
    palabras_peligrosas = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE", "TRUNCATE"]
    
    if any(palabra in query_upper for palabra in palabras_peligrosas):
        print("   -> ❌ Consulta INSEGURA (rechazada).")
        return "insegura"
    else:
        print("   -> ✅ Consulta SEGURA.")
        return "segura"

# =====================================================================
# 4. CONSTRUCCIÓN DEL GRAFO
# =====================================================================
workflow = StateGraph(AgentState)

# Añadimos los nodos
workflow.add_node("generar_query", generar_query_node)
workflow.add_node("ejecutar_db", ejecutar_db_node)
workflow.add_node("rechazo", generar_rechazo_node)
workflow.add_node("generar_respuesta", generar_respuesta_node)

# Añadimos las conexiones (Edges)
workflow.add_edge(START, "generar_query")

# La decisión condicional (rombo)
workflow.add_conditional_edges(
    "generar_query",
    validar_seguridad,
    {
        "segura": "ejecutar_db",
        "insegura": "rechazo"
    }
)

# Ambos caminos llevan a la generación de respuesta final
workflow.add_edge("ejecutar_db", "generar_respuesta")
workflow.add_edge("rechazo", "generar_respuesta")

workflow.add_edge("generar_respuesta", END)

# Compilamos el grafo
app = workflow.compile()

# =====================================================================
# 5. MODO INTERACTIVO (Chat)
# =====================================================================
if __name__ == "__main__":
    print("👋 ¡Hola! Soy tu Agente basado en un Grafo Personalizado (LangGraph).")
    print("Flujo: Generar Query -> Validar -> Ejecutar DB o Rechazar -> Responder.")
    print("Escribe 'salir' para terminar la conversación.")
    print("=" * 60)

    while True:
        pregunta = input("\n👤 Tú: ")
        
        if pregunta.lower() in ["salir", "exit", "quit", "q"]:
            print("¡Hasta luego! 👋")
            break
            
        if pregunta.strip():
            # Ejecutamos el grafo con el estado inicial
            initial_state = {"pregunta_usuario": pregunta}
            
            print("-" * 40)
            try:
                # invoke ejecutará todo el grafo hasta el nodo END
                resultado = app.invoke(initial_state)
                print("-" * 40)
                print(f"\n🤖 Agente Final:\n{resultado['respuesta_final']}")
                print("=" * 60)
            except Exception as e:
                print(f"\n🛡️ [Error en el grafo] {str(e)}")
                print("=" * 60)
