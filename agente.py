import os
import sqlite3
import uuid
import operator
from typing import TypedDict, Optional, Annotated
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send
from langgraph.checkpoint.memory import MemorySaver
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

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
        ("iPad Pro", 1100.0, 5, "Chip M4, pantalla Ultra Retina XDR OLED de 11 pulgadas, soporte para Apple Pencil Pro, Magic Keyboard rediseñado, diseño de 5.3mm, cámaras Pro y escáner LiDAR."),
        ("Apple Watch Series 9", 450.0, 20, "Chip S9, pantalla el doble de brillante, gesto de doble toque, monitoreo de oxígeno en sangre y ECG, resistencia al agua 50m."),
        ("Mac Studio", 2100.0, 3, "Chip M2 Max, 32GB RAM unificada, 512GB SSD, amplia conectividad con puertos Thunderbolt 4, diseño compacto para escritorio profesional."),
        ("HomePod mini", 100.0, 50, "Audio computacional, sonido envolvente 360 grados, integración perfecta con Siri, control del hogar inteligente, diseño de malla sin costuras."),
        ("iMac 24", 1600.0, 7, "Chip M3, pantalla Retina 4.5K de 24 pulgadas, cámara FaceTime HD 1080p, sistema de seis altavoces, teclado Magic Keyboard con Touch ID."),
        ("Apple TV 4K", 160.0, 15, "Procesador A15 Bionic, compatibilidad con Dolby Vision y HDR10+, Siri Remote con control táctil, acceso a Apple Arcade y Fitness+."),
        ("Magic Mouse", 85.0, 25, "Superficie Multi-Touch, batería recargable integrada (hasta un mes por carga), diseño optimizado que se desliza suavemente sobre el escritorio.")
    ]
)
conn.commit()

print("DEBUG ENDPOINT:", os.getenv("AZURE_OPENAI_ENDPOINT"))

model = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment="gpt-4o-mini",
    api_version="2025-01-01-preview",
    temperature=0
)

# =====================================================================
# 1. DEFINICIÓN DEL ESTADO
# =====================================================================
class AgentState(TypedDict):
    pregunta_usuario: str
    sql_query: Optional[str]
    db_result: Optional[str]
    db_raw_results: Optional[list]
    respuesta_final: Optional[str]
    es_segura: Optional[bool]
    query_explicacion: Optional[str]
    resultados_parciales: Annotated[list[str], operator.add]

class ValidationState(TypedDict):
    sql_query: Optional[str]
    es_segura: Optional[bool]
    query_explicacion: Optional[str]

class BatchState(TypedDict):
    productos: list[dict]

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
        raw_results = resultados
    except Exception as e:
        resultado_str = f"Error al ejecutar SQL: {str(e)}"
        raw_results = None
    
    return {"db_result": resultado_str, "db_raw_results": raw_results}

def generar_rechazo_node(state: AgentState):
    print("⚠️ [Nodo: Rechazo] Generando aviso de seguridad...")
    return {"db_result": "ACCESO DENEGADO: La consulta intentó modificar la base de datos o contenía palabras prohibidas."}

def preparar_map_node(state: AgentState):
    print("📋 [Nodo: Preparar Map-Reduce] Convirtiendo resultados a formato de producto...")
    return {"resultados_parciales": []}

def procesar_lote_node(state: BatchState):
    productos = state["productos"]
    print(f"📦 [Nodo Map: Procesar Lote] Analizando lote de {len(productos)} productos...")
    
    texto_productos = ""
    for prod in productos:
        texto_productos += f"- Nombre: {prod['nombre']}, Precio: {prod['precio']} €, Stock: {prod['stock']}, Características: {prod['caracteristicas']}\n"
        
    prompt = f"""Escribe una descripción súper corta y comercial (máximo 15 palabras) para CADA UNO de los siguientes productos del catálogo.
    Devuelve la respuesta exclusivamente como una lista de viñetas, una por cada producto, usando este formato: '- **Nombre** (Precio €): Descripción'.
    Productos:
    {texto_productos}"""
    
    try:
        response = model.invoke([HumanMessage(content=prompt)])
        summary = response.content.strip()
    except Exception as e:
        summary = f"Error al generar descripción para el lote: {str(e)}"
        
    return {"resultados_parciales": [summary]}

def generar_respuesta_node(state: AgentState):
    print("🗣️ [Nodo Reduce: Agente Respuesta] Consolidando respuestas...")
    
    db_res = state.get("db_result")
    parciales = state.get("resultados_parciales", [])
    raw = state.get("db_raw_results")
    
    # Si fue rechazo, cancelación o error, respondemos directamente
    if not raw or "ACCESO DENEGADO" in db_res or "cancelada" in db_res or "Error" in db_res:
        prompt = f"""Eres un asistente de inventario. El flujo de consulta fue bloqueado o cancelado.
        Mensaje del sistema: {db_res}
        Responde al usuario amablemente explicando lo sucedido en una frase natural."""
        response = model.invoke([HumanMessage(content=prompt)])
        return {"respuesta_final": response.content}
        
    # Si saltó el map-reduce por tener <= 5 productos
    if not parciales and raw and len(raw) <= 5:
        print("   -> (Procesando directamente lista pequeña de productos en el Reduce)")
        texto_productos = ""
        for row in raw:
            texto_productos += f"- {row[1]} ({row[2]} €): {row[4]}\n"
        lista_productos = texto_productos
    elif parciales:
        lista_productos = "\n".join(parciales)
    else:
        return {"respuesta_final": "No se encontraron productos en el inventario para esa consulta."}
        
    prompt = f"""Eres un asistente de inventario. Aquí tienes la información de los productos obtenidos:
    {lista_productos}
    
    Escribe una respuesta final muy amigable, natural y profesional para presentarle al usuario este listado de productos.
    Añade descripciones cortas comerciales para cada uno si la información es técnica.
    Formato deseado: un pequeño párrafo de saludo y una lista atractiva con los productos.
    No repitas todos los detalles técnicos, hazlo atractivo para la venta."""
    
    response = model.invoke([HumanMessage(content=prompt)])
    return {"respuesta_final": response.content.strip()}

def distribuir_o_saltar(state: AgentState):
    raw = state.get("db_raw_results")
    db_res = state.get("db_result")
    
    if not raw or "ACCESO DENEGADO" in db_res or "cancelada" in db_res or "Error" in db_res:
        print("⏭️ [Router Map-Reduce] Saltando fase de mapeo (consulta cancelada o sin resultados).")
        return "directo"
        
    if len(raw) <= 5:
        print(f"⏭️ [Router Map-Reduce] Saltando fase de mapeo ({len(raw)} elementos <= 5). Procesamiento directo.")
        return "directo"
        
    sends = []
    dicts = []
    for row in raw:
        dicts.append({
            "id": row[0],
            "nombre": row[1],
            "precio": row[2],
            "stock": row[3],
            "caracteristicas": row[4]
        })
        
    lote_size = 5
    for i in range(0, len(dicts), lote_size):
        lote = dicts[i:i + lote_size]
        sends.append(Send("procesar_lote", {"productos": lote}))
        
    print(f"🚀 [Router Map-Reduce] Generadas {len(sends)} tareas de mapeo por lotes.")
    return sends

def validar_seguridad_node(state: AgentState):
    print("🔍 [Nodo Paralelo: Seguridad] Validando la consulta...")
    query_upper = state['sql_query'].upper() if state['sql_query'] else ""
    palabras_peligrosas = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE", "TRUNCATE"]
    es_segura = not any(palabra in query_upper for palabra in palabras_peligrosas)
    if not es_segura:
        print("   -> ❌ Consulta INSEGURA detectada por el validador paralelo.")
    else:
        print("   -> ✅ Consulta SEGURA detectada por el validador paralelo.")
    return {"es_segura": es_segura}

def auditar_query_node(state: AgentState):
    query = state['sql_query']
    print(f"📋 [Nodo Paralelo: Auditoría] Analizando la consulta: {query}")
    if not query:
        return {"query_explicacion": "No se generó consulta para auditar."}
    
    prompt = f"""Explica en UNA frase corta y técnica qué hace esta consulta SQL en la tabla 'productos':
    {query}
    NO des formato markdown, sé muy conciso (máximo 15 palabras)."""
    
    try:
        response = model.invoke([HumanMessage(content=prompt)])
        explicacion = response.content.strip()
        print(f"   -> 📋 Auditoría completada: {explicacion}")
    except Exception as e:
        explicacion = f"Error en auditoría: {str(e)}"
        print(f"   -> ❌ Error en auditoría: {e}")
    
    return {"query_explicacion": explicacion}

def combinar_y_decidir_node(state: AgentState):
    print("📊 [Nodo: Combinar] Caminos paralelos unidos en barrera de sincronización.")
    return {}

# =====================================================================
# 3. LÓGICA CONDICIONAL (Edges)
# =====================================================================
def decidir_ruta(state: AgentState):
    if state.get("es_segura"):
        return "segura"
    else:
        return "insegura"

# =====================================================================
# 4. CONSTRUCCIÓN DEL SUBGRAFO (Validación y Auditoría)
# =====================================================================
validation_builder = StateGraph(ValidationState)

# Registramos los nodos en el subgrafo
validation_builder.add_node("validar_seguridad", validar_seguridad_node)
validation_builder.add_node("auditar_query", auditar_query_node)
validation_builder.add_node("combinar", combinar_y_decidir_node)

# Estructura de conexiones del subgrafo
validation_builder.add_edge(START, "validar_seguridad")
validation_builder.add_edge(START, "auditar_query")
validation_builder.add_edge("validar_seguridad", "combinar")
validation_builder.add_edge("auditar_query", "combinar")
validation_builder.add_edge("combinar", END)

# Compilamos el subgrafo
subgraph_validation = validation_builder.compile()

# =====================================================================
# 5. CONSTRUCCIÓN DEL GRAFO PRINCIPAL
# =====================================================================
workflow = StateGraph(AgentState)

# Añadimos los nodos principales y el subgrafo modular
workflow.add_node("generar_query", generar_query_node)
workflow.add_node("subgrafo_validacion", subgraph_validation)
workflow.add_node("ejecutar_db", ejecutar_db_node)
workflow.add_node("rechazo", generar_rechazo_node)
workflow.add_node("preparar_map", preparar_map_node)
workflow.add_node("procesar_lote", procesar_lote_node)
workflow.add_node("generar_respuesta", generar_respuesta_node)

# Conexiones en el grafo principal
workflow.add_edge(START, "generar_query")
workflow.add_edge("generar_query", "subgrafo_validacion")

# Decisión condicional tras la finalización del subgrafo
workflow.add_conditional_edges(
    "subgrafo_validacion",
    decidir_ruta,
    {
        "segura": "ejecutar_db",
        "insegura": "rechazo"
    }
)

# Ambos caminos de salida (BD o rechazo) entran a la preparación de Map-Reduce
workflow.add_edge("ejecutar_db", "preparar_map")
workflow.add_edge("rechazo", "preparar_map")

# Mapeo dinámico y condicional desde preparar_map
workflow.add_conditional_edges(
    "preparar_map",
    distribuir_o_saltar,
    {
        "procesar_lote": "procesar_lote",
        "directo": "generar_respuesta"
    }
)

# La salida de los mapeados va al reduce final
workflow.add_edge("procesar_lote", "generar_respuesta")

workflow.add_edge("generar_respuesta", END)

# Compilamos el grafo con un checkpointer en memoria y configuramos la interrupción antes de ejecutar en base de datos
memory = MemorySaver()
app = workflow.compile(checkpointer=memory, interrupt_before=["ejecutar_db"])

# =====================================================================
# 5. MODO INTERACTIVO (Chat)
# =====================================================================
if __name__ == "__main__":
    print("👋 ¡Hola! Soy tu Agente basado en un Grafo Personalizado (LangGraph) con Human-in-the-Loop.")
    print("Flujo: Generar Query -> Validar -> HITL -> Ejecutar DB o Rechazar -> Responder.")
    print("Escribe 'salir' para terminar la conversación.")
    print("=" * 60)

    while True:
        pregunta = input("\n👤 Tú: ")
        
        if pregunta.lower() in ["salir", "exit", "quit", "q"]:
            print("¡Hasta luego! 👋")
            break
            
        if pregunta.strip():
            # Generamos un thread_id único para esta pregunta
            thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}
            initial_state = {"pregunta_usuario": pregunta}
            
            print("-" * 40)
            try:
                # Se inicia la ejecución del grafo hasta que termine o se interrumpa
                resultado = app.invoke(initial_state, config=config)
                
                # Bucle para manejar interrupciones (HITL)
                while True:
                    state_info = app.get_state(config)
                    
                    # Si no hay más nodos pendientes de ejecución, hemos terminado
                    if not state_info.next:
                        break
                        
                    # Si la interrupción ocurre antes de ejecutar la consulta en la base de datos
                    if "ejecutar_db" in state_info.next:
                        sql_query = state_info.values.get("sql_query")
                        print(f"\n🔍 [HITL] El agente ha generado la siguiente consulta SQL:")
                        print(f"   👉 {sql_query}")
                        print("\n¿Qué deseas hacer?")
                        print("1. Aprobar y ejecutar (Pulsa Enter o escribe 's')")
                        print("2. Modificar la consulta (Escribe la nueva consulta SQL)")
                        print("3. Rechazar consulta (Escribe 'n' o 'rechazar')")
                        
                        opcion = input("\n✍️ HITL > ").strip()
                        
                        if opcion.lower() in ["n", "rechazar", "no", "cancelar"]:
                            print("⚠️ Consulta rechazada por el usuario. Cancelando ejecución...")
                            # Simulamos el resultado de la base de datos indicando la cancelación
                            app.update_state(
                                config,
                                {"db_result": "Consulta cancelada por el usuario."},
                                as_node="ejecutar_db"
                            )
                            resultado = app.invoke(None, config=config)
                        elif opcion == "" or opcion.lower() in ["s", "si", "sí", "y", "yes", "aprobar"]:
                            print("✅ Consulta aprobada. Ejecutando...")
                            resultado = app.invoke(None, config=config)
                        else:
                            # Se asume que el usuario introdujo una consulta SQL editada
                            print(f"✏️ Modificando la consulta a: {opcion}")
                            app.update_state(
                                config,
                                {"sql_query": opcion},
                                as_node="generar_query"
                            )
                            print("🚀 Ejecutando consulta modificada...")
                            resultado = app.invoke(None, config=config)
                    else:
                        # Si hay otra interrupción imprevista, simplemente reanudamos
                        resultado = app.invoke(None, config=config)
                
                print("-" * 40)
                print(f"\n🤖 Agente Final:\n{resultado.get('respuesta_final')}")
                print("=" * 60)
            except Exception as e:
                print(f"\n🛡️ [Error en el grafo] {str(e)}")
                print("=" * 60)
