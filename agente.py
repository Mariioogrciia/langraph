import os
import sqlite3
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

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
# 1. DEFINICIÓN DE HERRAMIENTAS (@tool)
# =====================================================================
@tool
def ejecutar_sql(query: str) -> str:
    """Ejecuta una consulta SQL SELECT en la base de datos de productos.
    La tabla se llama 'productos' y tiene las columnas (id, nombre, precio, stock, caracteristicas).
    Devuelve los resultados en formato texto.
    """
    print(f"🛠️ [Herramienta] Ejecutando SQL: {query}")
    query_upper = query.upper()
    
    # Lista de comandos SQL prohibidos
    palabras_peligrosas = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE", "TRUNCATE"]
    
    if any(palabra in query_upper for palabra in palabras_peligrosas):
        print("⚠️ ALERTA: Intento de modificación detectado.")
        return "ACCESO DENEGADO: Solo se permiten consultas SELECT por motivos de seguridad."
        
    try:
        cursor.execute(query)
        resultados = cursor.fetchall()
        return str(resultados) if resultados else "No se encontraron resultados."
    except Exception as e:
        return f"Error técnico al ejecutar SQL: {str(e)}"

# =====================================================================
# 2. CONFIGURACIÓN DEL AGENTE REACT
# =====================================================================
prompt_sistema = """Eres un asistente virtual experto en inventario de productos.
Tienes acceso a una base de datos SQLite con una tabla llamada 'productos' a través de la herramienta 'ejecutar_sql'.
El esquema de la tabla es: (id INTEGER, nombre TEXT, precio REAL, stock INTEGER, caracteristicas TEXT).
Usa la herramienta para buscar información concreta cuando el usuario pregunte sobre productos, precios, stock o características.
Si la herramienta deniega el acceso, explica al usuario que la operación no está permitida por seguridad."""

# Creamos el agente pasándole el modelo, las herramientas y el comportamiento inicial
app = create_react_agent(
    model,
    tools=[ejecutar_sql],
    prompt=prompt_sistema
)

# =====================================================================
# 3. MODO INTERACTIVO (Chat)
# =====================================================================
if __name__ == "__main__":
    print("👋 ¡Hola! Soy tu Agente ReAct seguro.")
    print("Puedo usar herramientas para consultar la base de datos de productos.")
    print("Escribe 'salir' para terminar la conversación.")
    print("=" * 60)

    # El agente ReAct requiere mantener el historial de mensajes
    historial = []

    while True:
        pregunta_usuario = input("\n👤 Tú: ")
        
        if pregunta_usuario.lower() in ["salir", "exit", "quit", "q"]:
            print("¡Hasta luego! 👋")
            break
            
        if pregunta_usuario.strip():
            # Añadimos el mensaje del usuario al historial
            historial.append(HumanMessage(content=pregunta_usuario))
            
            try:
                # Invocamos el agente con el historial completo
                resultado = app.invoke({"messages": historial})
                
                # Obtenemos la última respuesta generada por el agente
                ultimo_mensaje = resultado["messages"][-1]
                respuesta_agente = ultimo_mensaje.content
                
                print(f"\n🤖 Agente:\n{respuesta_agente}")
                print("-" * 60)
                
                # Actualizamos nuestro historial local para la siguiente iteración
                historial = resultado["messages"]
            except Exception as e:
                print(f"\n🛡️ [Error o Filtro] {str(e)}")
                print("-" * 60)
