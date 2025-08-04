import streamlit as st
import os
from dotenv import load_dotenv
import re
from typing import Dict, List, Optional
from dataclasses import dataclass
import openai
from openai import OpenAI

# Cargar variables de entorno
load_dotenv()

# Configuración de la página
st.set_page_config(
    page_title="Prompt Optimizer - MIA Chatbot",
    page_icon="🔧",
    layout="wide"
)

# Tipos de prompts disponibles
PROMPT_TYPES = {
    "role_and_goal": "🎯 Rol y Objetivo",
    "tone_style_and_response_format": "💬 Tono, Estilo y Formato", 
    "conversation_flow": "🔄 Flujo Conversacional",
    "examples_interaction": "💡 Ejemplos de Interacción",
    "restrictions": "🚫 Restricciones"
}

@dataclass
class OptimizationResult:
    original_prompt: str
    optimized_prompt: str
    context_description: Optional[str]
    prompt_type: str
    has_context: bool
    optimization_mode: str
    context_analysis: str
    optimizations_applied: List[str]
    metaprompt_alignment: List[str]
    explanation: str
    best_practices_applied: List[str]
    compatibility_score: int

class PromptOptimizer:
    def __init__(self):
        # Obtener API key del archivo .env
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY no encontrada en el archivo .env")
        
        self.client = OpenAI(api_key=self.api_key)
    
    def optimize_prompt(self, user_prompt: str, prompt_type: str, context_description: str = "") -> OptimizationResult:
        """Optimiza un prompt usando la lógica del sistema."""
        
        has_context = bool(context_description.strip())
        
        optimization_prompt = self._create_optimization_template(
            user_prompt, context_description, prompt_type, has_context
        )
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": optimization_prompt}],
                temperature=0.1,
                max_tokens=2500
            )
            
            parsed_result = self._parse_response(response.choices[0].message.content, has_context)
            compatibility_score = self._calculate_mia_compatibility_score(
                user_prompt, parsed_result.get("optimized_prompt", ""), prompt_type
            )
            
            return OptimizationResult(
                original_prompt=user_prompt,
                optimized_prompt=parsed_result.get("optimized_prompt", ""),
                context_description=context_description if has_context else None,
                prompt_type=prompt_type,
                has_context=has_context,
                optimization_mode="contextualized" if has_context else "generic",
                context_analysis=parsed_result.get("context_analysis", ""),
                optimizations_applied=parsed_result.get("optimizations_applied", []),
                metaprompt_alignment=parsed_result.get("metaprompt_alignment", []),
                explanation=parsed_result.get("explanation", ""),
                best_practices_applied=parsed_result.get("best_practices_applied", []),
                compatibility_score=compatibility_score
            )
            
        except Exception as e:
            st.error(f"Error optimizando prompt: {str(e)}")
            return None
    
    def _create_optimization_template(self, user_prompt: str, context_description: str, 
                                    prompt_type: str, has_context: bool) -> str:
        """Template ajustado para mejora preservativa del prompt original."""
        
        # CONTEXTO ESPECÍFICO DE MIA - SIEMPRE INCLUIDO
        mia_system_context = """
SISTEMA MIA - CONTEXTO TÉCNICO:
MIA es un chatbot que debe poder:
- Consultar catálogos de productos/servicios cuando sea relevante
- Acceder a guías empresariales para información oficial  
- Detectar intenciones de compra automáticamente
- Escalar a agentes humanos cuando sea necesario
- Mantener conversaciones naturales y orientadas a resultados

RESTRICCIONES TÉCNICAS CRÍTICAS:
❌ No puede inventar información que no esté en bases de conocimiento
❌ No debe usar nombres técnicos de herramientas en respuestas al cliente
❌ No debe contradecir el flujo automático de escalamiento del sistema
"""

        # FILOSOFÍA DE OPTIMIZACIÓN PRESERVATIVA
        optimization_philosophy = """
FILOSOFÍA DE OPTIMIZACIÓN - PRESERVATIVA:
🎯 PRESERVAR: Mantén TODA la personalidad, tono, estilo y elementos específicos del prompt original
🔧 MEJORAR: Solo ajusta aspectos que causen conflictos técnicos reales con MIA
➕ AGREGAR: Funcionalidades de MIA SOLO cuando complementen (no reemplacen) el contenido original
🚫 ELIMINAR: Solo patrones que causen conflictos críticos comprobados

PRINCIPIO CLAVE: "MEJORA ADITIVA, NO SUSTITUTIVA"
- Si el usuario quiere que sea "amigable con emojis" → MANTENERLO
- Si el usuario dice "agente humano" → MANTENERLO (no cambiar a "especialista")
- Si el usuario define un tono específico → RESPETARLO completamente
- Si el usuario especifica comportamientos → PRESERVARLOS
"""

        # GUÍAS ESPECÍFICAS POR TIPO (AJUSTADAS PARA SER PRESERVATIVAS)
        type_specific_guidance = {
            "role_and_goal": """
🎯 OPTIMIZACIONES PRESERVATIVAS PARA ROL Y OBJETIVO:

ELEMENTOS A PRESERVAR OBLIGATORIAMENTE:
✅ Nombre del asistente (si está especificado)
✅ Personalidad y tono definidos por el usuario
✅ Objetivos específicos mencionados
✅ Estilo de comunicación (formal, amigable, etc.)
✅ Terminología específica del usuario

MEJORAS ADITIVAS (solo agregar si no existe):
+ Consulta de información cuando sea relevante para el negocio
+ Escalamiento cuando no tenga información (usando términos del usuario)
+ Referencia a capacidades específicas del negocio

CONFLICTOS A RESOLVER (cambiar solo si existe):
❌ "siempre responde aunque no sepas" → "consulta información disponible o [escalamiento según usuario]"
❌ "nunca digas que no sabes" → "busca la información necesaria o [escalamiento según usuario]"
❌ "inventa información" → "usa información oficial disponible"

EJEMPLO DE OPTIMIZACIÓN PRESERVATIVA:
ORIGINAL: "Eres Ana, asistente amigable 😊 que siempre ayuda, aunque no sepas responde algo"
OPTIMIZADO: "Eres Ana, asistente amigable 😊 que siempre ayuda consultando la información disponible, y cuando no encuentres algo específico, conecta con quien pueda ayudar mejor"
""",

            "tone_style_and_response_format": """
💬 OPTIMIZACIONES PRESERVATIVAS PARA TONO Y ESTILO:

ELEMENTOS A PRESERVAR OBLIGATORIAMENTE:
✅ Tono específico definido (formal, casual, amigable, profesional, etc.)
✅ Uso de emojis (si el usuario los especifica)
✅ Estilo de comunicación definido
✅ Longitud de respuestas preferida
✅ Formato específico solicitado

MEJORAS ADITIVAS (solo si complementan):
+ Estructura de respuestas cuando no esté definida
+ Adaptabilidad contextual si no contradice el estilo original
+ Transiciones naturales hacia información relevante

CONFLICTOS A RESOLVER (solo cambios mínimos):
❌ Indicaciones de responder sin información → Ajustar para consultar primero
❌ Tono inconsistente con capacidades del sistema → Ajustar mínimamente

EJEMPLO DE OPTIMIZACIÓN PRESERVATIVA:
ORIGINAL: "Usa muchos emojis 🎉😊 y sé muy casual y divertido, siempre responde algo"
OPTIMIZADO: "Usa muchos emojis 🎉😊 y sé muy casual y divertido, consultando la info disponible para dar respuestas útiles"
""",

            "conversation_flow": """
🔄 OPTIMIZACIONES PRESERVATIVAS PARA FLUJO CONVERSACIONAL:

ELEMENTOS A PRESERVAR OBLIGATORIAMENTE:
✅ Pasos específicos definidos por el usuario
✅ Condiciones y reglas establecidas
✅ Puntos de escalamiento definidos (manteniendo terminología)
✅ Flujo lógico diseñado por el usuario

MEJORAS ADITIVAS (complementar, no reemplazar):
+ Consultas específicas cuando sean relevantes para pasos existentes
+ Detección de intenciones en puntos apropiados del flujo original
+ Escalamiento mejorado usando términos del usuario

CONFLICTOS A RESOLVER (ajustes mínimos):
❌ Pasos que requieren inventar información → Agregar consulta de fuentes
❌ Escalamiento indefinido → Clarificar usando términos del usuario

EJEMPLO DE OPTIMIZACIÓN PRESERVATIVA:
ORIGINAL: "1. Saluda, 2. Pregunta qué necesita, 3. Responde lo que sepas, 4. Si no sabes, deriva al equipo"
OPTIMIZADO: "1. Saluda, 2. Pregunta qué necesita, 3. Consulta información disponible para responder, 4. Si no encuentras información específica, deriva al equipo"
""",

            "examples_interaction": """
💡 OPTIMIZACIONES PRESERVATIVAS PARA EJEMPLOS DE INTERACCIÓN:

ELEMENTOS A PRESERVAR OBLIGATORIAMENTE:
✅ Ejemplos específicos proporcionados por el usuario
✅ Estilo de interacción mostrado en ejemplos
✅ Tipos de situaciones ejemplificadas
✅ Respuestas modelo del usuario

MEJORAS ADITIVAS (enriquecer ejemplos existentes):
+ Agregar consulta de información en ejemplos donde sea relevante
+ Mejorar ejemplos de escalamiento manteniendo el enfoque original
+ Complementar ejemplos con mejores prácticas

CONFLICTOS A RESOLVER (ajustes en ejemplos):
❌ Ejemplos que muestran inventar información → Mostrar consulta de fuentes
❌ Escalamiento vago → Clarificar manteniendo estilo original

EJEMPLO DE OPTIMIZACIÓN PRESERVATIVA:
ORIGINAL: "Cliente: ¿Precio de X? Ana: Cuesta $100 (aunque no lo sepas)"
OPTIMIZADO: "Cliente: ¿Precio de X? Ana: Déjame consultar el precio actual de X... Cuesta $100"
""",

            "restrictions": """
🚫 OPTIMIZACIONES PRESERVATIVAS PARA RESTRICCIONES:

ELEMENTOS A PRESERVAR OBLIGATORIAMENTE:
✅ Restricciones específicas del usuario
✅ Límites definidos por el negocio
✅ Comportamientos prohibidos según el usuario
✅ Políticas específicas mencionadas

MEJORAS ADITIVAS (agregar solo si es necesario):
+ Restricciones técnicas críticas para el funcionamiento
+ Clarificaciones para evitar conflictos con el sistema

CONFLICTOS A RESOLVER (solo cambios críticos):
❌ "Nunca digas que no sabes" → "Consulta información disponible antes de escalar"
❌ "Inventa si no sabes" → "Usa solo información oficial disponible"

EJEMPLO DE OPTIMIZACIÓN PRESERVATIVA:
ORIGINAL: "No hables de política, religión, y nunca digas que no sabes algo"
OPTIMIZADO: "No hables de política, religión, y consulta información disponible o conecta con quien pueda ayudar cuando no tengas datos específicos"
"""
        }

        specific_guidance = type_specific_guidance.get(prompt_type, type_specific_guidance["role_and_goal"])
        
        context_section = ""
        if has_context and context_description.strip():
            context_section = f"""
CONTEXTO ESPECÍFICO DEL NEGOCIO:
"{context_description}"

MODO: CONTEXTUALIZADA - Personaliza PRESERVANDO el prompt original para este negocio específico.
"""
        else:
            context_section = """
CONTEXTO DEL NEGOCIO: No se proporcionó contexto específico

MODO: GENÉRICA - Mejora PRESERVANDO el prompt original para que sea compatible con MIA.
"""

        return f"""
Eres un experto en optimización de prompts que mejora prompts PRESERVANDO completamente la intención, 
personalidad y características específicas del usuario.

{mia_system_context}

{optimization_philosophy}

TIPO DE PROMPT A OPTIMIZAR: {prompt_type.upper().replace('_', ' ')}

{specific_guidance}

{context_section}

PROMPT ORIGINAL DEL USUARIO (PRESERVAR SU ESENCIA):
"{user_prompt}"

METODOLOGÍA DE OPTIMIZACIÓN:
1. 🔍 ANALIZAR: Identificar qué elementos preservar vs qué problemas técnicos resolver
2. 🎯 PRESERVAR: Mantener personalidad, tono, terminología y elementos específicos
3. 🔧 MEJORAR: Solo ajustar aspectos que causen conflictos técnicos reales
4. ➕ COMPLEMENTAR: Agregar capacidades de MIA que enriquezcan (no reemplacen)

TRANSFORMACIONES PERMITIDAS (SOLO SI CAUSAN CONFLICTOS TÉCNICOS):
✅ "siempre responde aunque no sepas" → "consulta información disponible o [usar términos del usuario para escalamiento]"
✅ "nunca digas que no sabes" → "busca en las fuentes disponibles o [escalamiento según usuario]"
✅ "inventa información" → "usa información oficial disponible"

TRANSFORMACIONES PROHIBIDAS:
❌ Cambiar personalidad definida (amigable, formal, etc.)
❌ Eliminar emojis si el usuario los especifica
❌ Cambiar terminología específica del usuario ("agente humano" ≠ "especialista")
❌ Modificar tono o estilo sin razón técnica crítica
❌ Reemplazar contenido que funciona bien

EJEMPLOS DE OPTIMIZACIÓN PRESERVATIVA:

EJEMPLO 1:
ORIGINAL: "Eres Ana, súper amigable 😊, usa muchos emojis y siempre responde aunque no sepas"
OPTIMIZADO: "Eres Ana, súper amigable 😊, usa muchos emojis y consulta la información disponible para dar las mejores respuestas, conectando con el equipo cuando necesites ayuda adicional"

EJEMPLO 2:
ORIGINAL: "Sé formal y profesional. Si no tienes información, deriva al departamento de ventas"
OPTIMIZADO: "Sé formal y profesional. Consulta la información disponible y cuando no tengas datos específicos, deriva al departamento de ventas"

FORMATO DE RESPUESTA OBLIGATORIO:

ANÁLISIS DEL PROMPT ORIGINAL:
[Identificar elementos a preservar vs problemas técnicos a resolver]

OPTIMIZACIONES APLICADAS:
[Lista específica de cambios MÍNIMOS realizados y justificación técnica]

PROMPT OPTIMIZADO:
[Versión mejorada que PRESERVA la esencia original y resuelve conflictos técnicos]

MEJORAS IMPLEMENTADAS:
[Explicación de cómo se preservó el original mientras se mejoró la compatibilidad]

COMPATIBILIDAD CON MIA:
[Confirmación de que funciona con MIA sin perder la personalidad original]
"""
    
    def _parse_response(self, response_content: str, has_context: bool) -> Dict:
        """Parsea la respuesta del modelo, extrayendo limpiamente el prompt optimizado."""
        try:
            sections = {}
            
            # Análisis
            analysis_pattern = r'ANÁLISIS DEL PROMPT ORIGINAL:\s*(.*?)(?=\nOPTIMIZACIONES APLICADAS:|$)'
            analysis_match = re.search(analysis_pattern, response_content, re.DOTALL)
            sections["context_analysis"] = analysis_match.group(1).strip() if analysis_match else ""
            
            # Optimizaciones aplicadas
            optimizations_match = re.search(r'OPTIMIZACIONES APLICADAS:\s*(.*?)(?=\nPROMPT OPTIMIZADO:|$)', response_content, re.DOTALL)
            optimizations_text = optimizations_match.group(1).strip() if optimizations_match else ""
            sections["optimizations_applied"] = self._parse_list_items(optimizations_text)
            
            # Prompt optimizado (LIMPIO)
            optimized_match = re.search(r'PROMPT OPTIMIZADO:\s*(.*?)(?=\nMEJORAS IMPLEMENTADAS:|$)', response_content, re.DOTALL)
            if optimized_match:
                optimized_text = optimized_match.group(1).strip()
                # Limpiar cualquier texto explicativo
                lines = optimized_text.split('\n')
                clean_prompt_lines = []
                
                for line in lines:
                    # Parar si encontramos texto explicativo
                    if any(keyword in line.lower() for keyword in ['mejoras', 'explicación', 'este prompt', 'los cambios', 'implementadas']):
                        break
                    clean_prompt_lines.append(line)
                
                sections["optimized_prompt"] = '\n'.join(clean_prompt_lines).strip()
            else:
                sections["optimized_prompt"] = "Error: No se pudo extraer el prompt optimizado"
            
            # Mejoras implementadas
            explanation_match = re.search(r'MEJORAS IMPLEMENTADAS:\s*(.*?)(?=\nCOMPATIBILIDAD CON MIA:|$)', response_content, re.DOTALL)
            sections["explanation"] = explanation_match.group(1).strip() if explanation_match else ""
            
            # Compatibilidad con MIA
            compatibility_match = re.search(r'COMPATIBILIDAD CON MIA:\s*(.*?)$', response_content, re.DOTALL)
            compatibility_text = compatibility_match.group(1).strip() if compatibility_match else ""
            sections["metaprompt_alignment"] = self._parse_list_items(compatibility_text)
            
            # Best practices (para mantener compatibilidad)
            sections["best_practices_applied"] = self._extract_best_practices(sections["optimizations_applied"])
            
            return sections
            
        except Exception as e:
            return {
                "context_analysis": f"Error al procesar respuesta: {str(e)}",
                "optimizations_applied": [],
                "optimized_prompt": "Error al extraer el prompt optimizado. Respuesta completa del modelo:\n\n" + response_content,
                "explanation": f"Error en el procesamiento: {str(e)}",
                "metaprompt_alignment": [],
                "best_practices_applied": []
            }
    
    def _parse_list_items(self, text: str) -> List[str]:
        """Convierte texto con bullets a lista."""
        items = []
        if text:
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if line and (line.startswith('✅') or line.startswith('•') or line.startswith('-') or line.startswith('*')):
                    # Limpiar emojis y bullets
                    clean_line = re.sub(r'^[✅•\-*]\s*', '', line).strip()
                    if clean_line:
                        items.append(clean_line)
                elif line and not line.startswith('[') and len(line) > 15:
                    items.append(line)
        return items
    
    def _extract_best_practices(self, optimizations: List[str]) -> List[str]:
        """Extrae mejores prácticas de las optimizaciones aplicadas."""
        practices = []
        practice_keywords = {
            "preserv": "Preservación de elementos originales",
            "consulta": "Uso de consultas a fuentes oficiales",
            "escalamiento": "Escalamiento apropiado",
            "personalidad": "Mantenimiento de personalidad",
            "tono": "Preservación del tono original",
            "específico": "Personalización para el negocio",
            "compatibilidad": "Compatibilidad con sistema MIA"
        }
        
        for opt in optimizations:
            for keyword, practice in practice_keywords.items():
                if keyword in opt.lower() and practice not in practices:
                    practices.append(practice)
        
        return practices[:5]  # Máximo 5 prácticas
    
    def _calculate_mia_compatibility_score(self, original_prompt: str, optimized_prompt: str, prompt_type: str) -> int:
        """Sistema de validación específico para MIA con enfoque preservativo."""
        
        # Patrones críticos específicos de MIA que DEBEN estar
        critical_good_patterns = {
            "all_types": [
                "consulta", "información disponible", "fuentes", "conecta",
                "deriva", "equipo", "agente", "humano"
            ],
            "conversation_flow": [
                "si preguntan", "para información", "cuando no encuentres",
                "detecta", "facilita", "pasos"
            ],
            "role_and_goal": [
                "asistente", "ayuda", "consulta", "deriva cuando",
                "información", "disponible"
            ]
        }
        
        # Patrones que NUNCA deben aparecer
        critical_bad_patterns = [
            "inventa", "supón", "asume", "nunca digas que no sabes", 
            "siempre responde aunque no sepas", "retrieve_products", 
            "transfer_to_human"
        ]
        
        # NUEVO: Patrones de preservación (dan puntos por mantener elementos originales)
        preservation_patterns = [
            "amigable", "emojis", "😊", "🎉", "formal", "profesional",
            "casual", "divertido", "ana", "asistente"
        ]

        def count_patterns(text, patterns_list):
            text_lower = text.lower()
            return sum(1 for pattern in patterns_list if pattern in text_lower)

        def count_weighted_patterns(text, patterns_dict, prompt_type):
            text_lower = text.lower()
            score = 0
            
            # Patrones generales (peso 1)
            for pattern in patterns_dict.get("all_types", []):
                if pattern in text_lower:
                    score += 1
            
            # Patrones específicos del tipo (peso 2)
            for pattern in patterns_dict.get(prompt_type, []):
                if pattern in text_lower:
                    score += 2
                    
            return score

        # Calcular scores
        optimized_good = count_weighted_patterns(optimized_prompt, critical_good_patterns, prompt_type)
        optimized_bad = count_patterns(optimized_prompt, critical_bad_patterns)
        
        # NUEVO: Score de preservación (compara elementos originales mantenidos)
        original_elements = count_patterns(original_prompt, preservation_patterns)
        preserved_elements = count_patterns(optimized_prompt, preservation_patterns)
        preservation_score = min(preserved_elements, original_elements) if original_elements > 0 else 0
        
        # Calcular score final (0-10)
        base_score = min(optimized_good, 5)                    # Máximo 5 por patrones buenos
        preservation_bonus = min(preservation_score, 3)        # Máximo 3 por preservación
        penalty = min(optimized_bad * 3, 6)                   # Máximo 6 de penalización
        
        final_score = max(0, min(10, base_score + preservation_bonus - penalty + 2))  # +2 base
        
        return int(final_score)

def check_environment():
    """Verifica que el entorno esté configurado correctamente."""
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        st.error("**API Key no encontrada**")
        st.markdown("""
        **Necesitas configurar tu API Key de OpenAI en el archivo `.env`:**
        
        1. Crea un archivo `.env` en la misma carpeta que esta aplicación
        2. Agrega la línea: `OPENAI_API_KEY=tu_api_key_aquí`
        3. Reinicia la aplicación
        """)
        return False
    
    # Verificar que la API key tenga el formato correcto
    if not api_key.startswith('sk-'):
        st.warning("**Formato de API Key inválido**")
        st.markdown("La API Key de OpenAI debe comenzar con 'sk-'")
        return False
    
    return True

def main():
    st.title("Prompt Optimizer - MIA Chatbot")
    st.markdown("---")
    
    # Verificar configuración del entorno
    if not check_environment():
        st.stop()
    
    # Área principal - sin sidebar
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Entrada")
        
        # Formulario de entrada
        with st.form("prompt_optimizer_form"):
            # Prompt original
            user_prompt = st.text_area(
                "Prompt a Optimizar:",
                height=150,
                placeholder="Ejemplo: Eres Ana, súper amigable 😊 que usa emojis. Si no sabes algo, deriva al agente humano...",
                help="Escribe el prompt que quieres optimizar para el sistema MIA"
            )
            
            # Tipo de prompt
            prompt_type = st.selectbox(
                "Tipo de Prompt:",
                options=list(PROMPT_TYPES.keys()),
                format_func=lambda x: PROMPT_TYPES[x],
                help="Selecciona qué tipo de prompt estás optimizando"
            )
            
            # Contexto opcional
            st.markdown("**Contexto del Negocio (Opcional)**")
            context_description = st.text_area(
                "Describe tu negocio o contexto específico:",
                height=100,
                placeholder="Ejemplo: Tienda de ropa deportiva. Vendemos zapatillas Nike, Adidas. Atención presencial y online. Horarios 9-18h...",
                help="Esta información es opcional pero permite crear optimizaciones más específicas para tu negocio"
            )
            
            # Botón de optimización
            submitted = st.form_submit_button(
                "Optimizar Prompt",
                type="primary",
                use_container_width=True
            )
    
    with col2:
        st.header("Resultado")
        
        if submitted:
            if not user_prompt.strip():
                st.error("Por favor ingresa un prompt para optimizar")
            else:
                with st.spinner("Optimizando prompt..."):
                    try:
                        optimizer = PromptOptimizer()
                        result = optimizer.optimize_prompt(user_prompt, prompt_type, context_description)
                    except ValueError as e:
                        st.error(f"Error de configuración: {str(e)}")
                        result = None
                    except Exception as e:
                        st.error(f"Error inesperado: {str(e)}")
                        result = None
                
                if result:
                    # Mostrar resultados
                    display_results(result)

def display_results(result: OptimizationResult):
    """Muestra los resultados de la optimización."""
    
    # PROMPT OPTIMIZADO EN CAJA PRINCIPAL
    with st.container():
        st.text_area(
            "Prompt Optimizado:",
            value=result.optimized_prompt,
            height=200,
            disabled=True,
            label_visibility="collapsed"
        )
    
    # CAJA PARA COPIAR
    st.markdown("**Para copiar:**")
    st.code(result.optimized_prompt, language="text")
    
    # Tabs con detalles
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Análisis", "🛠️ Optimizaciones", "⚡ Compatibilidad", "🎯 Preservación"])
    
    with tab1:
        st.markdown("**Análisis del Prompt Original**")
        if result.context_analysis:
            st.markdown(result.context_analysis)
        
        if result.explanation:
            st.markdown("**Mejoras Implementadas**")
            st.markdown(result.explanation)
    
    with tab2:
        st.markdown("**Optimizaciones Aplicadas**")
        if result.optimizations_applied:
            for i, opt in enumerate(result.optimizations_applied, 1):
                st.markdown(f"{i}. {opt}")
        else:
            st.info("No se aplicaron optimizaciones - tu prompt ya era compatible")
    
    with tab3:
        st.markdown("**Compatibilidad con Sistema MIA**")
        if result.metaprompt_alignment:
            for alignment in result.metaprompt_alignment:
                st.markdown(f"✅ {alignment}")
        else:
            st.info("Compatible sin cambios adicionales necesarios")
    
    with tab4:
        st.markdown("**Elementos Preservados**")
        if result.best_practices_applied:
            for practice in result.best_practices_applied:
                st.markdown(f"🎯 {practice}")
        else:
            st.info("Prompt optimizado técnicamente sin cambios de estilo")
    
    # Comparación lado a lado al final
    st.markdown("---")
    st.subheader("Comparación")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Prompt Original:**")
        st.text_area(
            "Original",
            value=result.original_prompt,
            height=150,
            disabled=True,
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("**Prompt Optimizado:**")
        st.text_area(
            "Optimizado",
            value=result.optimized_prompt,
            height=150,
            disabled=True,
            label_visibility="collapsed"
        )

if __name__ == "__main__":
    main()