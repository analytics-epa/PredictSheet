import streamlit as st
import polars as pl
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from google.oauth2 import service_account

# Configuración de la página
st.set_page_config(page_title="Análisis de R²", layout="wide")
st.title("Reporte de Análisis R²")

# Función para conectar a Google Sheets
@st.cache_data
def cargar_datos_google_sheets(sheet_url):
    try:
        # Configuración para conectar a Google Sheets
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        
        # Necesitarás subir el archivo JSON de credenciales a Streamlit
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"], 
            scopes=scope
        )
        
        gc = gspread.authorize(credentials)
        
        # Extraer el ID del sheet de la URL
        if "spreadsheets/d/" in sheet_url:
            sheet_id = sheet_url.split("spreadsheets/d/")[1].split("/")[0]
            worksheet = gc.open_by_key(sheet_id).sheet1
        else:
            st.error("URL de Google Sheets no válida")
            return None
            
        # Obtener todos los valores
        data = worksheet.get_all_records()
        df = pl.DataFrame(data)
        return df
        
    except Exception as e:
        st.error(f"Error al cargar datos: {e}")
        return None

# Entrada para la URL de Google Sheets
sheet_url = st.text_input("Ingresa la URL de tu Google Sheet:")

if sheet_url:
    df = cargar_datos_google_sheets(sheet_url)
    
    if df is not None:
        st.success("Datos cargados correctamente!")
        
        # Mostrar los datos
        st.subheader("Vista previa de los datos")
        st.write(df.head().to_pandas())
        
        # Información básica
        st.subheader("Información básica")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Número de filas:** {df.height}")
            st.write(f"**Número de columnas:** {df.width}")
        with col2:
            st.write("**Tipos de datos:**")
            st.write(dict(zip(df.columns, df.dtypes)))
        
        # Selección de variables
        st.subheader("Selección de variables para análisis R²")
        
        # Convertir columnas adecuadas a numéricas
        numeric_cols = []
        for col in df.columns:
            try:
                df = df.with_columns(pl.col(col).cast(pl.Float64, strict=False))
                numeric_cols.append(col)
            except Exception:
                pass
        
        if len(numeric_cols) < 2:
            st.warning("Se necesitan al menos 2 columnas numéricas para el análisis R².")
        else:
            # Variables independientes (X)
            st.write("**Selecciona variables independientes (X):**")
            x_vars = st.multiselect("Variables X", numeric_cols)
            
            # Variable dependiente (Y)
            remaining_cols = [col for col in numeric_cols if col not in x_vars]
            st.write("**Selecciona variable dependiente (Y):**")
            y_var = st.selectbox("Variable Y", remaining_cols if remaining_cols else ["Ninguna"])
            
            if x_vars and y_var != "Ninguna":
                # Preparar datos para regresión
                X = df.select(x_vars).to_numpy()
                y = df[y_var].to_numpy()
                
                # Crear y entrenar modelo de regresión lineal
                model = LinearRegression()
                model.fit(X, y)
                
                # Realizar predicciones
                y_pred = model.predict(X)
                
                # Calcular R²
                r2 = r2_score(y, y_pred)
                
                # Mostrar resultados
                st.subheader("Resultados del Análisis")
                
                # Métrica de R²
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric(label="Coeficiente R²", value=f"{r2:.4f}")
                    if r2 < 0.3:
                        st.warning("R² bajo: relación débil entre variables")
                    elif r2 < 0.7:
                        st.info("R² moderado: relación media entre variables")
                    else:
                        st.success("R² alto: fuerte relación entre variables")
                
                # Coeficientes
                with col2:
                    st.write("**Coeficientes del modelo:**")
                    coef_df = pl.DataFrame({'Variable': x_vars, 'Coeficiente': model.coef_})
                    st.write(coef_df.to_pandas())
                    st.write(f"**Intercepto:** {model.intercept_:.4f}")
                
                # Gráficas
                st.subheader("Visualizaciones")
                
                # Gráfica real vs predicho
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=y, y=y_pred, mode="markers", name="Datos", opacity=0.5))
                fig.add_trace(
                    go.Scatter(
                        x=[y.min(), y.max()],
                        y=[y.min(), y.max()],
                        mode="lines",
                        line=dict(color="red", dash="dash"),
                        name="Ideal",
                    )
                )
                fig.update_layout(
                    xaxis_title=f"Valor real de {y_var}",
                    yaxis_title="Valor predicho",
                    title="Valores reales vs. predichos",
                )
                st.plotly_chart(fig)

                # Si hay una sola variable independiente, mostrar gráfica de dispersión con línea de regresión
                if len(x_vars) == 1:
                    fig2 = px.scatter(
                        df.select([x_vars[0], y_var]).to_pandas(),
                        x=x_vars[0],
                        y=y_var,
                        trendline="ols",
                        title=f'Regresión: {x_vars[0]} vs {y_var} (R² = {r2:.4f})',
                    )
                    st.plotly_chart(fig2)
                
                # Resumen estadístico
                st.subheader("Resumen estadístico")
                st.write(df.select(x_vars + [y_var]).describe().to_pandas())
                
                # Exportar resultados
                st.subheader("Exportar resultados")
                
                # Crear DataFrame con resultados
                results_df = pl.DataFrame(
                    {
                        'Variable': x_vars + ['Intercepto'],
                        'Coeficiente': list(model.coef_) + [model.intercept_],
                    }
                )
                results_df = results_df.with_columns(pl.lit(r2).alias('R²'))
                
                # Opción para descargar como CSV
                csv = results_df.write_csv()
                st.download_button(
                    label="Descargar resultados CSV",
                    data=csv,
                    file_name="resultados_r2.csv",
                    mime="text/csv",
                )
    else:
        st.info("Por favor, ingresa una URL válida de Google Sheets")

# Información adicional
with st.expander("Información sobre el coeficiente R²"):
    st.write("""
    **¿Qué es el coeficiente R²?**
    
    El coeficiente de determinación (R²) mide la proporción de la varianza en la variable dependiente que puede ser explicada por las variables independientes. 
    
    - **R² = 1**: Indica que el modelo explica toda la variabilidad de los datos.
    - **R² = 0**: Indica que el modelo no explica nada de la variabilidad de los datos.
    
    Generalmente:
    - **R² < 0.3**: Relación débil
    - **0.3 ≤ R² < 0.7**: Relación moderada
    - **R² ≥ 0.7**: Relación fuerte
    """)

# Instrucciones de configuración
with st.expander("Configuración de Google Sheets API"):
    st.write("""
    Para usar esta aplicación necesitas:
    
    1. **Crear un proyecto en Google Cloud Platform**
    2. **Habilitar la API de Google Sheets**
    3. **Crear una cuenta de servicio y descargar el archivo JSON de credenciales**
    4. **Compartir tu hoja de Google con la dirección de correo de la cuenta de servicio**
    5. **Subir tu archivo JSON de credenciales al área de secretos de Streamlit**
    
    Para más detalles, consulta la [documentación de gspread](https://docs.gspread.org/en/latest/oauth2.html).
    """)
