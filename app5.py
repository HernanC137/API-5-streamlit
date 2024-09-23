import streamlit as st
import pandas as pd
import pickle
from pycaret.classification import predict_model
import tempfile

# Cargar el modelo y los datos en st.session_state si no están ya cargados
if 'modelo' not in st.session_state:
    with open('ridge_model.pkl', 'rb') as model_file:
        st.session_state['modelo'] = pickle.load(model_file)

if 'prueba_APP' not in st.session_state:
    st.session_state['prueba_APP'] = pd.read_csv('prueba_APP.csv')

# Definir funciones y listas


# Función para predicción individual
def prediccion_individual():
    st.header("Predicción manual de datos")
    
    # Inputs manuales
    Email = st.text_input("Email", value="hcortes356@gmail.com")
    address = st.selectbox("Address", options=['Munich', 'Ausburgo', 'Berlin', 'Frankfurt'], index=0)  
    dominio = st.selectbox("Dominio", options=['yahoo', 'Otro', 'gmail', 'hotmail'], index=2)  
    tec = st.selectbox("Tecnología", options=['PC', 'Smartphone', 'Iphone', 'Portatil'], index=1)
    Avg_Session_Length = st.text_input("Avg. Session Length", value="32.063775")
    Time_on_App = st.text_input("Time on App", value="10.719")
    Time_on_Website = st.text_input("Time on Website", value="37.712")
    Length_of_Membership = st.text_input("Length of Membership", value="3.004743")

    if st.button("Calcular"):
        try:
            Avg_Session_Length = float(Avg_Session_Length)
            Time_on_App = float(Time_on_App)
            Time_on_Website = float(Time_on_Website)
            Length_of_Membership = float(Length_of_Membership)

            # Crear el dataframe a partir de los inputs del usuario
            user = pd.DataFrame({
                'Avg. Session Length': [Avg_Session_Length], 
                'Time on App': [Time_on_App], 
                'Time on Website': [Time_on_Website], 
                'Length of Membership': [Length_of_Membership],
                'Email': [Email], 
                'Address': [address], 
                'dominio': [dominio], 
                'Tec': [tec], 
                'price': [0]  
            })
            
            #  Quita las variables que no se usan en el modelo
            data_pred = user.drop(columns=['price', 'Email'])
            
            # Realizar predicción
            predictions = predict_model(st.session_state['modelo'], data=data_pred)
            
            # Mostrar predicciones
            st.write(f'Predicción de precio: {predictions["prediction_label"][0]}')
        
        except ValueError:
            st.error("Por favor, ingrese valores numéricos válidos en los campos correspondientes.")

    if st.button("Volver al menú principal"):
        st.session_state['menu'] = 'main'

# Función para predicción por base de datos
def prediccion_base_datos():
    st.header("Cargar archivo para predecir")
    uploaded_file = st.file_uploader("Cargar archivo Excel o CSV", type=["xlsx", "csv"])

    if st.button("Predecir con archivo"):
        if uploaded_file is not None:
            try:
                # Cargar el archivo subido
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name

                if uploaded_file.name.endswith(".csv"):
                    prueba = pd.read_csv(tmp_path)
                else:
                    prueba = pd.read_excel(tmp_path)

                base_modelo2 = prueba.drop(columns=['Email', 'price'])

                # Realizar predicción
                df_test = base_modelo2.copy()
                predictions = predict_model(st.session_state['modelo'], data=df_test)

                # Preparar archivo para descargar
                kaggle = pd.DataFrame({'Email': prueba["Email"], 'Precio': predictions["prediction_label"]})

                # Mostrar predicciones en pantalla
                st.write("Predicciones generadas correctamente!")
                st.write(kaggle)

                # Botón para descargar el archivo de predicciones
                st.download_button(label="Descargar archivo de predicciones",
                                   data=kaggle.to_csv(index=False),
                                   file_name="kaggle_predictions.csv",
                                   mime="text/csv")

            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.error("Por favor, cargue un archivo válido.")

    if st.button("Volver al menú principal"):
        st.session_state['menu'] = 'main'

# Función principal para mostrar el menú de opciones
def menu_principal():
    st.title("API de Predicción Académica")
    option = st.selectbox("Seleccione una opción", ["", "Predicción Individual", "Predicción Base de Datos"])

    if option == "Predicción Individual":
        st.session_state['menu'] = 'individual'
    elif option == "Predicción Base de Datos":
        st.session_state['menu'] = 'base_datos'

# Lógica para manejar el flujo de la aplicación
if 'menu' not in st.session_state:
    st.session_state['menu'] = 'main'

if st.session_state['menu'] == 'main':
    menu_principal()
elif st.session_state['menu'] == 'individual':
    prediccion_individual()
elif st.session_state['menu'] == 'base_datos':
    prediccion_base_datos()
