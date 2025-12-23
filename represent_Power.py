#represent_Power.py

#Representar deteccion de transmisiones mediante medida de potencia en los canales
#El segundo argumento de linea de comando es el numero de muestras por ventana NUM_SAMP_BATCH
#El tercer argumento de linea de comando debe ser el nombre de archivo que contiene datos de campo electrico
#El cuarto argumento de linea de comando (opcional) es el puerto donde queremos ejecutar el grafico interactivo, en http://127.0.0.1:[puerto]
#Si existe cualquier quinto argumento de linea de comando, el programa interpreta que se desea generar un archivo con el dataframe
# que contiene una vision general de la media de Actividad en cada canal y una vision general de las anomalias
#Al final, el programa hace dos cosas:
# Dibuja grafica que muestra las de anomalias (con umbral de 3 dB) y actividad en todos los canales
# Deja una grafica interactiva en http://127.0.0.1:[port]

import numpy as np
import pandas as pd
from pandas import Series, DataFrame

import sys
#Se necesitan dos argumentos de entrada en linea de comando al programa. 

if len(sys.argv) < 3:
    print('Error: At least, two parameters are needed. ' +
        'First, the number of samples for the window length (must be a multiple of 4). ' +
        'Second, the name of the file with the electric field data of the remote station. ' +
        'Also, a third parameter specifies the port at http://127.0.0.1:[port] . If void, the port is 8050. ' +
        'For example, python represent_Power.py 60 \"SMM 2024-06-7 al 8.csv\"')
    sys.exit()

#numero de muestras de cada lote. Debe ser un entero multiplo de 4
NUM_SAMP_BATCH = (int(sys.argv[1]) // 4) * 4
print('Number of samples in each batch:', NUM_SAMP_BATCH)

#archivo del que se toman los datos de la estacion remota (ER)
archivo_efield = sys.argv[2]
print('Electric field data file:', archivo_efield)

#puerto donde se ejecutaran los resultados en el navegador
localhost_port = '8050'
if len(sys.argv) > 3:
	localhost_port = sys.argv[3]
url = 'http://127.0.0.1:' + localhost_port
print('The browser will run the results at', url)
localhost_port = int(localhost_port)
if (localhost_port < 2000) or (localhost_port > 65535):
	print('Error: Invalid port: ' + localhost_port)
	sys.exit()

import os
#evitar fragmentaciones de memoria en la GPU
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

#Leer archivo de datos de la JPIT
print('Reading data file:', archivo_efield)
#Columnas: "Tiempo", "Frecuencia (Hz)" ,"Level (dBµV/m)", "Offset (Hz)", "Arch_ Info"
df_ER = pd.read_csv(archivo_efield, engine='python', sep='","', decimal=',', header=0, 
                    names=['time', 'freq', 'level', 'offset', 'info'], encoding='iso8859-1')
#quitar comillas del principio y del final
df_ER.iloc[:,0] = df_ER.iloc[:,0].str.replace('"', '')
df_ER.iloc[:,-1] = df_ER.iloc[:,-1].str.replace('"', '')

#print(df_ER.head())
#frecuencias y tiempos, no repetidos
frequencies = df_ER['freq'].unique()
time_steps = df_ER['time'].unique()
ncols = len(frequencies)
nrows = len(time_steps)
print('num frequencies:', ncols)
print('num time_steps:', nrows)

#calcular el intervalo de tiempo de la ventana e intervalo total
from datetime import datetime
date0 = datetime.strptime(time_steps[0],'%d/%m/%Y  %H:%M:%S,%f')
date1 = datetime.strptime(time_steps[1],'%d/%m/%Y  %H:%M:%S,%f')
date_batch = datetime.strptime(time_steps[NUM_SAMP_BATCH],'%d/%m/%Y  %H:%M:%S,%f')
date_last = datetime.strptime(time_steps[nrows-1],'%d/%m/%Y  %H:%M:%S,%f')
#longitud total, en horas enteras
total_hours_f = (date_last - date0).total_seconds() / 3600
total_hours = f"{total_hours_f:.3f}"
print('Total length = ' + total_hours + ' hours')
#periodo de muestreo
sample_period_f = (date1 - date0).total_seconds()
sample_period = f"{sample_period_f:.3f}"
print('Sample period = ' + sample_period + ' s')
#longitud de la ventana de tiempo, en minutos enteros
window_minutes_f = (date_batch - date0).total_seconds() / 60
window_minutes = f"{window_minutes_f:.3f}"
print('Time window length = ' + window_minutes + ' minutes')

#Canales minimo y maximo
channel_min_f = min(frequencies)
channel_max_f = max(frequencies)
channel_min_MHz = f"{min(frequencies)/1e6:.6f}"
channel_max_MHz = f"{max(frequencies)/1e6:.6f}"
channel_width_kHz = f"{(channel_max_f - channel_min_f)/(ncols-1)/1e3:.1f}"
num_channels = str(ncols)
print('There are ' + num_channels + ' channels, from ' + channel_min_MHz + ' to ' + channel_max_MHz + ' MHz. Channel width = ' + channel_width_kHz + ' kHz')

#Extraer un dataframe de los datos del archivo
print('Making dataframe')
#ir metiendo en una lista cada columna de levels correspondientes a cada frecuencia
lista = []
for ctr in range(ncols):
    col = Series(df_ER.loc[df_ER['freq'] == frequencies[ctr], 'level'])
    lista.append(col.values)
    ctr += 1
#hacer un dataframe con la lista
df = DataFrame(lista, columns=time_steps, index=frequencies)
#Se desea que las columnas de df sean las frecuencias de los diferentes canales, y las filas (index) sean los tiempos
df = df.transpose()

#Pasar a unidades naturales, en uV/m
df_nat = 10 ** (df/20)

#Series de valores cuadraticos medios
sr_power_serie = pd.DataFrame.sum(df_nat.pow(2)) / nrows
#Menor de los valores cuadraticos medios
min_power_serie = sr_power_serie.min()
channel_min_power_serie = sr_power_serie.idxmin()
print("min average power:", min_power_serie, "(uV/m)\u00b2 in channel", channel_min_power_serie)
#Mayor de los valores cuadraticos medios
max_power_serie = sr_power_serie.max()
channel_max_power_serie = sr_power_serie.idxmax()
print("max average power:", max_power_serie, "(uV/m)\u00b2 in channel", channel_max_power_serie)

#Crear secuencias correlativas. Se trata de trocear en lotes el vector de datos de amplitud de campo.
# Se trocea deslizando una ventana una muestra hacia la derecha, para cada lote.
# De esta manera, el vector queda convertido en una matriz en la que cada fila es un lote

#Funcion para crear vectores secuencias correlativas de los datos de la JPIT
def create_correlative_sequences(values, NUM_SAMP_BATCH=NUM_SAMP_BATCH):
    #crea sequencias correlativas de longitud NUM_SAMP_BATCH, que constituyen un lote
    #Cada lote se obtiene deslizando una ventana una muestra hacia la derecha
    #Cada lote sera una fila de la matriz resultante
    output = []
    for i in range(len(values) - NUM_SAMP_BATCH + 1):
        output.append(values[i : (i + NUM_SAMP_BATCH)])
    return np.stack(output)

#Matriz en que cada fila es una secuencia enventanada de la JPIT
x_JPIT = create_correlative_sequences(df_nat)
print("Array in which each row is a windowed data sequence of length", NUM_SAMP_BATCH)
print("Input shape: ", x_JPIT.shape)

#Hacer test power sobre los datos de la JPIT
print('Calculate power in each window for every channel')
#lista de potencias power obtenidas en cada ventana. Cada linea de la lista es un canal
lista_potencias = []
for canal in range(x_JPIT.shape[2]):
    print("frequency:", canal, "of", ncols, end='\r', flush=True)
    lista_ventanas = []
    for ventana in range(x_JPIT.shape[0]): 
        serie = x_JPIT[ventana, :, canal]
        #Calcular la potencia de la ventana y meterla en una lista
        window_power = np.sum(serie ** 2) / NUM_SAMP_BATCH
        lista_ventanas.append(window_power)
    lista_potencias.append(lista_ventanas)
    #print(*lista_potencias, sep = " ")
    
#apilar los vectores para hacer una matriz
#Matriz en que cada fila es un canal y las columnas son las potencias en cada ventana
JPIT_pot = np.stack(lista_potencias)
#print("JPIT_pot_dB shape: ", JPIT_pot.shape)
#Ahora cada columna es un canal y las filas son las potencias en cada ventana
JPIT_pot = JPIT_pot.transpose()
#Dar las medidas de potencia en dB con relacion a la minima potencia media
JPIT_pot_dB = 10 * np.log10(JPIT_pot / min_power_serie)

#Convertir las medidas de error a un dataframe
muestra_inicial = NUM_SAMP_BATCH // 2
muestra_final = len(df.index) - (NUM_SAMP_BATCH - muestra_inicial) + 1
#El prefijo es porque la primera ventana esta centrada en la muestra_inicial
prefijo = np.zeros((muestra_inicial, JPIT_pot_dB.shape[1])) * float("NaN")
JPIT_pot_dB = np.concatenate((prefijo, JPIT_pot_dB), axis=0)
#print("JPIT_pot_dB shape: ", JPIT_pot_dB.shape)
#La ultima ventana esta centrada en la muestra_final
index_JPIT_pot_dB = df.index[0 : muestra_final]
df_JPIT_pot_dB = pd.DataFrame(JPIT_pot_dB, index=index_JPIT_pot_dB, columns=df.columns)
#Numero de ventanas
num_windows = nrows - NUM_SAMP_BATCH + 1
print('Number of surveillance windows = ' + str(num_windows))
print("Total number of samples: ", nrows)

#Vision general de la media de Actividad en cada canal
df_Power_anomalies = 10*np.log10(sr_power_serie/min_power_serie).to_frame(name='activity')
df_Power_anomalies.index = df_Power_anomalies.index * 1E-6

#Vision general de las anomalias
# Serie hecha como copia de df_JPIT_pot_dB.
df_JPIT_pot_dB_copia = pd.DataFrame(df_JPIT_pot_dB, copy=True)
#Solo consideramos aquellas muestras que superen el umbral 3 dB
df_JPIT_pot_dB_copia[df_JPIT_pot_dB_copia <= 3] = float("NaN")
#vector de anomalias con umbral 3 dB
anomalies3 = df_JPIT_pot_dB_copia.to_numpy()
anomalies3 = ~np.isnan(anomalies3)
# Tanto por ciento de anomalias en el canal actual
serie_percent_anomalies = pd.Series(100 * np.sum(anomalies3, axis=0) / num_windows, index=df_Power_anomalies.index, name='anomalies')
df_Power_anomalies['anomalies'] = 100 * np.sum(anomalies3, axis=0) / num_windows

if len(sys.argv) > 4:
	#Generar un archivo con la vision general de la media de Actividad en cada canal
	nombreArchivo_anomalies_Power_Activity = "anomalies_Power_Activity_" + str(NUM_SAMP_BATCH) + "_" + archivo_efield
	df_Power_anomalies.to_csv(nombreArchivo_anomalies_Power_Activity)
	print("File", nombreArchivo_anomalies_Power_Activity, "has been generated")

#Dibujar anomalias y actividad
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])
# Add traces
fig.add_trace(
    go.Bar(x=df_Power_anomalies.index, y=df_Power_anomalies['anomalies'], name="% of anomalies, with threshold > 3 dB", marker={"color":'red'}, opacity=0.3),
    secondary_y=False,
)
fig.add_trace(
    go.Scatter(x=df_Power_anomalies.index, y=df_Power_anomalies['activity'], name="Channel Power Activity (dB)", line={"color":'blue', "width":3}),
    secondary_y=True,
)
# Add figure title
fig.update_layout(
    title_text="Anomalies vs Average Power relative to minimum power channel"
)
# Set x-axis title
fig.update_xaxes(title_text="Channel (MHz)")
# Set y-axes titles
fig.update_yaxes(title_text="% of anomalies, with threshold > 3 dB", secondary_y=False, title_font_color="red", tickfont={"color":"red"})
fig.update_yaxes(title_text="Average Power relative to minimum power channel (dB)", secondary_y=True, title_font_color="blue", tickfont={"color":"blue"})
#set legend position
fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.3, bgcolor='rgba(0,0,0,0)'), font={"size":30})

fig.show(renderer='browser')

#Ordenar potencias por orden descendente
sr_power_serie_descend = sr_power_serie.sort_values(ascending=False)

#Funcion para calcular anomalias
def calculate_anomalies(threshold, canal):
    # Serie hecha con un canal de JPIT_pot_dB. Sus valores iran variando a NaN en funcion del threshold
    df_JPIT_pot_canal_dB = pd.DataFrame(df_JPIT_pot_dB.loc[:, canal], copy=True)
    #Solo consideramos aquellas muestras que superen el umbral threshold dB
    df_JPIT_pot_canal_dB[df_JPIT_pot_canal_dB <= threshold] = float("NaN")
    #vector de anomalias en el canal
    anomalies = df_JPIT_pot_canal_dB.to_numpy(copy=True)
    anomalies = ~np.isnan(anomalies)
    # Tanto por ciento de anomalias en el canal actual
    percent_anomalies_per_channel = 100 * np.sum(anomalies) / num_windows
    return df_JPIT_pot_canal_dB, percent_anomalies_per_channel


#dibujar
#grafico interactivo
import plotly.graph_objects as go
from dash import Dash, html, dash_table, dcc, callback, Output, Input

# Initialize the app
print('Plotting')
app = Dash(__name__)

# App layout
app.layout = [
    html.Hr(),
    html.Div(children='CHANNEL ACTIVITY SURVEILLANCE APPLICATION BASED ON CHANNEL POWER. © Pablo Torío'),
    html.Hr(),
    html.Div(children='File: ' + archivo_efield),
    html.Div(children='Sample period = ' + str(sample_period) + ' s'),
    html.Div(children='Total length = ' + str(nrows) + ' samples = ' + total_hours + ' hours'),
    html.Div(children='Number of surveillance windows = ' + str(num_windows)),
    html.Div(children='Surveillance window length = ' + str(NUM_SAMP_BATCH) + ' samples = ' + window_minutes + ' minutes'),
    html.Hr(),
    html.Div(children='There are ' + num_channels + ' channels, from ' + channel_min_MHz + ' to ' + channel_max_MHz + ' MHz. Channel width = ' + channel_width_kHz + ' kHz'),
    html.Hr(),
    html.Div(children='Activity measured by signal to noise ratio (S/R) related to the channel with the minimum average power'),
    html.Hr(),
    html.Div(children='minimum mean square field:  ' + f"{min_power_serie:.2f}" + ' (uV/m)\u00b2,  in channel ' + str(channel_min_power_serie)),
    html.Div(children='maximum mean square field:  ' + f"{max_power_serie:.2f}" + ' (uV/m)\u00b2,  in channel ' + str(channel_max_power_serie)),
    html.Hr(),
    dcc.Slider(min=0, max=60, step=1, value=5, id='threshold_slider'),
    html.Div(id='threshold_val'),
    html.Hr(),
    html.Div(children='CHANNEL BROWSING BY ASCENDING FREQUENCIES --- Channel frequency list (Hz), sorted by ascending frequencies:'),
    dcc.Dropdown(options=list(df.columns), value=df.columns[0], id='channel_freqs', style={'color': 'blue'}),
    html.Div(id='SR_freqs_label'),
    html.Div(id= 'below_freqs_label'),
    dcc.Graph(id='graph_freqs'),   
    html.Hr(),
    html.Hr(),
    html.Div(children='CHANNEL BROWSING BY DESCENDING ACTIVITY --- Channel frequency list (Hz), sorted by descending S/R:'),
    dcc.Dropdown(options=list(sr_power_serie_descend.index), value=sr_power_serie_descend.index[0], id='channel_sorted_desc', style={'color': 'blue'}),
    html.Div(id='SR_sorted_desc_label'),
    html.Div(id= 'below_sorted_desc_label'),
    dcc.Graph(id='graph_sorted_desc')   
]

#callbacks

#actualizacion del valor umbral (threshold) para la S/R
@callback(
    Output('threshold_val', 'children'),
    Input('threshold_slider', 'value'))
def update_output(value):
    label_text_value = 'S/R threshold = ' + str(value) + ' dB'
    return label_text_value

    
#callbacks para graficos
@callback(
    Output(component_id='graph_freqs', component_property='figure'),
    Output(component_id='SR_freqs_label', component_property='children'),
    Output(component_id='below_freqs_label', component_property='children'),
    Input(component_id='channel_freqs', component_property='value'),
    Input(component_id='threshold_slider', component_property='value')
)
def update_channel_freqs(col_chosen, threshold_chosen):
    #print('**** update_heatmap de channel_freqs')
    df_JPIT_pot_canal_dB, percent_anomalies_per_channel = calculate_anomalies(threshold_chosen, col_chosen)
    df_JPIT_pot_canal_dB = pd.concat([df_JPIT_pot_canal_dB]*5, axis=1)
    df_JPIT_pot_canal_dB = df_JPIT_pot_canal_dB.transpose()
    trace1 = go.Scatter(x=df.index, y=df.loc[:, col_chosen], opacity=0.75, marker = {'color' : 'black'}, name='dBuV/m')
    trace2 = go.Heatmap(z=df_JPIT_pot_canal_dB, colorbar={"title": "Activity (dB)"}, name='SR')
    fig = go.FigureWidget(data=[trace1, trace2],
                    layout=go.Layout(yaxis_title='electric field (dBuV/m)'))
    #etiquetas
    label_text = ('Average Activity = ' + np.array2string(10*np.log10(sr_power_serie.loc[col_chosen]/min_power_serie), precision=2) + ' dB' +
        ' --- Percentage of surveillance windows exceeding S/R threshold (' + str(threshold_chosen) + ' dB) = ' + np.array2string(percent_anomalies_per_channel, precision=2) + ' %')
    below_text = ('GRAPH --- Above: Channel electric field (dBuV/m) --- Below: Color map of channel Activity. Color bars are only present when S/R > ' +
        str(threshold_chosen)  + ' dB')
    return fig, label_text, below_text
    
@callback(
    Output(component_id='graph_sorted_desc', component_property='figure'),
    Output(component_id='SR_sorted_desc_label', component_property='children'),
    Output(component_id='below_sorted_desc_label', component_property='children'),
    Input(component_id='channel_sorted_desc', component_property='value'),
    Input(component_id='threshold_slider', component_property='value')
)
def update_channel_sorted_desc(col_chosen, threshold_chosen):
    #print('&&&& update_heatmap de channel_sorted_desc')
    df_JPIT_pot_canal_dB, percent_anomalies_per_channel = calculate_anomalies(threshold_chosen, col_chosen)
    df_JPIT_pot_canal_dB = pd.concat([df_JPIT_pot_canal_dB]*5, axis=1)
    df_JPIT_pot_canal_dB = df_JPIT_pot_canal_dB.transpose()
    trace1 = go.Scatter(x=df.index, y=df.loc[:, col_chosen], opacity=0.75, marker = {'color' : 'black'}, name='dBuV/m')
    trace2 = go.Heatmap(z=df_JPIT_pot_canal_dB, colorbar={"title": "Activity (dB)"}, name='SR')
    fig = go.FigureWidget(data=[trace1, trace2],
                    layout=go.Layout(yaxis_title='electric field (dBuV/m)'))
    #etiquetas
    label_text = ('Average Activity = ' + np.array2string(10*np.log10(sr_power_serie.loc[col_chosen]/min_power_serie), precision=2) + ' dB' +
        ' --- Percentage of surveillance windows exceeding S/R threshold (' + str(threshold_chosen) + ' dB) = ' + np.array2string(percent_anomalies_per_channel, precision=2) + ' %')
    below_text = ('GRAPH --- Above: Channel electric field (dBuV/m) --- Below: Color map of channel Activity. Color bars are only present when S/R > ' +
        str(threshold_chosen) + ' dB')
    return fig, label_text, below_text

import webbrowser 
print('Open browser at', url)
webbrowser.open(url)

if __name__ == '__main__':
    app.run(debug=False, port=localhost_port)















