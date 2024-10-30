import pandas as pd
from keras import Sequential, Input
from keras.activations import relu
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from math import sqrt
import matplotlib.pyplot as plt



if __name__ == '__main__':
    df_adra = pd.read_csv('adra.csv', delimiter=';', na_values='n/d')
    df_adra['FECHA'] = pd.to_datetime(df_adra['FECHA'], format='%d/%m/%y')
    df_adra.dropna(inplace=True)
    df_adra.set_index('FECHA', inplace=True)


    # Temperatura, humedad, precipitacion.
    df_input_output = df_adra[['Al10TMed', 'Al10VelViento', 'Al10Rad', 'Al10ETo']]


    df_train, df_val = train_test_split(df_input_output, train_size=0.8, random_state=123)

    arr_train_inputs = df_train.to_numpy()[:, :-1]
    arr_train_outputs = df_train.to_numpy()[:, -1]
    arr_val_inputs = df_val.to_numpy()[:, :-1]
    arr_val_outputs = df_val.to_numpy()[:, -1]

    numero_inputs = arr_train_inputs.shape[1]
    numero_outpus = 1

    # Decisiones en proceso de mnodelado.
    numero_neuronas_capas_ocultas = 60
    function_activacion = 'relu'
    uso_bias = True
    funcion_entrenamiento = 'adam'
    loss_function = 'mse'
    list_metricas = ['mae', 'mape']
    batch_size = 5
    numero_etapas = 50

    # Modelo.
    model = Sequential()
    model.add(Input(shape=(numero_inputs,), name='capa_entrada'))
    model.add(Dense(units=numero_neuronas_capas_ocultas, activation=function_activacion, name='capa_oculta_1', use_bias=uso_bias))
    model.add(Dropout(0.2))
    model.add(Dense(units=numero_neuronas_capas_ocultas, activation=function_activacion, name='capa_oculta_2', use_bias=uso_bias))
    model.add(Dropout(0.2))
    model.add(Dense(units=numero_outpus, name='capa_salida'))


    model.compile(optimizer=funcion_entrenamiento, loss= loss_function, metrics= list_metricas)

    history_object = model.fit(arr_train_inputs,
                              arr_train_outputs,
                              batch_size=batch_size,
                              validation_data=(arr_val_inputs, arr_val_outputs),
                              epochs=numero_etapas)


    dict_resultados_historico = history_object.history

    x_values = history_object.epoch
    y_entrenamiento = dict_resultados_historico['loss']
    y_validacion = dict_resultados_historico['val_loss']

    plt.plot(x_values, y_entrenamiento, '-', c='blue', label='Entrenamiento')
    plt.plot(x_values, y_validacion, '--', c='darkblue', label='Validacion')

    plt.xlabel('Etapas')
    plt.ylabel('Error (mape, %)')
    plt.legend()
    plt.show()

    y_entrenamiento_mape = dict_resultados_historico['mape']
    y_validacion_mape = dict_resultados_historico['val_mape']
    plt.plot(x_values, y_entrenamiento_mape, '-', c='blue', label='Entrenamiento')
    plt.plot(x_values, y_validacion_mape, '--', c='darkblue', label='Validacion')

    plt.xlabel('Etapas')
    plt.ylabel('Error')
    plt.legend()
    plt.show()









