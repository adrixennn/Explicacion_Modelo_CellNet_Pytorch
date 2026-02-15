# Explicacion_Modelo_CellNet_Pytorch
Aquí presento una pequeña guía sobre el funcionamiento del algoritmo de Pytorch utilizado para un modelo supervisado de clasificación de tumores en función de las medidas de la célula en las clases "Normal", "Benigno" y "Maligno".

## Funcionamiento del Algoritmo.
El algoritmo utilizado para la clasificación en las predicciones consta de una red neuronal diseñada para aprender patrones no lineales a partir de 4 variables de entrada.

### Arquitectura del Modelo
La clase "Modelo" (la cuál hereda de "nn.Module") se compone de un constructor el cuál define:
* **El número de variables de entrada**: Recibe 4 features.
* **Capas ocultas**: Dos capas de 9 neuronas ("h1"=9 y "h2"=9) que se conectan entre sí mediante "fc1" (la capa de entrada con la primera oculta), "fc2" (la primera capa oculta con la segunda oculta) y "out" (la segunda capa oculta con la de salida). En las conexiones se establece la función de activación ReLU permitiendo al modelo aprender de relaciones complejas.
* **Capa de salida**: 3 neuronas de salida que corresponden con los valores asignados a cada clase (0 para "Normal", 1 para "Benigno" y 2 para "Maligno").

### Validación cruzada en 5 folds
Para precisar los hiperparámetros y que sean lo más efectivos posibles evitando el sobreajuste, implemento validación cruzada con "StratifiedKFold" con 5 splits. 
Para hacer los splits se utilizan los datos de los conjuntos X e y de entrenamiento obtenidos al principio, de esta forma no habrá visto el test hasta la evaluación del modelo permitiendo una generalización óptima evitando el "Data Leakage".
Mediante "enumerate(skfolds.split())" me devuelve los indices de los subconjuntos de entreno y validación por cada división creada en cada iteración.

### Entrenamiento
Por cada fold se precisan 100 épocas. Lo principal que hace posible el entrenamiento del modelo son los siguientes elementos:
* **Optimizador Adam**: Se encarga de elaborar la función matemática con la que se actualizan los pesos de la red para minimizar el error. Se inicializa con un "Learning Rate" de 0.01.
* **Función de pérdida CrossEntropyLoss**: Calcula el error basándose en la seguridad del modelo con sus predicciones y verifica si la respuesta es correcta o no.
* **"Backpropagation"**, propagación hacia atrás:
  - **optimizer.zero_grad()**: Pone a 0 todos los gradientes acumulados para evitar sumas erróneas en la siguiente iteración.
  - **error.backward()**: En base al error entre el valor predicho y el actual establece el camino por el que tienen que modificarse los pesos hacia atrás en la red neuronal, el cuál es indicado por la derivada. Establece el gradiente en el tensor.
  - **optimizer.step()**: Asigna los nuevos pesos en función del gradiente calculado.

### Evaluación
La evaluación nos muestra por consola el valor predicho y el real y maneja un contador con los aciertos del modelo.
Al final he añadido la métrica del "Accuracy" con el import siguiente "from torchmetrics import Accuracy".

## Consideraciones y detalles
El siguiente algoritmo se ha desarrollado teniendo en cuenta los siguientes detalles:
* El reinicio del modelo en cada iteración de la asignación de folds para evitar el filtrado de datos. De este modo el modelo no acumula aprendizaje falseando la validación.
* La gestión de los tipos de Tensores para que las entradas se conviertan a "torch.FloatTensor" y las salidas a "torch.LongTensor", lo cuál es necesario para el cálculo del error.
* Modos del modelo --> "modelo.eval()" y "torch.no_grad()"
   - "modelo.eval()": Pone al modelo en modo evaluación, la red congela su aprendizaje. No se modifican los pesos ni los parámetros internos.
   - "torch.no_grad()": Desactiva el motor de autogradiente y la "Backpropagation", ahorrando memoria y computación a la hora de querer exclusivamente predecir sin aprender.
* La semilla aleatoria se fija para garantizar que la división de los datos e inicialización de pesos sean iguales en cada ejecución.
* La red no devuelve la clase directamente, sino un tensor con valores de activación (logits) y utilizando la función "argmax()" se obtiene el índice de la neurona con mayor activación para posteriormente traducirla a la clase resultante con el diccionario "traduccion".
* En cada iteración de los folds, he guardado los errores en una matriz formada cada fila por los errores de cada fold, contando con 5 filas y 100 columnas. Para mostrar una gráfica lineal de la progresión de los errores con MatPlotLib, he tenido que transponer la matriz para que la variable Y fueran el error por época y la X cada división.
