# Proyecto de Métodos Numéricos

Aplicación educativa en Python para resolver sistemas lineales $Ax=b$ con Gauss-Jacobi y Gauss-Seidel, y aproximar raíces de polinomios reales con Bairstow. Incluye una interfaz gráfica de escritorio y una versión de consola; Bairstow está integrado en la interfaz gráfica.

## Contenido

- [Características](#características)
- [Requisitos e instalación](#requisitos-e-instalación)
- [Ejecución](#ejecución)
- [Uso](#uso)
- [Métodos y métricas](#métodos-y-métricas)
- [Bairstow en la interfaz](#bairstow-en-la-interfaz)
- [Validación y límites](#validación-y-límites)
- [Archivos generados](#archivos-generados)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Pruebas](#pruebas)

## Características

- Seis sistemas de ejemplo: tres configurados para Jacobi y tres para Gauss-Seidel.
- Ingreso de sistemas personalizados de 1 a 10 incógnitas.
- Configuración de método, iteraciones máximas y tolerancia.
- Historial tabular de iteraciones, valores de las incógnitas y errores.
- Gráficas de convergencia con estilos de líneas, barras, dispersión y mapa de calor. La interfaz también ofrece temas visuales y animación.
- Exportación de resultados a CSV.
- Comparación de Jacobi y Gauss-Seidel sobre el primer ejercicio precargado.
- Cálculo de raíces reales y complejas de polinomios mediante Bairstow en una pestaña independiente.

## Requisitos e instalación

- Python 3.10 o posterior.
- Tkinter para la aplicación gráfica. En muchas instalaciones de Python para Windows ya viene incluido; en Linux puede requerir el paquete del sistema `python3-tk`.
- NumPy y Matplotlib, declarados en [`requirements.txt`](requirements.txt).

Desde la carpeta del proyecto, instala las dependencias:

```powershell
python -m pip install -r requirements.txt
```

En Windows, si el comando `python` no está disponible, se puede usar el lanzador `py`:

```powershell
py -m pip install -r requirements.txt
```

## Ejecución

Interfaz gráfica:

```powershell
python interfaz_grafica.py
```

Versión de consola:

```powershell
python demo.py
```

En la consola, las opciones 1–6 ejecutan los ejercicios precargados, 7 solicita un sistema personalizado, 8 compara ambos métodos sobre el ejercicio 1, 9 permite configurar los parámetros y 0 termina el programa.

## Uso

En la interfaz gráfica:

1. Abre **Ejercicios Precargados** y selecciona un ejercicio, o abre **Ingreso de Datos** para escribir una matriz $A$ y un vector $b$.
2. Elige Jacobi o Seidel y define iteraciones máximas y tolerancia.
3. Pulsa **Resolver Sistema**. La pestaña **Resultados** presenta el sistema, la solución aproximada, los errores, la tabla por iteración y la gráfica.
4. Activa la exportación CSV o la animación antes de resolver si necesitas esas salidas.

Para hallar raíces con Bairstow, abre **Raíces (Bairstow)**, ingresa los coeficientes en orden descendente, define `r`, `s`, tolerancia e iteraciones máximas y pulsa **Resolver polinomio**. La pestaña muestra los factores, las raíces y los residuos de cada iteración.

Para un sistema personalizado, $A$ debe ser cuadrada, $b$ debe tener una entrada por fila y todos los valores deben ser finitos. La diagonal de $A$ no puede contener ceros.

## Métodos y métricas

Los dos métodos parten de $x^{(0)}=0$ y recalculan las incógnitas iterativamente.

**Gauss-Jacobi** calcula cada componente nueva usando únicamente los valores de la iteración anterior:

$$
x_i^{(k+1)} = \frac{b_i - \sum_{j \ne i} a_{ij}x_j^{(k)}}{a_{ii}}.
$$

**Gauss-Seidel** actualiza las componentes en orden y utiliza dentro de la misma iteración los valores nuevos que ya calculó:

$$
x_i^{(k+1)} = \frac{b_i - \sum_{j < i} a_{ij}x_j^{(k+1)} - \sum_{j > i} a_{ij}x_j^{(k)}}{a_{ii}}.
$$

El programa muestra estas métricas:

| Métrica | Definición implementada | Uso |
| --- | --- | --- |
| Error absoluto | Norma euclídea del cambio: $\|x^{(k+1)}-x^{(k)}\|_2$ | Detiene el método cuando es menor que la tolerancia. |
| Error relativo | Error absoluto dividido por $\|x^{(k+1)}\|_2 + 10^{-10}$ | Indicador adicional que se muestra y grafica. |
| Iteraciones | Número de actualizaciones realizadas | Se detiene al alcanzar el máximo configurado. |
| Dominancia diagonal | Comprueba si $|a_{ii}| > \sum_{j\ne i}|a_{ij}|$ en todas las filas | Informa una condición suficiente de convergencia para estos métodos. |

Los valores iniciales comunes son 6 iteraciones y tolerancia $10^{-6}$. La GUI permite de 1 a 100 iteraciones; la consola permite configurar cualquier entero positivo. La tolerancia debe ser positiva.

**Interpretación importante:** el error absoluto implementado mide el cambio entre dos aproximaciones consecutivas; no es el residuo del sistema $\|Ax-b\|_2$. Una matriz que no sea diagonalmente dominante todavía podría converger, pero el programa no garantiza que lo haga.

## Bairstow en la interfaz

Bairstow trabaja con un polinomio real cuyos coeficientes se introducen desde el mayor grado hasta el término independiente. Por ejemplo, para

$$
x^4 - 3x^3 + 3x^2 - 3x + 2
$$

se ingresan `1, -3, 3, -3, 2`. El programa busca factores cuadráticos de la forma $x^2-rx-s$, los divide sucesivamente del polinomio y resuelve directamente el factor residual de grado 1 o 2. Las raíces complejas se muestran en forma `a + bi`.

| Parámetro/métrica | Alcance |
| --- | --- |
| Grado | De 2 a 12 en la interfaz. |
| Iteraciones máximas | De 1 a 1000 por factor cuadrático; valor inicial 100. |
| Tolerancia | Máximo absoluto de los dos coeficientes del residuo, calculado después de normalizar el polinomio a coeficiente principal 1; valor inicial $10^{-8}$. |
| Valores iniciales | `r = 1` y `s = -1` por defecto; pueden cambiarse si no converge. |
| Salida | Factores, raíces aproximadas e historial de `r`, `s`, residuos, correcciones y matriz Jacobiana por iteración. |

La convergencia depende del polinomio y de los valores iniciales. Selecciona una fila del historial para ver el polinomio de la etapa, los coeficientes de la división sintética, el cociente provisional, los residuos, la matriz Jacobiana, la ecuación de corrección, los incrementos y los nuevos valores de `r` y `s`. Si el sistema de corrección se vuelve singular o no se alcanza la tolerancia, la aplicación informa el problema y permite probar otros valores iniciales o más iteraciones. Esta función no modifica ni reemplaza los métodos para sistemas lineales.

## Validación y límites

- Se valida que $A$ sea cuadrada, que las dimensiones de $A$ y $b$ coincidan y que los datos sean finitos.
- Se rechazan una diagonal nula, iteraciones menores que 1 y tolerancias no positivas.
- La dimensión máxima para sistemas personalizados es 10 incógnitas.
- La opción de comparación usa el primer ejercicio precargado; no compara automáticamente todos los ejercicios.
- Bairstow está disponible en la interfaz gráfica; el menú de consola conserva Jacobi, Seidel y su comparación.
- Los métodos iterativos pueden no converger para ciertos sistemas. El máximo de iteraciones limita el tiempo de cálculo, pero no asegura una solución precisa.
- El proyecto no calcula actualmente el residuo $\|Ax-b\|_2$ ni incluye benchmarks de rendimiento.

## Archivos generados

Los resultados se escriben en el directorio de trabajo desde el que se ejecuta el programa:

- `grafica_*.png`: convergencia de las variables desde la consola.
- `comparacion_*.png`: comparación gráfica de errores desde la consola.
- `resultados_*.csv`: tabla de iteraciones exportada con `;` como separador.

La interfaz permite guardar la gráfica y exportar resultados desde sus controles. Estos artefactos y la caché de Python están excluidos del control de versiones por [`.gitignore`](.gitignore).

## Estructura del proyecto

| Archivo | Propósito |
| --- | --- |
| `demo.py` | Menú de consola, ejercicios, métodos, comparación, gráficas y exportación CSV. |
| `interfaz_grafica.py` | Interfaz Tkinter, entrada de datos, resultados tabulares y gráficas integradas. |
| `requirements.txt` | Dependencias de Python. |
| `.gitignore` | Archivos generados que no deben versionarse. |

## Pruebas

El núcleo de Bairstow cuenta con pruebas unitarias para polinomios cuadráticos, cúbicos y cuárticos, raíces reales y complejas, y entradas inválidas. Jacobi y Seidel todavía no tienen una suite automatizada. Como comprobación básica de sintaxis se pueden compilar ambos scripts:

```powershell
python -m py_compile demo.py interfaz_grafica.py
```

Las pruebas unitarias del núcleo de Bairstow se ejecutan con:

```powershell
python -m unittest discover -s tests
```

La prueba funcional manual recomendada es ejecutar ejercicios precargados de Jacobi y Gauss-Seidel, comparar ambos métodos y resolver desde la pestaña Bairstow el polinomio de ejemplo cuya factorización es $(x^2+1)(x^2-3x+2)$.