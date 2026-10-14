# Proyecto de Métodos Numéricos

Aplicación educativa en Python para aproximar la solución de sistemas lineales $Ax=b$ mediante los métodos iterativos de Gauss-Jacobi y Gauss-Seidel. Incluye una interfaz gráfica de escritorio y una versión de consola.

## Contenido

- [Características](#características)
- [Requisitos e instalación](#requisitos-e-instalación)
- [Ejecución](#ejecución)
- [Uso](#uso)
- [Métodos y métricas](#métodos-y-métricas)
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

## Validación y límites

- Se valida que $A$ sea cuadrada, que las dimensiones de $A$ y $b$ coincidan y que los datos sean finitos.
- Se rechazan una diagonal nula, iteraciones menores que 1 y tolerancias no positivas.
- La dimensión máxima para sistemas personalizados es 10 incógnitas.
- La opción de comparación usa el primer ejercicio precargado; no compara automáticamente todos los ejercicios.
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

No hay actualmente una suite automatizada de pruebas en el repositorio. Como comprobación básica de sintaxis se pueden compilar ambos scripts:

```powershell
python -m py_compile demo.py interfaz_grafica.py
```

La prueba funcional manual recomendada es ejecutar los ejercicios precargados de Jacobi y Gauss-Seidel, comparar ambos métodos y resolver un sistema personalizado pequeño cuya solución se conozca.