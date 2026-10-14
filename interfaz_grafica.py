import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
import csv
from datetime import datetime


def formatear_factor_bairstow(r, s):
    """Devuelve el factor x^2-rx-s en una forma legible."""
    terminos = ["x^2"]
    for coeficiente, variable in ((-r, "x"), (-s, "")):
        if abs(coeficiente) < 1e-10:
            continue
        signo = "+" if coeficiente > 0 else "-"
        terminos.append(f" {signo} {abs(coeficiente):.8g}{variable}")
    return "".join(terminos)


def resolver_bairstow(coeficientes, iteraciones_maximas, tolerancia, r_inicial, s_inicial):
    """Calcula las raíces de un polinomio real mediante deflación de Bairstow."""
    coeficientes = np.asarray(coeficientes, dtype=float)
    if coeficientes.ndim != 1 or len(coeficientes) < 3:
        raise ValueError("El polinomio debe tener grado igual o mayor que 2.")
    if not np.isfinite(coeficientes).all() or coeficientes[0] == 0:
        raise ValueError("Los coeficientes deben ser finitos y el principal no puede ser cero.")
    if not isinstance(iteraciones_maximas, (int, np.integer)) or iteraciones_maximas < 1:
        raise ValueError("Las iteraciones máximas deben ser un entero mayor que cero.")
    if not np.isfinite(tolerancia) or tolerancia <= 0:
        raise ValueError("La tolerancia debe ser un número positivo y finito.")
    if not np.isfinite(r_inicial) or not np.isfinite(s_inicial):
        raise ValueError("Los valores iniciales r y s deben ser finitos.")

    factor_escala = float(coeficientes[0])
    coeficientes = coeficientes / factor_escala
    raices = []
    historial = []
    factores = []
    r = float(r_inicial)
    s = float(s_inicial)
    etapa = 1

    while len(coeficientes) - 1 >= 3:
        grado = len(coeficientes) - 1
        convergio = False

        for iteracion in range(1, iteraciones_maximas + 1):
            b = np.zeros(grado + 1)
            derivada_r = np.zeros(grado + 1)
            derivada_s = np.zeros(grado + 1)
            b[0] = coeficientes[0]

            for indice in range(1, grado + 1):
                b[indice] = coeficientes[indice] + r * b[indice - 1]
                if indice >= 2:
                    b[indice] += s * b[indice - 2]
                    derivada_r[indice] = (
                        b[indice - 1]
                        + r * derivada_r[indice - 1]
                        + s * derivada_r[indice - 2]
                    )
                    derivada_s[indice] = (
                        b[indice - 2]
                        + r * derivada_s[indice - 1]
                        + s * derivada_s[indice - 2]
                    )
                elif indice == 1:
                    derivada_r[indice] = b[0]

            residuo_x = b[-2]
            residuo_constante = b[-1] - r * residuo_x
            residuos = np.array([residuo_x, residuo_constante])
            norma_residuo = float(np.linalg.norm(residuos, ord=np.inf))
            detalle = {
                "etapa": etapa,
                "iteracion": iteracion,
                "polinomio": coeficientes.copy(),
                "b": b.copy(),
                "cociente": b[:-2].copy(),
                "r": r,
                "s": s,
                "residuo_x": residuo_x,
                "residuo_constante": residuo_constante,
                "norma": norma_residuo,
                "jacobiano": None,
                "correcciones": None,
                "r_nuevo": r,
                "s_nuevo": s,
                "convergio": norma_residuo <= tolerancia,
            }

            if norma_residuo <= tolerancia:
                historial.append(detalle)
                convergio = True
                break

            jacobiano = np.array([
                [derivada_r[-2], derivada_s[-2]],
                [derivada_r[-1] - residuo_x - r * derivada_r[-2],
                 derivada_s[-1] - r * derivada_s[-2]],
            ])
            escala = max(1.0, float(np.linalg.norm(jacobiano, ord=np.inf)) ** 2)
            if abs(float(np.linalg.det(jacobiano))) <= np.finfo(float).eps * escala:
                raise ValueError(
                    f"El sistema de corrección se volvió singular en la etapa {etapa}. "
                    "Prueba otros valores iniciales de r y s."
                )

            correcciones = np.linalg.solve(jacobiano, -residuos)
            if not np.isfinite(correcciones).all():
                raise ValueError("Bairstow produjo correcciones no finitas; prueba otros valores iniciales.")
            detalle["jacobiano"] = jacobiano.copy()
            detalle["correcciones"] = correcciones.copy()
            detalle["r_nuevo"] = r + float(correcciones[0])
            detalle["s_nuevo"] = s + float(correcciones[1])
            historial.append(detalle)
            r = detalle["r_nuevo"]
            s = detalle["s_nuevo"]

        if not convergio:
            raise ValueError(
                f"Bairstow no convergió en la etapa {etapa} tras {iteraciones_maximas} iteraciones. "
                "Aumenta el máximo o prueba otros valores iniciales de r y s."
            )

        discriminante = complex(r * r + 4 * s) ** 0.5
        raices.extend(((r + discriminante) / 2, (r - discriminante) / 2))
        factores.append(formatear_factor_bairstow(r, s))
        coeficientes = b[:-2].copy()
        etapa += 1

    if len(coeficientes) == 3:
        a, b_lineal, c = coeficientes
        discriminante = complex(b_lineal * b_lineal - 4 * a * c) ** 0.5
        raices.extend(((-b_lineal + discriminante) / (2 * a), (-b_lineal - discriminante) / (2 * a)))
        factores.append(formatear_factor_bairstow(-b_lineal / a, -c / a))
    elif len(coeficientes) == 2:
        raiz = complex(-coeficientes[1] / coeficientes[0])
        raices.append(raiz)
        factores.append(f"x - ({raiz.real:.8g})")

    if factor_escala != 1.0:
        factores.insert(0, f"{factor_escala:.8g}")

    return raices, historial, factores


class InterfazMetodosNumericos:
    def __init__(self, root):
        self.root = root
        self.root.title("Aplicativo Profesional de Métodos Numéricos")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Configuración de estilo
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('TLabel', background='#f0f0f0', font=('Arial', 10))
        self.style.configure('TButton', font=('Arial', 10, 'bold'), padding=10)
        self.style.configure('Header.TLabel', font=('Arial', 14, 'bold'), background='#2c3e50', foreground='white')
        self.style.configure('TNotebook', background='#f0f0f0')
        self.style.configure('TNotebook.Tab', font=('Arial', 11, 'bold'))
        
        # Variables
        self.n_var = tk.IntVar(value=3)
        self.metodo = tk.StringVar(value="jacobi")
        self.iteraciones = tk.IntVar(value=6)
        self.tolerancia = tk.DoubleVar(value=1e-6)
        self.exportar_csv = tk.BooleanVar(value=False)
        self.mostrar_animacion = tk.BooleanVar(value=False)
        self.tipo_grafica = tk.StringVar(value="lineas")
        self.tema = tk.StringVar(value="profesional")
        
        # Matrices para entrada
        self.entradas_A = []
        self.entradas_b = []
        self.modo_resultados = None
        self.ultimo_historial = None
        self.ultimo_errores = None
        self.ultimo_metodo = None
        self.grado_bairstow = tk.IntVar(value=4)
        self.iteraciones_bairstow = tk.IntVar(value=100)
        self.tolerancia_bairstow = tk.DoubleVar(value=1e-8)
        self.r_bairstow = tk.DoubleVar(value=1.0)
        self.s_bairstow = tk.DoubleVar(value=-1.0)
        self.entradas_coef_bairstow = []
        
        # Crear interfaz
        self.crear_interfaz()
        
    def crear_interfaz(self):
        # Notebook para pestañas
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Pestaña 1: Ingreso de datos
        self.pestana_entrada = ttk.Frame(self.notebook)
        self.notebook.add(self.pestana_entrada, text='📊 Ingreso de Datos')
        self.crear_pestana_entrada()
        
        # Pestaña 2: Ejercicios precargados
        self.pestana_ejercicios = ttk.Frame(self.notebook)
        self.notebook.add(self.pestana_ejercicios, text='📚 Ejercicios Precargados')
        self.crear_pestana_ejercicios()
        
        # Pestaña 3: Resultados
        self.pestana_resultados = ttk.Frame(self.notebook)
        self.notebook.add(self.pestana_resultados, text='📈 Resultados')
        self.crear_pestana_resultados()

        # Pestaña 4: Método de Bairstow
        self.pestana_bairstow = ttk.Frame(self.notebook)
        self.notebook.add(self.pestana_bairstow, text='Raíces (Bairstow)')
        self.crear_pestana_bairstow()

    def crear_pestana_bairstow(self):
        frame = ttk.Frame(self.pestana_bairstow)
        frame.pack(fill='both', expand=True, padx=20, pady=20)

        titulo = ttk.Label(
            frame, text="Cálculo de raíces por el método de Bairstow",
            style='Header.TLabel', padding=8
        )
        titulo.pack(fill='x', pady=(0, 10))

        configuracion = ttk.LabelFrame(frame, text="Polinomio y configuración", padding=8)
        configuracion.pack(fill='x', pady=(0, 8))

        ttk.Label(configuracion, text="Grado:").grid(row=0, column=0, sticky='w', padx=5, pady=4)
        ttk.Spinbox(
            configuracion, from_=2, to=12, textvariable=self.grado_bairstow, width=8
        ).grid(row=0, column=1, sticky='w', padx=5, pady=4)
        ttk.Button(
            configuracion, text="Aplicar grado", command=self.actualizar_coeficientes_bairstow
        ).grid(row=0, column=2, sticky='w', padx=5, pady=4)

        ttk.Label(configuracion, text="Iteraciones máximas por factor:").grid(
            row=0, column=3, sticky='w', padx=5, pady=4
        )
        ttk.Spinbox(
            configuracion, from_=1, to=1000, textvariable=self.iteraciones_bairstow, width=8
        ).grid(row=0, column=4, sticky='w', padx=5, pady=4)

        ttk.Label(configuracion, text="Tolerancia:").grid(
            row=1, column=0, sticky='w', padx=5, pady=4
        )
        ttk.Entry(configuracion, textvariable=self.tolerancia_bairstow, width=12).grid(
            row=1, column=1, sticky='w', padx=5, pady=4
        )
        ttk.Label(configuracion, text="r inicial:").grid(
            row=1, column=2, sticky='w', padx=5, pady=4
        )
        ttk.Entry(configuracion, textvariable=self.r_bairstow, width=12).grid(
            row=1, column=3, sticky='w', padx=5, pady=4
        )
        ttk.Label(configuracion, text="s inicial:").grid(
            row=1, column=4, sticky='w', padx=5, pady=4
        )
        ttk.Entry(configuracion, textvariable=self.s_bairstow, width=12).grid(
            row=1, column=5, sticky='w', padx=5, pady=4
        )

        ttk.Label(
            configuracion,
            text="Coeficientes en orden descendente: a_n, a_(n-1), ..., a_0",
        ).grid(row=2, column=0, columnspan=6, sticky='w', padx=5, pady=(8, 3))
        self.frame_coef_bairstow = ttk.Frame(configuracion)
        self.frame_coef_bairstow.grid(row=3, column=0, columnspan=6, sticky='ew', padx=5)
        ttk.Button(
            configuracion, text="Resolver polinomio", command=self.resolver_polinomio_bairstow
        ).grid(row=4, column=0, columnspan=6, pady=(10, 2))
        self.actualizar_coeficientes_bairstow()

        resultados = ttk.LabelFrame(frame, text="Resultados", padding=8)
        resultados.pack(fill='both', expand=True)
        self.notebook_resultados_bairstow = ttk.Notebook(resultados)
        self.notebook_resultados_bairstow.pack(fill='both', expand=True)

        pestana_resumen = ttk.Frame(self.notebook_resultados_bairstow)
        pestana_iteraciones = ttk.Frame(self.notebook_resultados_bairstow)
        pestana_desarrollo = ttk.Frame(self.notebook_resultados_bairstow)
        self.notebook_resultados_bairstow.add(pestana_resumen, text='Resumen')
        self.notebook_resultados_bairstow.add(pestana_iteraciones, text='Iteraciones')
        self.notebook_resultados_bairstow.add(pestana_desarrollo, text='Desarrollo detallado')
        self.pestana_iteraciones_bairstow = pestana_iteraciones

        self.resultado_bairstow = ttk.Label(
            pestana_resumen, text="Ingresa los coeficientes y pulsa Resolver polinomio.",
            justify='left', wraplength=1080
        )
        self.resultado_bairstow.pack(fill='both', expand=True, anchor='nw', padx=10, pady=10)

        columnas = (
            'Etapa', 'Iter.', 'r actual', 's actual', 'R1', 'R0',
            'Δr', 'Δs', 'r nuevo', 's nuevo', '||R||∞'
        )
        tabla_frame = ttk.Frame(pestana_iteraciones)
        tabla_frame.pack(fill='both', expand=True, padx=5, pady=5)
        self.tabla_bairstow = ttk.Treeview(
            tabla_frame, columns=columnas, show='headings', height=10
        )
        for columna in columnas:
            self.tabla_bairstow.heading(columna, text=columna)
            self.tabla_bairstow.column(columna, width=110, anchor='center')
        scrollbar_y = ttk.Scrollbar(tabla_frame, orient='vertical', command=self.tabla_bairstow.yview)
        scrollbar_x = ttk.Scrollbar(tabla_frame, orient='horizontal', command=self.tabla_bairstow.xview)
        self.tabla_bairstow.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        self.tabla_bairstow.grid(row=0, column=0, sticky='nsew')
        scrollbar_y.grid(row=0, column=1, sticky='ns')
        scrollbar_x.grid(row=1, column=0, sticky='ew')
        tabla_frame.rowconfigure(0, weight=1)
        tabla_frame.columnconfigure(0, weight=1)

        ttk.Label(
            pestana_desarrollo,
            text="Selecciona una iteración en la pestaña Iteraciones para consultar sus cálculos.",
        ).pack(anchor='w', padx=8, pady=(8, 3))
        detalle_frame = ttk.Frame(pestana_desarrollo)
        detalle_frame.pack(fill='both', expand=True, padx=5, pady=5)
        self.detalle_texto_bairstow = tk.Text(
            detalle_frame, height=12, wrap='word', font=('Consolas', 9), state='disabled'
        )
        detalle_scroll = ttk.Scrollbar(
            detalle_frame, orient='vertical', command=self.detalle_texto_bairstow.yview
        )
        self.detalle_texto_bairstow.configure(yscrollcommand=detalle_scroll.set)
        self.detalle_texto_bairstow.pack(side='left', fill='both', expand=True)
        detalle_scroll.pack(side='right', fill='y')
        self.tabla_bairstow.bind('<<TreeviewSelect>>', self.mostrar_detalle_bairstow)

    def actualizar_coeficientes_bairstow(self):
        try:
            grado = self.grado_bairstow.get()
        except tk.TclError:
            messagebox.showerror("Grado inválido", "El grado debe ser un entero entre 2 y 12.")
            return
        if not 2 <= grado <= 12:
            messagebox.showerror("Grado inválido", "El grado debe estar entre 2 y 12.")
            return

        valores = [entrada.get() for entrada in self.entradas_coef_bairstow]
        cantidad = grado + 1
        if valores:
            if cantidad > len(valores):
                nuevos_grados = cantidad - len(valores)
                valores = ['1'] + ['0'] * (nuevos_grados - 1) + valores
            else:
                valores = valores[-cantidad:]
        elif grado == 4:
            valores = ['1', '-3', '3', '-3', '2']
        else:
            valores = ['1'] + ['0'] * grado

        for widget in self.frame_coef_bairstow.winfo_children():
            widget.destroy()
        self.entradas_coef_bairstow = []

        for indice, valor in enumerate(valores):
            potencia = grado - indice
            campo = ttk.Frame(self.frame_coef_bairstow)
            campo.grid(row=indice // 6, column=indice % 6, sticky='w', padx=(0, 8), pady=3)
            ttk.Label(campo, text=f"a{potencia}:").pack(side='left', padx=(0, 3))
            entrada = ttk.Entry(campo, width=9)
            entrada.insert(0, valor)
            entrada.pack(side='left')
            self.entradas_coef_bairstow.append(entrada)

    def formatear_raiz_bairstow(self, raiz):
        raiz = complex(raiz)
        if abs(raiz.imag) < 1e-10:
            return f"{raiz.real:.10g}"
        signo = '+' if raiz.imag >= 0 else '-'
        return f"{raiz.real:.10g} {signo} {abs(raiz.imag):.10g}i"

    def mostrar_detalle_bairstow(self, _evento=None):
        seleccion = self.tabla_bairstow.selection()
        if not seleccion:
            return

        detalle = self.historial_bairstow[int(seleccion[0])]
        polinomio = ', '.join(f"{valor:.10g}" for valor in detalle["polinomio"])
        coeficientes_b = ', '.join(f"{valor:.10g}" for valor in detalle["b"])
        cociente = ', '.join(f"{valor:.10g}" for valor in detalle["cociente"])
        lineas = [
            f"Etapa {detalle['etapa']}, iteración {detalle['iteracion']}",
            f"Polinomio actual (coeficientes descendentes): [{polinomio}]",
            f"Factor candidato: x^2 - ({detalle['r']:.10g})x - ({detalle['s']:.10g})",
            "División sintética: b0=a0; b1=a1+r*b0; bi=ai+r*b(i-1)+s*b(i-2)",
            f"Coeficientes calculados b: [{coeficientes_b}]",
            f"Cociente provisional (coeficientes): [{cociente}]",
            f"Residuos: R1={detalle['residuo_x']:.10g}; R0={detalle['residuo_constante']:.10g}",
            f"Criterio: ||R||inf=max(|R1|,|R0|)={detalle['norma']:.6e}",
        ]

        if detalle["convergio"]:
            lineas.append("Convergencia: la norma del residuo alcanzó la tolerancia; se acepta este factor.")
        else:
            jacobiano = detalle["jacobiano"]
            correcciones = detalle["correcciones"]
            lineas.extend([
                "Jacobiano J = d(R1, R0)/d(r, s):",
                f"  [[{jacobiano[0, 0]:.8g}, {jacobiano[0, 1]:.8g}],",
                f"   [{jacobiano[1, 0]:.8g}, {jacobiano[1, 1]:.8g}]]",
                "Sistema de corrección: J * [Δr, Δs] = -[R1, R0]",
                f"Corrección Newton: Δr={correcciones[0]:.10g}; Δs={correcciones[1]:.10g}",
                f"Actualización: r_nuevo={detalle['r_nuevo']:.10g}; s_nuevo={detalle['s_nuevo']:.10g}",
            ])

        self.detalle_texto_bairstow.configure(state='normal')
        self.detalle_texto_bairstow.delete('1.0', tk.END)
        self.detalle_texto_bairstow.insert('1.0', '\n'.join(lineas))
        self.detalle_texto_bairstow.configure(state='disabled')

    def resolver_polinomio_bairstow(self):
        try:
            coeficientes = [float(entrada.get()) for entrada in self.entradas_coef_bairstow]
            raices, historial, factores = resolver_bairstow(
                coeficientes,
                self.iteraciones_bairstow.get(),
                self.tolerancia_bairstow.get(),
                self.r_bairstow.get(),
                self.s_bairstow.get(),
            )
        except (ValueError, tk.TclError, np.linalg.LinAlgError) as error:
            messagebox.showerror("No se pudo resolver", str(error))
            return

        for fila in self.tabla_bairstow.get_children():
            self.tabla_bairstow.delete(fila)
        self.historial_bairstow = historial
        for indice, detalle in enumerate(historial):
            correcciones = detalle["correcciones"]
            delta_r = '-' if correcciones is None else f"{correcciones[0]:.3e}"
            delta_s = '-' if correcciones is None else f"{correcciones[1]:.3e}"
            self.tabla_bairstow.insert('', 'end', iid=str(indice), values=(
                detalle["etapa"], detalle["iteracion"], f"{detalle['r']:.8g}",
                f"{detalle['s']:.8g}", f"{detalle['residuo_x']:.3e}",
                f"{detalle['residuo_constante']:.3e}", delta_r, delta_s,
                f"{detalle['r_nuevo']:.8g}", f"{detalle['s_nuevo']:.8g}",
                f"{detalle['norma']:.3e}"
            ))

        if historial:
            self.tabla_bairstow.selection_set('0')
            self.mostrar_detalle_bairstow()

        texto_raices = '\n'.join(
            f"x{indice + 1} = {self.formatear_raiz_bairstow(raiz)}"
            for indice, raiz in enumerate(raices)
        )
        texto_factores = ' * '.join(f"({factor})" for factor in factores)
        self.resultado_bairstow.config(
            text=f"Factorización aproximada: {texto_factores}\nRaíces aproximadas:\n{texto_raices}"
        )
        self.notebook_resultados_bairstow.select(self.pestana_iteraciones_bairstow)
        
    def crear_pestana_entrada(self):
        frame = ttk.Frame(self.pestana_entrada)
        frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Título
        titulo = ttk.Label(frame, text="Ingreso de Sistema de Ecuaciones", 
                          style='Header.TLabel', padding=15)
        titulo.pack(fill='x', pady=(0, 20))
        
        # Panel de configuración
        config_frame = ttk.LabelFrame(frame, text="Configuración", padding=15)
        config_frame.pack(fill='x', pady=(0, 15))
        
        # Número de variables
        ttk.Label(config_frame, text="Número de variables:").grid(row=0, column=0, sticky='w', padx=5)
        spin_n = ttk.Spinbox(config_frame, from_=1, to=10, textvariable=self.n_var, 
                            command=self.actualizar_entradas, width=10)
        spin_n.grid(row=0, column=1, padx=5, pady=5)
        
        # Método
        ttk.Label(config_frame, text="Método:").grid(row=0, column=2, sticky='w', padx=5)
        metodo_combo = ttk.Combobox(config_frame, textvariable=self.metodo, 
                                   values=["jacobi", "seidel"], state="readonly", width=15)
        metodo_combo.grid(row=0, column=3, padx=5, pady=5)
        
        # Iteraciones
        ttk.Label(config_frame, text="Iteraciones máximas:").grid(row=1, column=0, sticky='w', padx=5)
        spin_iter = ttk.Spinbox(config_frame, from_=1, to=100, textvariable=self.iteraciones, width=10)
        spin_iter.grid(row=1, column=1, padx=5, pady=5)
        
        # Tolerancia
        ttk.Label(config_frame, text="Tolerancia:").grid(row=1, column=2, sticky='w', padx=5)
        entry_tol = ttk.Entry(config_frame, textvariable=self.tolerancia, width=15)
        entry_tol.grid(row=1, column=3, padx=5, pady=5)
        
        # Opciones de visualización
        ttk.Label(config_frame, text="Tipo de gráfica:").grid(row=2, column=0, sticky='w', padx=5)
        grafica_combo = ttk.Combobox(config_frame, textvariable=self.tipo_grafica, 
                                   values=["lineas", "barras", "scatter", "heatmap"], state="readonly", width=15)
        grafica_combo.grid(row=2, column=1, padx=5, pady=5)
        
        ttk.Label(config_frame, text="Tema:").grid(row=2, column=2, sticky='w', padx=5)
        tema_combo = ttk.Combobox(config_frame, textvariable=self.tema, 
                                values=["profesional", "oscuro", "colorido", "minimalista"], state="readonly", width=15)
        tema_combo.grid(row=2, column=3, padx=5, pady=5)
        
        # Exportar CSV y animación
        ttk.Checkbutton(config_frame, text="Exportar resultados a CSV", 
                       variable=self.exportar_csv).grid(row=3, column=0, columnspan=2, sticky='w', pady=5)
        ttk.Checkbutton(config_frame, text="Mostrar animación de convergencia", 
                       variable=self.mostrar_animacion).grid(row=3, column=2, columnspan=2, sticky='w', pady=5)
        
        # Panel de matriz A
        matriz_frame = ttk.LabelFrame(frame, text="Matriz A (Coeficientes)", padding=15)
        matriz_frame.pack(fill='both', expand=True, pady=(0, 15))
        
        self.frame_matriz = ttk.Frame(matriz_frame)
        self.frame_matriz.pack(fill='both', expand=True)
        
        # Panel de vector b
        vector_frame = ttk.LabelFrame(frame, text="Vector b (Términos independientes)", padding=15)
        vector_frame.pack(fill='x', pady=(0, 15))
        
        self.frame_vector = ttk.Frame(vector_frame)
        self.frame_vector.pack(fill='x')
        
        # Botones
        boton_frame = ttk.Frame(frame)
        boton_frame.pack(fill='x', pady=(10, 0))
        
        ttk.Button(boton_frame, text="🔄 Actualizar Matriz", 
                  command=self.actualizar_entradas).pack(side='left', padx=5)
        ttk.Button(boton_frame, text="🧮 Resolver Sistema", 
                  command=self.resolver_sistema).pack(side='left', padx=5)
        ttk.Button(boton_frame, text="🧹 Limpiar", 
                  command=self.limpiar_entradas).pack(side='left', padx=5)
        
        # Inicializar entradas
        self.actualizar_entradas()
        
    def crear_pestana_ejercicios(self):
        frame = ttk.Frame(self.pestana_ejercicios)
        frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Título
        titulo = ttk.Label(frame, text="Ejercicios Precargados", 
                          style='Header.TLabel', padding=15)
        titulo.pack(fill='x', pady=(0, 20))
        
        # Lista de ejercicios
        self.ejercicios = {
            1: {
                "A": np.array([[4, -1, 1], [-1, 5, -2], [1, 2, 6]], dtype=float), 
                "b": np.array([7, 12, 23], dtype=float), 
                "tipo": "jacobi", "nombre": "Ejercicio 1 (Jacobi 3x3)"
            },
            2: {
                "A": np.array([[8, 1, -1], [2, 9, 2], [1, -3, 10]], dtype=float), 
                "b": np.array([8, 13, 18], dtype=float), 
                "tipo": "jacobi", "nombre": "Ejercicio 2 (Jacobi 3x3)"
            },
            3: {
                "A": np.array([[10, -1, 2, 0], [-1, 11, -1, 3], [2, -1, 10, -1], [0, 3, -1, 8]], dtype=float), 
                "b": np.array([6, 25, -11, 15], dtype=float), 
                "tipo": "jacobi", "nombre": "Ejercicio 3 (Jacobi 4x4)"
            },
            4: {
                "A": np.array([[5, 1, -1], [1, 4, 1], [2, -1, 5]], dtype=float), 
                "b": np.array([6, 10, 5], dtype=float), 
                "tipo": "seidel", "nombre": "Ejercicio 4 (Gauss-Seidel 3x3)"
            },
            5: {
                "A": np.array([[6, 2, -1], [1, 5, 1], [2, 1, 7]], dtype=float), 
                "b": np.array([18, 20, 25], dtype=float), 
                "tipo": "seidel", "nombre": "Ejercicio 5 (Gauss-Seidel 3x3)"
            },
            6: {
                "A": np.array([[7, -1, 2], [2, 8, -1], [-1, 2, 9]], dtype=float), 
                "b": np.array([14, 15, 22], dtype=float), 
                "tipo": "seidel", "nombre": "Ejercicio 6 (Gauss-Seidel 3x3)"
            }
        }
        
        # Crear botones para cada ejercicio
        for i in range(1, 4):
            row_frame = ttk.Frame(frame)
            row_frame.pack(fill='x', pady=5)
            
            for j in range(1, 3):
                idx = (i-1)*2 + j
                if idx <= 6:
                    ej = self.ejercicios[idx]
                    btn = ttk.Button(row_frame, text=f"{ej['nombre']}", 
                                   command=lambda x=idx: self.cargar_ejercicio(x))
                    btn.pack(side='left', fill='x', expand=True, padx=5)
        
        # Botón comparación
        comp_frame = ttk.Frame(frame)
        comp_frame.pack(fill='x', pady=20)
        ttk.Button(comp_frame, text="⚖️ Comparar Métodos (Jacobi vs Seidel)", 
                  command=self.comparar_metodos).pack(fill='x', padx=50)
        
    def crear_pestana_resultados(self):
        # Crear canvas con scroll
        canvas = tk.Canvas(self.pestana_resultados)
        scrollbar = ttk.Scrollbar(self.pestana_resultados, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        frame = ttk.Frame(scrollable_frame)
        frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Título
        titulo = ttk.Label(frame, text="Resultados", 
                          style='Header.TLabel', padding=15)
        titulo.pack(fill='x', pady=(0, 20))
        
        # Panel de información
        info_frame = ttk.LabelFrame(frame, text="Información del Sistema", padding=15)
        info_frame.pack(fill='x', pady=(0, 15))
        
        self.info_label = ttk.Label(info_frame, text="Seleccione un ejercicio o ingrese datos")
        self.info_label.pack(fill='x')
        
        # Panel de tabla
        tabla_frame = ttk.LabelFrame(frame, text="Tabla de Iteraciones", padding=15)
        tabla_frame.pack(fill='x', pady=(0, 15))
        
        # Treeview para la tabla. Las columnas de las variables se ajustan al sistema.
        self.tabla_frame = tabla_frame
        self.configurar_tabla_iteraciones(3)
        
        # Panel de gráfica
        grafica_frame = ttk.LabelFrame(frame, text="Gráfica de Convergencia", padding=15)
        grafica_frame.pack(fill='both', expand=True, pady=(0, 10))
        
        # Opciones de gráfica
        grafica_ops_frame = ttk.Frame(grafica_frame)
        grafica_ops_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Button(grafica_ops_frame, text="🔄 Refrescar Gráfica", 
                  command=self.refrescar_grafica).pack(side='left', padx=5)
        ttk.Button(grafica_ops_frame, text="📊 Ver Evolución de Errores", 
                  command=self.mostrar_comparacion_errores).pack(side='left', padx=5)
        ttk.Button(grafica_ops_frame, text="🎨 Cambiar Estilo", 
                  command=self.cambiar_estilo_grafica).pack(side='left', padx=5)
        
        self.canvas_frame = ttk.Frame(grafica_frame, height=400)
        self.canvas_frame.pack(fill='both', expand=True)
        
        # Botones de exportación
        export_frame = ttk.Frame(frame)
        export_frame.pack(fill='x', pady=(10, 0))
        
        ttk.Button(export_frame, text="📥 Exportar CSV", 
                  command=self.exportar_csv_manual).pack(side='left', padx=5)
        ttk.Button(export_frame, text="📥 Guardar Gráfica", 
                  command=self.guardar_grafica).pack(side='left', padx=5)

    def configurar_tabla_iteraciones(self, cantidad_variables):
        """Actualiza las columnas de la tabla según el tamaño de la matriz."""
        columnas = (
            ['Iteración']
            + [f'x{i + 1}' for i in range(cantidad_variables)]
            + ['Error Absoluto', 'Error Relativo']
        )

        if hasattr(self, 'tree'):
            self.tree.destroy()
        if hasattr(self, 'scrollbar_tabla'):
            self.scrollbar_tabla.destroy()

        self.tree = ttk.Treeview(self.tabla_frame, columns=columnas, show='headings', height=8)
        for columna in columnas:
            self.tree.heading(columna, text=columna)
            ancho = 90 if columna.startswith('Error') else 80
            self.tree.column(columna, width=ancho, anchor='center')

        self.scrollbar_tabla = ttk.Scrollbar(
            self.tabla_frame, orient='vertical', command=self.tree.yview
        )
        self.tree.configure(yscrollcommand=self.scrollbar_tabla.set)
        self.tree.pack(side='left', fill='both', expand=True)
        self.scrollbar_tabla.pack(side='right', fill='y')

    def actualizar_entradas(self):
        # Limpiar entradas anteriores
        for widget in self.frame_matriz.winfo_children():
            widget.destroy()
        for widget in self.frame_vector.winfo_children():
            widget.destroy()
        
        self.entradas_A = []
        self.entradas_b = []
        
        n = self.n_var.get()
        
        # Crear matriz de entradas
        for i in range(n):
            fila = []
            for j in range(n):
                entry = ttk.Entry(self.frame_matriz, width=8)
                entry.grid(row=i, column=j, padx=2, pady=2)
                fila.append(entry)
            self.entradas_A.append(fila)
        
        # Crear vector de entradas
        for i in range(n):
            entry = ttk.Entry(self.frame_vector, width=8)
            entry.grid(row=0, column=i, padx=5, pady=5)
            self.entradas_b.append(entry)
        
    def limpiar_entradas(self):
        for fila in self.entradas_A:
            for entry in fila:
                entry.delete(0, tk.END)
        for entry in self.entradas_b:
            entry.delete(0, tk.END)
            
    def obtener_datos(self):
        try:
            n = self.n_var.get()
            A = np.zeros((n, n))
            b = np.zeros(n)
            
            for i in range(n):
                for j in range(n):
                    valor = self.entradas_A[i][j].get()
                    if not valor.strip():
                        raise ValueError(f"Falta el valor A[{i + 1}][{j + 1}].")
                    A[i][j] = float(valor)
            
            for i in range(n):
                valor = self.entradas_b[i].get()
                if not valor.strip():
                    raise ValueError(f"Falta el valor b[{i + 1}].")
                b[i] = float(valor)

            self.validar_problema(A, b, self.iteraciones.get(), self.tolerancia.get())
            
            return A, b
        except (ValueError, tk.TclError) as error:
            messagebox.showerror("Datos inválidos", str(error) or "Ingrese valores numéricos válidos.")
            return None, None
            
    def verificar_diagonal_dominante(self, A):
        n = len(A)
        for i in range(n):
            suma = sum(abs(A[i][j]) for j in range(n) if i != j)
            if abs(A[i][i]) <= suma:
                return False
        return True
        
    def calcular_error(self, x_new, x_old):
        error_abs = np.linalg.norm(x_new - x_old)
        error_rel = error_abs / (np.linalg.norm(x_new) + 1e-10)
        return error_abs, error_rel

    def validar_problema(self, A, b, iteraciones, tolerancia):
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float)

        if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] == 0:
            raise ValueError("La matriz A debe ser cuadrada y no estar vacía.")
        if b.ndim != 1 or len(b) != A.shape[0]:
            raise ValueError("El vector b debe tener el mismo número de filas que A.")
        if not np.isfinite(A).all() or not np.isfinite(b).all():
            raise ValueError("La matriz y el vector solo pueden contener valores finitos.")
        if np.any(np.isclose(np.diag(A), 0.0)):
            raise ValueError("La diagonal de A no puede contener ceros.")
        if not isinstance(iteraciones, (int, np.integer)) or iteraciones < 1:
            raise ValueError("El número de iteraciones debe ser un entero mayor que cero.")
        if not np.isfinite(tolerancia) or tolerancia <= 0:
            raise ValueError("La tolerancia debe ser un número mayor que cero.")

        return A, b
        
    def formatear_numero(self, num):
        return f"{num:.6f}"
        
    def jacobi(self, A, b, iteraciones, tolerancia):
        A, b = self.validar_problema(A, b, iteraciones, tolerancia)
        n = len(b)
        x = np.zeros(n)
        historial = [x.copy()]
        errores = []
        
        for k in range(iteraciones):
            x_new = np.zeros(n)
            for i in range(n):
                suma = 0
                for j in range(n):
                    if i != j:
                        suma += A[i][j] * x[j]
                x_new[i] = (b[i] - suma) / A[i][i]
            
            error_abs, error_rel = self.calcular_error(x_new, x)
            errores.append((error_abs, error_rel))
            
            x = x_new.copy()
            historial.append(x.copy())
            
            if error_abs < tolerancia:
                break
                
        return x, historial, errores
        
    def gauss_seidel(self, A, b, iteraciones, tolerancia):
        A, b = self.validar_problema(A, b, iteraciones, tolerancia)
        n = len(b)
        x = np.zeros(n)
        historial = [x.copy()]
        errores = []
        
        for k in range(iteraciones):
            x_old = x.copy()
            for i in range(n):
                suma = 0
                for j in range(n):
                    if i != j:
                        suma += A[i][j] * x[j]
                x[i] = (b[i] - suma) / A[i][i]
            
            error_abs, error_rel = self.calcular_error(x, x_old)
            errores.append((error_abs, error_rel))
            
            historial.append(x.copy())
            
            if error_abs < tolerancia:
                break
                
        return x, historial, errores
        
    def resolver_sistema(self):
        A, b = self.obtener_datos()
        if A is None:
            return
            
        try:
            metodo = self.metodo.get()
            iteraciones = self.iteraciones.get()
            tolerancia = self.tolerancia.get()

            if self.verificar_diagonal_dominante(A):
                convergencia = "[OK] La matriz es diagonalmente dominante"
            else:
                convergencia = "[!] La matriz NO es diagonalmente dominante"

            if metodo == "jacobi":
                x, historial, errores = self.jacobi(A, b, iteraciones, tolerancia)
                nombre_metodo = "Gauss-Jacobi"
            else:
                x, historial, errores = self.gauss_seidel(A, b, iteraciones, tolerancia)
                nombre_metodo = "Gauss-Seidel"
        except (ValueError, tk.TclError) as error:
            messagebox.showerror("Datos inválidos", str(error))
            return
        
        # Mostrar resultados
        self.mostrar_resultados(A, b, x, historial, errores, nombre_metodo, convergencia)
        
        # Exportar CSV si está activado
        if self.exportar_csv.get():
            self.exportar_resultados_csv(historial, errores, nombre_metodo)
            
    def cargar_ejercicio(self, num):
        ej = self.ejercicios[num]
        
        # Cambiar a pestaña de entrada
        self.notebook.select(0)
        
        # Configurar parámetros
        n = len(ej["b"])
        self.n_var.set(n)
        self.metodo.set(ej["tipo"])
        self.actualizar_entradas()
        
        # Llenar matriz A
        for i in range(n):
            for j in range(n):
                self.entradas_A[i][j].delete(0, tk.END)
                self.entradas_A[i][j].insert(0, str(ej["A"][i][j]))
        
        # Llenar vector b
        for i in range(n):
            self.entradas_b[i].delete(0, tk.END)
            self.entradas_b[i].insert(0, str(ej["b"][i]))
            
    def comparar_metodos(self):
        # Usar el primer ejercicio para comparación
        ej = self.ejercicios[1]
        A, b = ej["A"], ej["b"]
        try:
            iteraciones = self.iteraciones.get()
            tolerancia = self.tolerancia.get()
            x_jacobi, hist_jacobi, err_jacobi = self.jacobi(A, b, iteraciones, tolerancia)
            x_seidel, hist_seidel, err_seidel = self.gauss_seidel(A, b, iteraciones, tolerancia)
        except (ValueError, tk.TclError) as error:
            messagebox.showerror("Datos inválidos", str(error))
            return
        
        # Mostrar comparación
        self.mostrar_comparacion(x_jacobi, x_seidel, err_jacobi, err_seidel)
        
    def mostrar_resultados(self, A, b, x, historial, errores, nombre_metodo, convergencia):
        self.modo_resultados = "metodo"
        self.configurar_tabla_iteraciones(len(b))
        # Cambiar a pestaña de resultados
        self.notebook.select(2)
        
        # Mostrar información
        n = len(b)
        sistema_str = "Sistema:\n"
        for i in range(n):
            ecuacion = " + ".join([f"({A[i][j]})x{j+1}" for j in range(n)])
            sistema_str += f"  {ecuacion} = {b[i]}\n"
        
        resultado_str = f"\nMétodo: {nombre_metodo}\n{convergencia}\n\n"
        resultado_str += "Solución final:\n"
        for i in range(n):
            resultado_str += f"  x{i+1} = {self.formatear_numero(x[i])}\n"
        
        if errores:
            resultado_str += f"\nError final:\n"
            resultado_str += f"  Absoluto: {self.formatear_numero(errores[-1][0])}\n"
            resultado_str += f"  Relativo: {self.formatear_numero(errores[-1][1])}\n"
        
        self.info_label.config(text=sistema_str + resultado_str)
        
        # Limpiar tabla
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Llenar tabla
        for i, (x_val, (err_abs, err_rel)) in enumerate(zip(historial, [(0, 0)] + errores)):
            fila = [i] + [self.formatear_numero(val) for val in x_val] + [self.formatear_numero(err_abs), self.formatear_numero(err_rel)]
            self.tree.insert('', 'end', values=fila)
        
        # Crear gráfica
        self.crear_grafica(historial, nombre_metodo)
        
        # Guardar datos para exportación
        self.ultimo_historial = historial
        self.ultimo_errores = errores
        self.ultimo_metodo = nombre_metodo
        self.ultimo_titulo = nombre_metodo
        
        # Mostrar animación si está activado
        if self.mostrar_animacion.get():
            self.mostrar_animacion_convergencia(historial, nombre_metodo)
        
    def obtener_configuracion_tema(self):
        """Retorna la configuración de colores según el tema seleccionado."""
        tema = self.tema.get()
        
        temas = {
            "profesional": {
                "bg_color": '#f0f0f0',
                "fig_bg": '#ffffff',
                "grid_color": '#e0e0e0',
                "text_color": '#333333',
                "colores": ['#FF4B4B', '#4B8BFF', '#28A745', '#FFC107', '#9B59B6', '#E67E22']
            },
            "oscuro": {
                "bg_color": '#2c3e50',
                "fig_bg": '#34495e',
                "grid_color": '#5d6d7e',
                "text_color": '#ecf0f1',
                "colores": ['#e74c3c', '#3498db', '#2ecc71', '#f1c40f', '#9b59b6', '#e67e22']
            },
            "colorido": {
                "bg_color": '#ffeaa7',
                "fig_bg": '#ffffff',
                "grid_color": '#fab1a0',
                "text_color": '#2d3436',
                "colores": ['#fd79a8', '#0984e3', '#00b894', '#fdcb6e', '#6c5ce7', '#e17055']
            },
            "minimalista": {
                "bg_color": '#ffffff',
                "fig_bg": '#ffffff',
                "grid_color": '#dfe6e9',
                "text_color": '#2d3436',
                "colores": ['#2d3436', '#636e72', '#b2bec3', '#dfe6e9', '#74b9ff', '#a29bfe']
            }
        }
        
        return temas.get(tema, temas["profesional"])
    
    def crear_grafica(self, historial, titulo):
        # Limpiar gráfica anterior
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()
        
        # Forzar actualización del frame
        self.canvas_frame.update_idletasks()
        
        historial_arr = np.array(historial)
        num_vars = historial_arr.shape[1]
        iteraciones = historial_arr.shape[0] - 1
        
        print(f"Creando gráfica: {titulo}, variables: {num_vars}, iteraciones: {iteraciones}")
        
        try:
            # Obtener configuración del tema
            tema_config = self.obtener_configuracion_tema()
            tipo_grafica = self.tipo_grafica.get()
            
            # Crear figura con tamaño adecuado
            fig = Figure(figsize=(12, 6), dpi=100)
            fig.patch.set_facecolor(tema_config["fig_bg"])
            
            if tipo_grafica == "heatmap" and num_vars <= 4:
                # Crear heatmap para sistemas pequeños
                ax = fig.add_subplot(111)
                im = ax.imshow(historial_arr.T, cmap='viridis', aspect='auto', interpolation='nearest')
                ax.set_xticks(range(iteraciones + 1))
                ax.set_yticks(range(num_vars))
                ax.set_yticklabels([f'x{i+1}' for i in range(num_vars)])
                ax.set_xlabel('Iteración', color=tema_config["text_color"])
                ax.set_ylabel('Variable', color=tema_config["text_color"])
                ax.set_title(f'Heatmap de Convergencia - {titulo}', fontsize=14, fontweight='bold', color=tema_config["text_color"])
                fig.colorbar(im, ax=ax, label='Valor')
                
            elif tipo_grafica == "barras":
                # Crear gráfica de barras
                ax = fig.add_subplot(111)
                x = np.arange(iteraciones + 1)
                width = 0.8 / num_vars
                
                for i in range(num_vars):
                    offset = (i - num_vars/2 + 0.5) * width
                    ax.bar(x + offset, historial_arr[:, i], width, 
                          label=f'x{i+1}', color=tema_config["colores"][i % len(tema_config["colores"])], alpha=0.8)
                
                ax.set_title(f'Convergencia de Variables (Barras) - {titulo}', fontsize=14, fontweight='bold', color=tema_config["text_color"])
                ax.set_xlabel('Número de Iteración', fontsize=11, fontweight='bold', color=tema_config["text_color"])
                ax.set_ylabel('Valor calculado', fontsize=11, fontweight='bold', color=tema_config["text_color"])
                ax.set_xticks(range(iteraciones + 1))
                ax.grid(True, linestyle='--', alpha=0.3, color=tema_config["grid_color"])
                ax.legend()
                
            elif tipo_grafica == "scatter":
                # Crear gráfica de dispersión
                ax = fig.add_subplot(111)
                for i in range(num_vars):
                    color = tema_config["colores"][i % len(tema_config["colores"])]
                    ax.scatter(range(iteraciones + 1), historial_arr[:, i], 
                             s=100, color=color, label=f'x{i+1}', alpha=0.7, edgecolors='black')
                
                ax.set_title(f'Convergencia de Variables (Scatter) - {titulo}', fontsize=14, fontweight='bold', color=tema_config["text_color"])
                ax.set_xlabel('Número de Iteración', fontsize=11, fontweight='bold', color=tema_config["text_color"])
                ax.set_ylabel('Valor calculado', fontsize=11, fontweight='bold', color=tema_config["text_color"])
                ax.set_xticks(range(iteraciones + 1))
                ax.grid(True, linestyle='--', alpha=0.3, color=tema_config["grid_color"])
                ax.legend()
                
            else:  # líneas (default)
                # Crear gráfica de líneas original
                ax = fig.add_subplot(111)
                for i in range(num_vars):
                    color = tema_config["colores"][i % len(tema_config["colores"])]
                    ax.plot(range(iteraciones + 1), historial_arr[:, i], marker='o', 
                           linewidth=2.5, color=color, label=f'x{i+1}', markersize=8, alpha=0.8)
                
                ax.set_title(f'Convergencia de Variables - {titulo}', fontsize=14, fontweight='bold', color=tema_config["text_color"])
                ax.set_xlabel('Número de Iteración', fontsize=11, fontweight='bold', color=tema_config["text_color"])
                ax.set_ylabel('Valor calculado', fontsize=11, fontweight='bold', color=tema_config["text_color"])
                ax.set_xticks(range(iteraciones + 1))
                ax.grid(True, linestyle='--', alpha=0.3, color=tema_config["grid_color"])
                ax.legend()
            
            # Configurar colores de los ejes
            ax.tick_params(colors=tema_config["text_color"])
            for spine in ax.spines.values():
                spine.set_edgecolor(tema_config["text_color"])
            
            fig.tight_layout()
            
            # Crear canvas con toolbar de navegación
            canvas_frame = ttk.Frame(self.canvas_frame)
            canvas_frame.pack(fill='both', expand=True)
            
            canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
            canvas.draw()
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(fill='both', expand=True)
            
            # Agregar toolbar de navegación (zoom, pan, etc.)
            toolbar = NavigationToolbar2Tk(canvas, canvas_frame)
            toolbar.update()
            toolbar.pack(fill='x')
            
            # Forzar actualización del widget
            self.canvas_frame.update_idletasks()
            
            self.ultima_figura = fig
            self.ultimo_canvas = canvas
            self.ultimo_historial = historial
            self.ultimo_titulo = titulo
            
            print("Gráfica creada exitosamente")
            
        except Exception as e:
            print(f"Error al crear gráfica: {e}")
            import traceback
            traceback.print_exc()
            # Mostrar mensaje de error en el frame
            error_label = ttk.Label(self.canvas_frame, text=f"Error al crear gráfica: {e}", 
                                   foreground='red', font=('Arial', 10))
            error_label.pack(fill='both', expand=True)
        
    def mostrar_comparacion(self, x_jacobi, x_seidel, err_jacobi, err_seidel):
        self.modo_resultados = "comparacion"
        self.ultimo_historial = None
        self.ultimo_errores = None
        self.ultimo_metodo = None
        self.notebook.select(2)
        
        n = len(x_jacobi)
        comparacion_str = "COMPARACIÓN DE MÉTODOS\n\n"
        comparacion_str += f"{'Variable':<12} {'Gauss-Jacobi':<20} {'Gauss-Seidel':<20} {'Diferencia':<15}\n"
        comparacion_str += "-" * 70 + "\n"
        
        for i in range(n):
            diff = abs(x_jacobi[i] - x_seidel[i])
            comparacion_str += f"x{i+1:<10} {self.formatear_numero(x_jacobi[i]):<20} {self.formatear_numero(x_seidel[i]):<20} {self.formatear_numero(diff):<15}\n"
        
        comparacion_str += f"\nIteraciones Jacobi: {len(err_jacobi)} | Iteraciones Seidel: {len(err_seidel)}\n"
        comparacion_str += f"Error Final Jacobi: {self.formatear_numero(err_jacobi[-1][0])} | Error Final Seidel: {self.formatear_numero(err_seidel[-1][0])}"
        
        self.info_label.config(text=comparacion_str)
        
        # Crear gráfica comparativa
        self.crear_grafica_comparativa(err_jacobi, err_seidel)
        
    def crear_grafica_comparativa(self, errores_jacobi, errores_seidel):
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        iter_jacobi = range(1, len(errores_jacobi) + 1)
        iter_seidel = range(1, len(errores_seidel) + 1)
        
        # Error absoluto
        ax1.plot(iter_jacobi, [e[0] for e in errores_jacobi], 'o-', color='#FF4B4B', 
                linewidth=2.5, markersize=8, label='Gauss-Jacobi')
        ax1.plot(iter_seidel, [e[0] for e in errores_seidel], 's-', color='#4B8BFF', 
                linewidth=2.5, markersize=8, label='Gauss-Seidel')
        ax1.set_xlabel('Iteración', fontweight='bold')
        ax1.set_ylabel('Error Absoluto', fontweight='bold')
        ax1.set_title('Error Absoluto', fontweight='bold')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()
        ax1.set_yscale('log')
        
        # Error relativo
        ax2.plot(iter_jacobi, [e[1] for e in errores_jacobi], 'o-', color='#FF4B4B', 
                linewidth=2.5, markersize=8, label='Gauss-Jacobi')
        ax2.plot(iter_seidel, [e[1] for e in errores_seidel], 's-', color='#4B8BFF', 
                linewidth=2.5, markersize=8, label='Gauss-Seidel')
        ax2.set_xlabel('Iteración', fontweight='bold')
        ax2.set_ylabel('Error Relativo', fontweight='bold')
        ax2.set_title('Error Relativo', fontweight='bold')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend()
        ax2.set_yscale('log')
        
        plt.suptitle('Comparación de Métodos', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
        self.ultima_figura = fig
        
    def exportar_resultados_csv(self, historial, errores, titulo):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        nombre_archivo = f"resultados_{titulo.replace(' ', '_')}_{timestamp}.csv"
        
        try:
            with open(nombre_archivo, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile, delimiter=';')
                
                n = len(historial[0])
                headers = ['Iteración'] + [f'x{i+1}' for i in range(n)] + ['Error Absoluto', 'Error Relativo']
                writer.writerow(headers)
                
                for i, (x, (err_abs, err_rel)) in enumerate(zip(historial, [(0, 0)] + errores)):
                    fila = [i] + [float(val) for val in x] + [float(err_abs), float(err_rel)]
                    writer.writerow(fila)
            
            messagebox.showinfo("Exportación Exitosa", f"Resultados exportados a:\n{nombre_archivo}")
        except Exception as e:
            messagebox.showerror("Error", f"Error al exportar CSV: {e}")
            
    def exportar_csv_manual(self):
        if getattr(self, 'modo_resultados', None) == "metodo" and self.ultimo_historial is not None:
            self.exportar_resultados_csv(self.ultimo_historial, self.ultimo_errores, self.ultimo_metodo)
        elif getattr(self, 'modo_resultados', None) == "comparacion":
            messagebox.showwarning("Advertencia", "La comparación no tiene un único historial para exportar.")
        else:
            messagebox.showwarning("Advertencia", "No hay resultados para exportar")
            
    def guardar_grafica(self):
        if hasattr(self, 'ultima_figura'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            nombre_archivo = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
                initialfile=f"grafica_{timestamp}.png"
            )
            if nombre_archivo:
                self.ultima_figura.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Guardado Exitoso", f"Gráfica guardada como:\n{nombre_archivo}")
        else:
            messagebox.showwarning("Advertencia", "No hay gráfica para guardar")
    
    def refrescar_grafica(self):
        """Refresca la gráfica actual con la configuración actual."""
        if getattr(self, 'modo_resultados', None) == "metodo" and self.ultimo_historial is not None and hasattr(self, 'ultimo_titulo'):
            self.crear_grafica(self.ultimo_historial, self.ultimo_titulo)
        elif getattr(self, 'modo_resultados', None) == "comparacion":
            messagebox.showwarning("Advertencia", "La comparación actual no se puede refrescar con un solo método.")
        else:
            messagebox.showwarning("Advertencia", "No hay gráfica para refrescar")
    
    def mostrar_comparacion_errores(self):
        """Muestra una gráfica comparativa de errores."""
        if hasattr(self, 'ultimo_errores') and self.ultimo_errores:
            self.crear_grafica_comparacion_errores(self.ultimo_errores, self.ultimo_titulo)
        else:
            messagebox.showwarning("Advertencia", "No hay datos de errores para comparar")
    
    def crear_grafica_comparacion_errores(self, errores, titulo):
        """Crea gráfica comparativa de errores absolutos y relativos."""
        # Limpiar gráfica anterior
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()
        
        self.canvas_frame.update_idletasks()
        
        try:
            tema_config = self.obtener_configuracion_tema()
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            fig.patch.set_facecolor(tema_config["fig_bg"])
            
            iteraciones = range(1, len(errores) + 1)
            errores_abs = [e[0] for e in errores]
            errores_rel = [e[1] for e in errores]
            
            # Error absoluto
            ax1.plot(iteraciones, errores_abs, 'o-', color=tema_config["colores"][0], 
                    linewidth=2.5, markersize=8, label='Error Absoluto')
            ax1.set_xlabel('Iteración', fontweight='bold', color=tema_config["text_color"])
            ax1.set_ylabel('Error Absoluto', fontweight='bold', color=tema_config["text_color"])
            ax1.set_title('Convergencia - Error Absoluto', fontweight='bold', color=tema_config["text_color"])
            ax1.grid(True, linestyle='--', alpha=0.3, color=tema_config["grid_color"])
            ax1.legend()
            ax1.set_yscale('log')
            ax1.tick_params(colors=tema_config["text_color"])
            
            # Error relativo
            ax2.plot(iteraciones, errores_rel, 's-', color=tema_config["colores"][1], 
                    linewidth=2.5, markersize=8, label='Error Relativo')
            ax2.set_xlabel('Iteración', fontweight='bold', color=tema_config["text_color"])
            ax2.set_ylabel('Error Relativo', fontweight='bold', color=tema_config["text_color"])
            ax2.set_title('Convergencia - Error Relativo', fontweight='bold', color=tema_config["text_color"])
            ax2.grid(True, linestyle='--', alpha=0.3, color=tema_config["grid_color"])
            ax2.legend()
            ax2.set_yscale('log')
            ax2.tick_params(colors=tema_config["text_color"])
            
            plt.suptitle(f'Análisis de Errores - {titulo}', fontsize=14, fontweight='bold', color=tema_config["text_color"])
            plt.tight_layout()
            
            # Crear canvas con toolbar
            canvas_frame = ttk.Frame(self.canvas_frame)
            canvas_frame.pack(fill='both', expand=True)
            
            canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
            canvas.draw()
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(fill='both', expand=True)
            
            toolbar = NavigationToolbar2Tk(canvas, canvas_frame)
            toolbar.update()
            
            self.canvas_frame.update_idletasks()
            self.ultima_figura = fig
            self.ultimo_canvas = canvas
            
        except Exception as e:
            print(f"Error al crear gráfica comparativa: {e}")
            error_label = ttk.Label(self.canvas_frame, text=f"Error: {e}", foreground='red')
            error_label.pack(fill='both', expand=True)
    
    def cambiar_estilo_grafica(self):
        """Cambia entre diferentes estilos de gráfica."""
        estilos = ["lineas", "barras", "scatter", "heatmap"]
        estilo_actual = self.tipo_grafica.get()
        indice_actual = estilos.index(estilo_actual) if estilo_actual in estilos else 0
        nuevo_estilo = estilos[(indice_actual + 1) % len(estilos)]
        
        self.tipo_grafica.set(nuevo_estilo)
        self.refrescar_grafica()
        
        messagebox.showinfo("Estilo Cambiado", f"Estilo actual: {nuevo_estilo}")
    
    def mostrar_animacion_convergencia(self, historial, titulo):
        """Muestra una animación de la convergencia paso a paso."""
        try:
            # Crear ventana de animación
            anim_window = tk.Toplevel(self.root)
            anim_window.title(f"Animación de Convergencia - {titulo}")
            anim_window.geometry("800x500")
            
            canvas_anim = tk.Canvas(anim_window, bg='white')
            canvas_anim.pack(fill='both', expand=True)
            
            tema_config = self.obtener_configuracion_tema()
            historial_arr = np.array(historial)
            num_vars = historial_arr.shape[1]
            iteraciones = historial_arr.shape[0] - 1
            
            fig, ax = plt.subplots(figsize=(8, 5))
            fig.patch.set_facecolor(tema_config["fig_bg"])
            
            # Configurar ejes
            ax.set_xlim(0, iteraciones)
            y_min, y_max = historial_arr.min(), historial_arr.max()
            margin = max((y_max - y_min) * 0.1, 1.0)
            ax.set_ylim(y_min - margin, y_max + margin)
            ax.set_xlabel('Iteración', color=tema_config["text_color"])
            ax.set_ylabel('Valor', color=tema_config["text_color"])
            ax.set_title(f'Animación de Convergencia - {titulo}', color=tema_config["text_color"])
            ax.grid(True, linestyle='--', alpha=0.3)
            
            # Inicializar líneas
            lines = []
            colores = tema_config["colores"]
            for i in range(num_vars):
                line, = ax.plot([], [], marker='o', color=colores[i % len(colores)], 
                               label=f'x{i+1}', linewidth=2, markersize=6)
                lines.append(line)
            
            ax.legend()
            
            def animate(frame):
                for i, line in enumerate(lines):
                    line.set_data(range(frame + 1), historial_arr[:frame + 1, i])
                return lines
            
            anim = FuncAnimation(fig, animate, frames=iteraciones + 1, 
                              interval=500, blit=True, repeat=True)
            
            canvas_anim_fig = FigureCanvasTkAgg(fig, master=canvas_anim)
            canvas_anim_fig.draw()
            canvas_anim_fig.get_tk_widget().pack(fill='both', expand=True)
            
            # Botón para cerrar
            ttk.Button(anim_window, text="Cerrar Animación", 
                      command=anim_window.destroy).pack(pady=10)
            
        except Exception as e:
            print(f"Error al crear animación: {e}")
            messagebox.showerror("Error", f"No se pudo crear la animación: {e}")

if __name__ == '__main__':
    root = tk.Tk()
    app = InterfazMetodosNumericos(root)
    root.mainloop()