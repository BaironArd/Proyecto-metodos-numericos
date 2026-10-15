import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend sin ventana interactiva
import matplotlib.pyplot as plt
import csv
from datetime import datetime
import os

class Colors:
    """Clase para manejar colores en terminal ANSI."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def formatear_numero(num):
    """Formatea el número a 6 decimales y cambia el punto por coma."""
    return f"{num:.6f}".replace('.', ',')
 
def print_colored(text, color=Colors.ENDC):
    """Imprime texto con color."""
    print(f"{color}{text}{Colors.ENDC}")

def imprimir_sistema(A, b):
    """Muestra el sistema de ecuaciones en pantalla de forma legible."""
    n = len(b)
    print_colored("\n" + "=" * 70, Colors.HEADER)
    print_colored("SISTEMA DE ECUACIONES A RESOLVER:", Colors.BOLD)
    print_colored("=" * 70, Colors.HEADER)
    for i in range(n):
        ecuacion = " + ".join([f"({A[i][j]})x{j+1}" for j in range(n)])
        print(f"  {ecuacion} = {b[i]}")
    print_colored("=" * 70, Colors.HEADER)

def verificar_diagonal_dominante(A):
    """Verifica si la matriz es diagonalmente dominante."""
    n = len(A)
    for i in range(n):
        suma = sum(abs(A[i][j]) for j in range(n) if i != j)
        if abs(A[i][i]) <= suma:
            return False
    return True

def calcular_error(x_new, x_old):
    """Calcula el error absoluto y relativo."""
    error_abs = np.linalg.norm(x_new - x_old)
    error_rel = error_abs / (np.linalg.norm(x_new) + 1e-10)
    return error_abs, error_rel

def validar_problema(A, b, iteraciones, tolerancia):
    """Valida los datos antes de ejecutar un método iterativo."""
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

def jacobi(A, b, iteraciones=6, tolerancia=1e-6, titulo="Método de Gauss-Jacobi", exportar_csv=False):
    """Resuelve el sistema usando Gauss-Jacobi y guarda el historial."""
    A, b = validar_problema(A, b, iteraciones, tolerancia)
    n = len(b)
    x = np.zeros(n)
    historial = [x.copy()] # Se guarda la iteración 0 (valores iniciales)
    errores = []
    
    imprimir_sistema(A, b)
    
    # Verificar convergencia
    if verificar_diagonal_dominante(A):
        print_colored("[OK] La matriz es diagonalmente dominante (garantiza convergencia)", Colors.OKGREEN)
    else:
        print_colored("[!] La matriz NO es diagonalmente dominante (convergencia no garantizada)", Colors.WARNING)
    
    print_colored(f"\n--- INICIANDO {titulo.upper()} ---", Colors.BOLD)
    print_colored(f"Parámetros: {iteraciones} iteraciones máximas, tolerancia = {tolerancia}", Colors.OKCYAN)
    
    # Imprimir cabecera de tabla
    print_colored("\n" + "-" * 80, Colors.OKBLUE)
    print(f"{'Iter':<8} {'Variables':<40} {'Error Abs':<15} {'Error Rel':<15}")
    print_colored("-" * 80, Colors.OKBLUE)
    
    for k in range(iteraciones):
        x_new = np.zeros(n)
        for i in range(n):
            suma = 0
            for j in range(n):
                if i != j:
                    suma += A[i][j] * x[j]
            x_new[i] = (b[i] - suma) / A[i][i]
        
        # Calcular errores
        error_abs, error_rel = calcular_error(x_new, x)
        errores.append((error_abs, error_rel))
        
        x = x_new.copy()
        historial.append(x.copy())
        
        # Formato exacto exigido en la rúbrica (6 decimales, con coma)
        valores_str = " | ".join([f"x{i+1} = {formatear_numero(x[i])}" for i in range(n)])
        print(f"{k+1:<8} {valores_str:<40} {formatear_numero(error_abs):<15} {formatear_numero(error_rel):<15}")
        
        # Criterio de parada por tolerancia
        if error_abs < tolerancia:
            print_colored(f"\n[OK] Convergencia alcanzada en iteración {k+1} (error < {tolerancia})", Colors.OKGREEN)
            break
    
    print_colored("-" * 80, Colors.OKBLUE)
    print_colored("\n>>> SOLUCIÓN FINAL APROXIMADA:", Colors.BOLD)
    for i in range(n):
        print(f"  x{i+1} = {formatear_numero(x[i])}")
    
    print_colored(f"\n>>> ERROR FINAL:", Colors.BOLD)
    print(f"  Error Absoluto: {formatear_numero(errores[-1][0])}")
    print(f"  Error Relativo: {formatear_numero(errores[-1][1])}")
    
    graficar_iteraciones(historial, titulo)
    
    if exportar_csv:
        exportar_resultados_csv(historial, errores, titulo)
    
    return x, historial, errores

def gauss_seidel(A, b, iteraciones=6, tolerancia=1e-6, titulo="Método de Gauss-Seidel", exportar_csv=False):
    """Resuelve el sistema usando Gauss-Seidel y guarda el historial."""
    A, b = validar_problema(A, b, iteraciones, tolerancia)
    n = len(b)
    x = np.zeros(n)
    historial = [x.copy()] # Se guarda la iteración 0 (valores iniciales)
    errores = []
    
    imprimir_sistema(A, b)
    
    # Verificar convergencia
    if verificar_diagonal_dominante(A):
        print_colored("[OK] La matriz es diagonalmente dominante (garantiza convergencia)", Colors.OKGREEN)
    else:
        print_colored("[!] La matriz NO es diagonalmente dominante (convergencia no garantizada)", Colors.WARNING)
    
    print_colored(f"\n--- INICIANDO {titulo.upper()} ---", Colors.BOLD)
    print_colored(f"Parámetros: {iteraciones} iteraciones máximas, tolerancia = {tolerancia}", Colors.OKCYAN)
    
    # Imprimir cabecera de tabla
    print_colored("\n" + "-" * 80, Colors.OKBLUE)
    print(f"{'Iter':<8} {'Variables':<40} {'Error Abs':<15} {'Error Rel':<15}")
    print_colored("-" * 80, Colors.OKBLUE)
    
    for k in range(iteraciones):
        x_old = x.copy()
        for i in range(n):
            suma = 0
            for j in range(n):
                if i != j:
                    suma += A[i][j] * x[j]
            x[i] = (b[i] - suma) / A[i][i]
            
        # Calcular errores
        error_abs, error_rel = calcular_error(x, x_old)
        errores.append((error_abs, error_rel))
        
        historial.append(x.copy())
        
        # Formato exacto exigido en la rúbrica (6 decimales, con coma)
        valores_str = " | ".join([f"x{i+1} = {formatear_numero(x[i])}" for i in range(n)])
        print(f"{k+1:<8} {valores_str:<40} {formatear_numero(error_abs):<15} {formatear_numero(error_rel):<15}")
        
        # Criterio de parada por tolerancia
        if error_abs < tolerancia:
            print_colored(f"\n[OK] Convergencia alcanzada en iteración {k+1} (error < {tolerancia})", Colors.OKGREEN)
            break
    
    print_colored("-" * 80, Colors.OKBLUE)
    print_colored("\n>>> SOLUCIÓN FINAL APROXIMADA:", Colors.BOLD)
    for i in range(n):
        print(f"  x{i+1} = {formatear_numero(x[i])}")
    
    print_colored(f"\n>>> ERROR FINAL:", Colors.BOLD)
    print(f"  Error Absoluto: {formatear_numero(errores[-1][0])}")
    print(f"  Error Relativo: {formatear_numero(errores[-1][1])}")
    
    graficar_iteraciones(historial, titulo)
    
    if exportar_csv:
        exportar_resultados_csv(historial, errores, titulo)
    
    return x, historial, errores

def graficar_iteraciones(historial, titulo):
    """Genera una gráfica profesional de la convergencia de las variables y la guarda como PNG."""
    historial_arr = np.array(historial)
    num_vars = historial_arr.shape[1]
    iteraciones = historial_arr.shape[0] - 1
    
    plt.figure(figsize=(12, 6))
    
    # Paleta de colores moderna para las líneas de la gráfica
    colores = ['#FF4B4B', '#4B8BFF', '#28A745', '#FFC107', '#9B59B6', '#E67E22']
    for i in range(num_vars):
        color = colores[i % len(colores)]
        plt.plot(range(iteraciones + 1), historial_arr[:, i], marker='o', linewidth=2.5, 
                color=color, label=f'x{i+1}', markersize=8, alpha=0.8)
        
    plt.title(f'Convergencia de Variables - {titulo}', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Número de Iteración', fontsize=12, fontweight='bold')
    plt.ylabel('Valor calculado de la Variable', fontsize=12, fontweight='bold')
    
    # Eje X con números enteros solamente
    plt.xticks(range(iteraciones + 1))
    
    plt.grid(True, linestyle='--', alpha=0.7, linewidth=1)
    plt.legend(loc='best', fontsize=11, framealpha=0.9)
    plt.tight_layout()
    
    # Guardar gráfica como archivo PNG
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_archivo = f"grafica_{titulo.replace(' ', '_')}_{timestamp}.png"
    plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
    print_colored(f"[OK] Gráfica guardada como: {nombre_archivo}", Colors.OKGREEN)
    plt.close()

def graficar_comparacion_errores(errores_jacobi, errores_seidel, titulo):
    """Genera gráfica comparativa de errores entre ambos métodos."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    iter_jacobi = range(1, len(errores_jacobi) + 1)
    iter_seidel = range(1, len(errores_seidel) + 1)
    
    # Error absoluto
    ax1.plot(iter_jacobi, [e[0] for e in errores_jacobi], 'o-', color='#FF4B4B', 
             linewidth=2.5, markersize=8, label='Gauss-Jacobi')
    ax1.plot(iter_seidel, [e[0] for e in errores_seidel], 's-', color='#4B8BFF', 
             linewidth=2.5, markersize=8, label='Gauss-Seidel')
    ax1.set_xlabel('Iteración', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Error Absoluto', fontsize=11, fontweight='bold')
    ax1.set_title('Convergencia - Error Absoluto', fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    ax1.set_yscale('log')
    
    # Error relativo
    ax2.plot(iter_jacobi, [e[1] for e in errores_jacobi], 'o-', color='#FF4B4B', 
             linewidth=2.5, markersize=8, label='Gauss-Jacobi')
    ax2.plot(iter_seidel, [e[1] for e in errores_seidel], 's-', color='#4B8BFF', 
             linewidth=2.5, markersize=8, label='Gauss-Seidel')
    ax2.set_xlabel('Iteración', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Error Relativo', fontsize=11, fontweight='bold')
    ax2.set_title('Convergencia - Error Relativo', fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    ax2.set_yscale('log')
    
    plt.suptitle(f'Comparación de Métodos - {titulo}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    nombre_archivo = f"comparacion_{titulo.replace(' ', '_')}_{timestamp}.png"
    plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
    print_colored(f"[OK] Comparación guardada como: {nombre_archivo}", Colors.OKGREEN)
    plt.close(fig)

def exportar_resultados_csv(historial, errores, titulo):
    """Exporta los resultados a un archivo CSV."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    nombre_archivo = f"resultados_{titulo.replace(' ', '_')}_{timestamp}.csv"
    
    try:
        with open(nombre_archivo, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            
            # Cabecera
            n = len(historial[0])
            headers = ['Iteración'] + [f'x{i+1}' for i in range(n)] + ['Error Absoluto', 'Error Relativo']
            writer.writerow(headers)
            
            # Datos
            for i, (x, (err_abs, err_rel)) in enumerate(zip(historial, [(0, 0)] + errores)):
                fila = [i] + [float(val) for val in x] + [float(err_abs), float(err_rel)]
                writer.writerow(fila)
        
        print_colored(f"\n[OK] Resultados exportados a: {nombre_archivo}", Colors.OKGREEN)
    except Exception as e:
        print_colored(f"\n[ERROR] Error al exportar CSV: {e}", Colors.FAIL)

def ingresar_sistema_personalizado():
    """Permite al usuario ingresar un sistema de ecuaciones personalizado."""
    print_colored("\n--- INGRESAR SISTEMA PERSONALIZADO ---", Colors.BOLD)
    
    try:
        n = int(input("Ingrese el número de variables (n): "))
        if n <= 0 or n > 10:
            print_colored("[ERROR] El número de variables debe estar entre 1 y 10", Colors.FAIL)
            return None, None
        
        print(f"\nIngrese la matriz A ({n}x{n}):")
        A = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                A[i][j] = float(input(f"  A[{i+1}][{j+1}]: "))
        
        print(f"\nIngrese el vector b ({n} elementos):")
        b = np.zeros(n)
        for i in range(n):
            b[i] = float(input(f"  b[{i+1}]: "))
        
        validar_problema(A, b, 1, 1e-6)
        return A, b
    except ValueError:
        print_colored("[ERROR] Error: Ingrese solo valores numéricos válidos", Colors.FAIL)
        return None, None

def comparar_metodos(A, b, iteraciones=6, tolerancia=1e-6, titulo="Comparación"):
    """Ejecuta ambos métodos y compara su convergencia."""
    print_colored("\n" + "=" * 70, Colors.HEADER)
    print_colored("COMPARACIÓN DE MÉTODOS: GAUSS-JACOBI vs GAUSS-SEIDEL", Colors.BOLD)
    print_colored("=" * 70, Colors.HEADER)
    
    # Ejecutar Gauss-Jacobi
    print_colored("\n" + "─" * 70, Colors.OKBLUE)
    print_colored("EJECUTANDO GAUSS-JACOBI", Colors.BOLD)
    print_colored("─" * 70, Colors.OKBLUE)
    x_jacobi, hist_jacobi, err_jacobi = jacobi(A, b, iteraciones, tolerancia, "Gauss-Jacobi")
    
    # Ejecutar Gauss-Seidel
    print_colored("\n" + "─" * 70, Colors.OKBLUE)
    print_colored("EJECUTANDO GAUSS-SEIDEL", Colors.BOLD)
    print_colored("─" * 70, Colors.OKBLUE)
    x_seidel, hist_seidel, err_seidel = gauss_seidel(A, b, iteraciones, tolerancia, "Gauss-Seidel")
    
    # Comparación de resultados
    print_colored("\n" + "=" * 70, Colors.HEADER)
    print_colored("RESUMEN COMPARATIVO", Colors.BOLD)
    print_colored("=" * 70, Colors.HEADER)
    
    n = len(x_jacobi)
    print(f"\n{'Variable':<12} {'Gauss-Jacobi':<20} {'Gauss-Seidel':<20} {'Diferencia':<15}")
    print_colored("-" * 70, Colors.OKBLUE)
    for i in range(n):
        diff = abs(x_jacobi[i] - x_seidel[i])
        print(f"x{i+1:<10} {formatear_numero(x_jacobi[i]):<20} {formatear_numero(x_seidel[i]):<20} {formatear_numero(diff):<15}")
    
    print_colored("-" * 70, Colors.OKBLUE)
    print(f"\nIteraciones Jacobi: {len(err_jacobi)} | Iteraciones Seidel: {len(err_seidel)}")
    print(f"Error Final Jacobi: {formatear_numero(err_jacobi[-1][0])} | Error Final Seidel: {formatear_numero(err_seidel[-1][0])}")
    
    # Gráfica comparativa
    graficar_comparacion_errores(err_jacobi, err_seidel, titulo)
    
    return x_jacobi, x_seidel

def main():
    # Diccionario con los 6 ejercicios exigidos, precargados y listos
    sistemas = {
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
    
    # Configuración por defecto
    iteraciones_default = 6
    tolerancia_default = 1e-6
    exportar_csv_default = False
    
    # Bucle del Menú Principal
    while True:
        print_colored("\n" + "="*70, Colors.HEADER)
        print_colored("    APLICATIVO PROFESIONAL DE MÉTODOS NUMÉRICOS", Colors.BOLD)
        print_colored("="*70, Colors.HEADER)
        print_colored("Ejercicios precargados listos para ejecutar:\n", Colors.OKCYAN)
        
        for key, val in sistemas.items():
            print(f"  [{key}] -> {val['nombre']}")
        
        print_colored("\nOpciones adicionales:", Colors.WARNING)
        print(f"  [7] -> Ingresar sistema personalizado")
        print(f"  [8] -> Comparar ambos métodos (Jacobi vs Seidel)")
        print(f"  [9] -> Configurar parámetros (iteraciones, tolerancia, CSV)")
        print(f"  [0] -> Salir del aplicativo")
        print_colored("="*70, Colors.HEADER)
        
        try:
            opcion = int(input(f"\nSelecciona una opción (0-9): "))
            
            if opcion == 0:
                print_colored("\nSaliendo del aplicativo. ¡Éxitos con tus estudios!\n", Colors.OKGREEN)
                break
                
            elif opcion in sistemas:
                sis = sistemas[opcion]
                print_colored(f"\n>>> Ejecutando: {sis['nombre']}", Colors.BOLD)
                if sis["tipo"] == "jacobi":
                    jacobi(sis["A"], sis["b"], iteraciones=iteraciones_default, 
                          tolerancia=tolerancia_default, titulo=sis["nombre"], 
                          exportar_csv=exportar_csv_default)
                else:
                    gauss_seidel(sis["A"], sis["b"], iteraciones=iteraciones_default, 
                               tolerancia=tolerancia_default, titulo=sis["nombre"], 
                               exportar_csv=exportar_csv_default)
                    
            elif opcion == 7:
                A, b = ingresar_sistema_personalizado()
                if A is not None and b is not None:
                    metodo = input("\n¿Qué método deseas usar? (jacobi/seidel): ").lower()
                    if metodo == "jacobi":
                        jacobi(A, b, iteraciones=iteraciones_default, 
                              tolerancia=tolerancia_default, titulo="Sistema Personalizado (Jacobi)",
                              exportar_csv=exportar_csv_default)
                    elif metodo == "seidel":
                        gauss_seidel(A, b, iteraciones=iteraciones_default, 
                                   tolerancia=tolerancia_default, titulo="Sistema Personalizado (Seidel)",
                                   exportar_csv=exportar_csv_default)
                    else:
                        print_colored("[ERROR] Método no válido. Usando Gauss-Seidel por defecto.", Colors.WARNING)
                        gauss_seidel(A, b, iteraciones=iteraciones_default, 
                                   tolerancia=tolerancia_default, titulo="Sistema Personalizado (Seidel)",
                                   exportar_csv=exportar_csv_default)
                        
            elif opcion == 8:
                # Usar el primer sistema para comparación
                sis = sistemas[1]
                comparar_metodos(sis["A"], sis["b"], iteraciones=iteraciones_default, 
                                tolerancia=tolerancia_default, titulo=sis["nombre"])
                
            elif opcion == 9:
                print_colored("\n--- CONFIGURACIÓN DE PARÁMETROS ---", Colors.BOLD)
                try:
                    nuevas_iteraciones = int(input(f"Número de iteraciones (actual: {iteraciones_default}): ") or iteraciones_default)
                    nueva_tolerancia = float(input(f"Tolerancia (actual: {tolerancia_default}): ") or tolerancia_default)
                    if nuevas_iteraciones < 1 or nueva_tolerancia <= 0 or not np.isfinite(nueva_tolerancia):
                        raise ValueError
                    iteraciones_default = nuevas_iteraciones
                    tolerancia_default = nueva_tolerancia
                    csv_op = input(f"¿Exportar resultados a CSV? (s/n, actual: {'s' if exportar_csv_default else 'n'}): ").lower()
                    exportar_csv_default = csv_op == 's'
                    print_colored("\n[OK] Parámetros actualizados correctamente", Colors.OKGREEN)
                except ValueError:
                    print_colored("[ERROR] Error: Valores no válidos. Se mantienen los valores anteriores.", Colors.FAIL)
                    
            else:
                print_colored("\n[ERROR] Opción no válida. Intenta de nuevo con un número del 0 al 9.", Colors.FAIL)
                
        except ValueError:
            print_colored("\n[ERROR] Error: Por favor, ingresa solo un número válido.", Colors.FAIL)

if __name__ == '__main__':
    main()