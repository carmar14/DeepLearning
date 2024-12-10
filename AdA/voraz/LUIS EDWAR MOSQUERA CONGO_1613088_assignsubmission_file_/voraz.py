def mochila_fraccionaria(pesos, valores, capacidad):
    """
    Algoritmo voraz para resolver el problema de la mochila fraccionaria.
    
    Parámetros:
        pesos: Lista de pesos de los objetos.
        valores: Lista de valores de los objetos.
        capacidad: Capacidad máxima de la mochila.
        
    Retorna:
        El valor máximo que se puede obtener llenando la mochila.
    """
    # Calcular el beneficio por unidad y asociarlo a cada objeto
    n = len(pesos)
    items = [(valores[i] / pesos[i], pesos[i], valores[i]) for i in range(n)]
    
    # Ordenar los objetos por beneficio por unidad en orden descendente
    items.sort(reverse=True, key=lambda x: x[0])
    
    valor_total = 0  # Valor acumulado en la mochila
    
    # Iterar sobre los objetos seleccionándolos por orden de mayor beneficio por unidad
    for beneficio, peso, valor in items:
        if capacidad == 0:  # Si la mochila está llena, detener
            break
        if peso <= capacidad:  # Si el objeto cabe completo, tomarlo
            capacidad -= peso
            valor_total += valor
        else:  # Si no cabe completo, tomar la fracción que quepa
            valor_total += beneficio * capacidad
            capacidad = 0  # La mochila ahora está llena
    
    return valor_total

# Ejemplo de uso
pesos = [10, 20, 30]  # Pesos de los objetos
valores = [60, 100, 120]  # Valores de los objetos
capacidad = 50  # Capacidad máxima de la mochila

max_valor = mochila_fraccionaria(pesos, valores, capacidad)
print(f"El valor máximo que se puede obtener es: {max_valor}")
