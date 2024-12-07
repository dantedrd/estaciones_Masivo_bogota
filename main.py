import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Crear datos del sistema de transporte
data = {
    'start_station': ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'E'],
    'end_station': ['B', 'C', 'C', 'D', 'D', 'E', 'E', 'F'],
    'time': [5, 10, 4, 8, 6, 7, 3, 2],
}
df = pd.DataFrame(data)

# Crear el grafo del sistema de transporte
G = nx.DiGraph()
for _, row in df.iterrows():
    G.add_edge(row['start_station'], row['end_station'], weight=row['time'])






# Generar datos de rutas óptimas entre pares de nodos (datos para el modelo)
routes = []
for source in G.nodes():
    for target in G.nodes():
        if source != target:
            try:
                # Encontrar la ruta más corta basada en el tiempo
                shortest_path = nx.shortest_path(G, source=source, target=target, weight='weight')
                total_time = nx.shortest_path_length(G, source=source, target=target, weight='weight')
                routes.append({
                    'start_station': source,
                    'end_station': target,
                    'path': '->'.join(shortest_path),
                    'time': total_time
                })
            except nx.NetworkXNoPath:
                continue

routes_df = pd.DataFrame(routes)