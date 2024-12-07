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

# Crear características y etiquetas para el modelo
X = routes_df[['start_station', 'end_station']]
y = routes_df['path']
print(y)

copyDataFrameX = X.copy()
copyDataFrameY = y.copy()

copyDataFrameX_1 = X.copy()
copyDataFrameY_1 = y.copy()


copyDataFrameX['start_station'] = X['start_station'].astype('category')
copyDataFrameX['end_station'] = X['end_station'].astype('category')
copyDataFrameY=copyDataFrameY.astype('category')

copyDataFrameX_1['start_station'] = X['start_station'].astype('category').cat.codes
copyDataFrameX_1['end_station'] = X['end_station'].astype('category').cat.codes
copyDataFrameY_1=copyDataFrameY.astype('category').cat.codes


X_train, X_test, y_train, y_test = train_test_split(copyDataFrameX_1, copyDataFrameY_1, test_size=0.2, random_state=42)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy del modelo: {accuracy:.2f}")


start = 'A'
end = 'F'

start_code = copyDataFrameX['start_station'].cat.categories.get_loc(start)
end_code = copyDataFrameX.loc[:, 'end_station'].cat.categories.get_loc(end)
print(start_code)
print(end_code)

predicted_path_code = clf.predict([[start_code, end_code]])
predicted_path = copyDataFrameY.cat.categories[predicted_path_code[0]]
print(f"Ruta predicha para ir de {start} a {end}: {predicted_path}")


# Obtener los pesos de las aristas (tiempo)
edge_labels = nx.get_edge_attributes(G, 'weight')

# Dibujar el grafo
pos = nx.spring_layout(G)  # Posiciones de los nodos
plt.figure(figsize=(8, 6))

# Dibujar nodos y aristas
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=3000, font_size=15, font_weight='bold', edge_color='gray')

# Dibujar los pesos de las aristas
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12, font_color='red')

plt.title("Grafo del Sistema de Transporte", size=16)
plt.show()