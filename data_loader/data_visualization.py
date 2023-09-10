import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objs as go

def visualize_distribution(train_dataset):
    all_inputs_0 = [x['input_ids'].numpy() for x in train_dataset if x['labels'] == 0]  
    all_inputs_1 = [x['input_ids'].numpy() for x in train_dataset if x['labels'] == 1]

    pca = PCA(n_components=3)
    transformed_0 = pca.fit_transform(all_inputs_0)
    transformed_1 = pca.fit_transform(all_inputs_1)

    x0 = [x[0] for x in transformed_0]
    y0 = [x[1] for x in transformed_0]
    z0 = [x[2] for x in transformed_0]

    x1 = [x[0] for x in transformed_1]
    y1 = [x[1] for x in transformed_1]
    z1 = [x[2] for x in transformed_1]

    # Create a scatter3d trace
    trace = go.Scatter3d(
        x=x0,
        y=y0,
        z=z0,
        mode='markers',
        marker=dict(size=5, color='blue'),
        text='Data Points'
    )

    trace_1 = go.Scatter3d(
        x=x1,
        y=y1,
        z=z1,
        mode='markers',
        marker=dict(size=5, color='red'),
        text='Data Points'
    )


    # Create a layout for the interactive plot
    layout = go.Layout(
        scene=dict(
            xaxis=dict(title='X Label'),
            yaxis=dict(title='Y Label'),
            zaxis=dict(title='Z Label')
        ),
        title='Interactive 3D Scatter Plot'
    )

    # Create a Figure object
    fig = go.Figure(data=[trace, trace_1], layout=layout)

    # Display the interactive plot in a web browser
    fig.show()


    plt.savefig('scatter3d.png')
    plt.savefig('scatter3d.jpg')