import streamlit as st
import random
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from deap import base, creator, tools, algorithms

# Streamlit Config
st.set_page_config(page_title="DarwinDispatch", layout="wide")
st.title("🚚 DarwinDispatch – Genetic Algorithm for Logistics Optimization")
st.markdown("Simulating a multi-vehicle routing optimization using Genetic Algorithms over a random graph.")

# Sidebar Inputs
st.sidebar.header("Algorithmic Parameters")
num_locations = st.sidebar.slider("Number of Locations", 10, 200, 50, step=10)
num_vehicles = st.sidebar.slider("Number of Vehicles", 1, 10, 3)
pop_size = st.sidebar.slider("Population Size", 50, 500, 200, step=50)
num_generations = st.sidebar.slider("Max Generations", 50, 500, 200, step=50)
cx_prob = st.sidebar.slider("Crossover Probability", 0.5, 1.0, 0.7)
mut_prob = st.sidebar.slider("Mutation Probability", 0.0, 1.0, 0.2)
seed = st.sidebar.number_input("Random Seed", min_value=0, value=42)

# Generate Data
random.seed(seed)
locations = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(num_locations)]
depot = (50, 50)

# GA Setup
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()
toolbox.register("indices", random.sample, range(num_locations), num_locations)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalVRP(individual):
    total_distance = 0
    distances = []
    for i in range(num_vehicles):
        route = [depot] + [locations[individual[j]] for j in range(i, len(individual), num_vehicles)] + [depot]
        dist = sum(np.linalg.norm(np.array(route[k+1]) - np.array(route[k])) for k in range(len(route)-1))
        total_distance += dist
        distances.append(dist)
    balance_penalty = np.std(distances)
    return total_distance, balance_penalty

toolbox.register("evaluate", evalVRP)
toolbox.register("mate", tools.cxPartialyMatched)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# Run GA
def run_ga():
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)

    gen = 0
    threshold = 0.001
    stagnation = 0
    stagnation_limit = 20
    best_fitnesses = []
    avg_fitnesses = []

    previous_best = None
    while gen < num_generations and stagnation < stagnation_limit:
        pop, log = algorithms.eaSimple(pop, toolbox, cx_prob, mut_prob, 1, stats=stats, halloffame=hof, verbose=False)
        current_best = hof[0].fitness.values[0]
        best_fitnesses.append(current_best)
        avg_fitnesses.append(stats.compile(pop)['avg'])
        if previous_best is not None:
            improvement = abs(previous_best - current_best) / previous_best
            if improvement < threshold:
                stagnation += 1
            else:
                stagnation = 0
        previous_best = current_best
        gen += 1

    return hof[0], best_fitnesses, avg_fitnesses

# Plotting
def plot_best_route(individual):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[x for x, y in locations], y=[y for x, y in locations],
                             mode='markers', name='Locations'))
    fig.add_trace(go.Scatter(x=[depot[0]], y=[depot[1]], mode='markers', marker=dict(size=12, color='red'),
                             name='Depot'))

    for i in range(num_vehicles):
        route = [depot] + [locations[individual[j]] for j in range(i, len(individual), num_vehicles)] + [depot]
        x, y = zip(*route)
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name=f'Vehicle {i+1}'))

    fig.update_layout(title="Optimized Routes", xaxis_title="X", yaxis_title="Y", height=500)
    return fig

def plot_fitness_chart(best, avg):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=best, mode='lines', name='Best Fitness'))
    fig.add_trace(go.Scatter(y=avg, mode='lines', name='Avg Fitness'))
    fig.update_layout(title="Fitness Over Generations", xaxis_title="Generation", yaxis_title="Fitness")
    return fig

# Run and Display
if st.button("Run Genetic Algorithm"):
    with st.spinner("Running Genetic Algorithm..."):
        best_ind, best_fits, avg_fits = run_ga()

        # Compute route distances
        route_dists = []
        for i in range(num_vehicles):
            route = [depot] + [locations[best_ind[j]] for j in range(i, len(best_ind), num_vehicles)] + [depot]
            dist = sum(np.linalg.norm(np.array(route[k+1]) - np.array(route[k])) for k in range(len(route)-1))
            route_dists.append(dist)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Fitness Evolution")
            st.plotly_chart(plot_fitness_chart(best_fits, avg_fits), use_container_width=True)

        with col2:
            st.subheader("🗺️ Optimized Routes")
            st.plotly_chart(plot_best_route(best_ind), use_container_width=True)

        st.success(f"Best Total Distance: {sum(route_dists):.2f}")
        st.info(f"Route Balance (std dev): {np.std(route_dists):.2f}")
        st.write(f"🚛 Individual Vehicle Distances: {['{:.2f}'.format(d) for d in route_dists]}")
