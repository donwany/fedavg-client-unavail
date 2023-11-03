import numpy as np


class ClientSelector:
    def __init__(self, num_clients):
        self.num_clients = num_clients

    def random_selection(self, m):
        # Randomly select m clients
        return np.random.choice(range(self.num_clients), m, replace=False)

    def random_selection_with_replacement(self, m):
        # Randomly select m clients with replacement
        selected_clients = np.random.choice(range(self.num_clients), m, replace=True)
        return selected_clients

    def stratified_sampling(self, m, labels):
        # Stratified sampling based on client labels
        unique_labels = np.unique(labels)
        selected_clients = []
        for label in unique_labels:
            label_indices = np.where(labels == label)[0]
            num_per_label = min(m // len(unique_labels), len(label_indices))
            selected_clients.extend(np.random.choice(label_indices, num_per_label, replace=False))
        return selected_clients

    def active_learning(self, m, uncertainty_scores):
        # Active learning: select clients with high uncertainty scores
        selected_clients = np.argsort(uncertainty_scores)[-m:]
        return selected_clients

    def cohort_selection(self, m, cohort_labels):
        """group sampling"""
        unique_cohorts = np.unique(cohort_labels)
        selected_clients = []
        for _ in range(m):
            cohort = np.random.choice(unique_cohorts)  # Select a random cohort
            cohort_clients = np.where(cohort_labels == cohort)[0]
            selected_clients.append(np.random.choice(cohort_clients))
        return selected_clients

    def learning_to_rank_selection(self, m, client_features, client_performance):
        # Implement a learning-to-rank algorithm to learn client selection order
        # Example: Rank clients based on a combination of features and past performance
        client_ranking = np.argsort(client_features + client_performance)
        selected_clients = client_ranking[-m:]
        return selected_clients

    def budget_constrained_selection(self, budget=100, client_costs=None):
        # Select clients while staying within a budget
        if client_costs is None:
            client_costs = []
        selected_clients = []
        remaining_budget = budget
        client_indices = list(range(self.num_clients))
        while remaining_budget > 0 and client_indices:
            selected_client = np.random.choice(client_indices)
            selected_clients.append(selected_client)
            remaining_budget -= client_costs[selected_client]
            client_indices.remove(selected_client)
        return selected_clients

    def reputation_selection(self, m, reputation_scores):
        # Select clients with the highest reputation scores
        selected_clients = np.argsort(reputation_scores)[-m:]
        return selected_clients

    def experience_replay(self, m, historical_data):
        """
        num_clients = 100
        m = 10
        historical_data = [np.random.rand(100) for _ in range(num_clients)]

        selector = ClientSelector(num_clients)
        experience_replay_selected_clients = selector.experience_replay(m, historical_data)

        """
        # Select clients with historical data that the model hasn't seen recently
        recently_seen_clients = []  # Track the clients seen in the last K rounds
        selected_clients = []
        for _ in range(m):
            available_clients = [c for c in range(self.num_clients) if c not in recently_seen_clients]
            if not available_clients:
                # If all clients have been recently seen, reset the list
                recently_seen_clients = []
                available_clients = list(range(self.num_clients))
            selected_client = np.random.choice(available_clients)
            selected_clients.append(selected_client)
            recently_seen_clients.append(selected_client)
        return selected_clients

    def priority_selection(self, m, priority_scores):
        """
        responsiveness_sampling
        # Select clients based on responsiveness scores (higher responsiveness is better)
        """
        # Select clients with the highest priority scores
        selected_clients = np.argsort(priority_scores)[-m:]
        return selected_clients

    def weighted_sampling(self, m, client_weights):
        # Select clients based on assigned weights
        if len(client_weights) != self.num_clients:
            raise ValueError("Number of client weights must match the number of clients.")

        # Normalize weights to create a probability distribution
        normalized_weights = client_weights / sum(client_weights)

        # Sample clients based on the probability distribution
        selected_clients = np.random.choice(range(self.num_clients), m, replace=False, p=normalized_weights)

        return selected_clients

    def bias_correction_selection(self, m, client_data_sizes):
        # Correct for bias using Probability Proportional to Size (PPS) sampling
        if len(client_data_sizes) != self.num_clients:
            raise ValueError("Number of client data sizes must match the number of clients.")

        total_data_size = sum(client_data_sizes)
        probabilities = [data_size / total_data_size for data_size in client_data_sizes]

        selected_clients = np.random.choice(range(self.num_clients), m, replace=False, p=probabilities)

        return selected_clients
