from src.trainers.base import BaseTrainer
from src.models.model import choose_model
from src.optimizers.gd import GD
import numpy as np
import torch
from core.client_selection import ClientSelector
from main import read_options


class FedAvgTrainer(BaseTrainer):
    def __init__(self, options, dataset, result_dir='results'):
        model = choose_model(options)
        self.move_model_to_gpu(model, options)
        self.optimizer = GD(model.parameters(), lr=options['lr'], weight_decay=options['wd'])
        super(FedAvgTrainer, self).__init__(options, dataset, model, self.optimizer, result_dir)
        self.clients_per_round = min(options['clients_per_round'], len(self.clients))
        self.args = read_options()

    def train(self):
        print('>>> Select {} clients per round \n'.format(self.clients_per_round))

        # Fetch latest flat model parameter
        self.latest_model = self.worker.get_flat_model_params().detach()

        true_round = 0

        avail_clients = set()
        global set_selected_clients, selected_clients
        # selected_clients = np.random.choice(self.clients, self.clients_per_round, replace=False).tolist()
        # set_selected_clients = set([c.cid for c in selected_clients])

        m = max(int(self.args.frac * self.clients), 1)
        if self.args.sample_type == 'random':
            selector = ClientSelector(num_clients=self.clients)
            selected_clients = selector.random_selection(m)
            set_selected_clients = set([c.cid for c in selected_clients])
        elif self.args.sample_type == 'replacement':
            selector = ClientSelector(self.clients)
            selected_clients = selector.random_selection_with_replacement(m)
            set_selected_clients = set([c.cid for c in selected_clients])
        elif self.args.sample_type == 'stratified':
            labels = np.random.randint(1, 10, self.clients)
            selector = ClientSelector(self.clients)
            selected_clients = selector.stratified_sampling(m, labels)
            set_selected_clients = set([c.cid for c in selected_clients])
        elif self.args.sample_type == 'active-learning':
            uncertainty_scores = np.random.rand(self.clients)
            selector = ClientSelector(self.clients)
            selected_clients = selector.active_learning(m, uncertainty_scores)
            set_selected_clients = set([c.cid for c in selected_clients])
        elif self.args.sample_type == 'cohort':
            cohort_labels = np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
                                             self.clients)
            selector = ClientSelector(self.clients)
            selected_clients = selector.cohort_selection(m, cohort_labels)
            set_selected_clients = set([c.cid for c in selected_clients])
        elif self.args.sample_type == 'rank':
            client_features = np.random.rand(self.clients)
            client_performance = np.random.rand(self.clients)
            selector = ClientSelector(self.clients)
            selected_clients = selector.learning_to_rank_selection(m, client_features, client_performance)
            set_selected_clients = set([c.cid for c in selected_clients])
        elif self.args.sample_type == 'budget':
            budget = 100  # Budget for communication or computation
            client_costs = np.random.randint(1, 10, self.clients)
            selector = ClientSelector(self.clients)
            selected_clients = selector.budget_constrained_selection(budget, client_costs)
            set_selected_clients = set([c.cid for c in selected_clients])
        elif self.args.sample_type == 'reputation':
            reputation_scores = np.random.rand(self.clients)
            selector = ClientSelector(self.clients)
            selected_clients = selector.reputation_selection(m, reputation_scores)
            set_selected_clients = set([c.cid for c in selected_clients])
        elif self.args.sample_type == 'priority':
            priority_scores = np.random.rand(self.clients)
            selector = ClientSelector(self.clients)
            selected_clients = selector.priority_selection(m, priority_scores)
            set_selected_clients = set([c.cid for c in selected_clients])
        elif self.args.sample_type == 'weighted':
            # Generate random client weights (for example, based on data size or data quality)
            client_weights = np.random.uniform(0.1, 1.0, self.clients)
            selector = ClientSelector(self.clients)
            selected_clients = selector.weighted_sampling(m, client_weights)
            set_selected_clients = set([c.cid for c in selected_clients])
        elif self.args.sample_type == 'bias-correction':
            # Generate random client data sizes (e.g., based on the number of data samples each client has)
            client_data_sizes = np.random.randint(100, 1000, self.clients)
            selector = ClientSelector(self.clients)
            selected_clients = selector.bias_correction_selection(m, client_data_sizes)
            set_selected_clients = set([c.cid for c in selected_clients])

        for round_i in range(self.num_round):

            print("round", round_i)

            new_clients = self.get_avail_clients(seed=round_i)
            avail_clients = avail_clients.union([c.cid for c in new_clients])
            if set_selected_clients.issubset(
                    avail_clients):  # repeated query each device until devices in the selected subset are all available
                # Solve minimization locally
                solns, stats = self.local_train(true_round, selected_clients)

                # Update latest model
                self.latest_model = self.aggregate(solns)
                self.optimizer.inverse_prop_decay_learning_rate(true_round + 1)

                train_loss, train_acc = self.evaluate_train()
                test_loss, test_acc = self.evaluate_test()
                out_dict = {'train_loss': train_loss, 'train_acc': train_acc, 'test_loss': test_loss,
                            'test_acc': test_acc}
                print("training loss & acc", train_loss, train_acc)
                print("test loss & acc", test_loss, test_acc)
                self.logger.log(round_i, out_dict)
                self.logger.dump()
                true_round += 1

                avail_clients = set()
                selected_clients = np.random.choice(self.clients, self.clients_per_round, replace=False).tolist()
                set_selected_clients = set([c.cid for c in selected_clients])

    def aggregate(self, solns, **kwargs):
        """Aggregate local solutions and output new global parameter

        Args:
            solns: a generator or (list) with element (num_sample, local_solution)

        Returns:
            flat global model parameter
        """
        averaged_solution = torch.zeros_like(self.latest_model)
        for _, local_solution in solns:
            averaged_solution += local_solution
        averaged_solution /= len(solns)
        return averaged_solution.detach()
