import utils

class Env:
    def __init__(
        self,
        bs_num,
        acu_budget,
        data_path,
        state_dim,
        init_vector,
        n_max=3,
        d_max_m=4800.0,
        weights=None,
        service_capacity=11000.0,
        capacity_constrained_mapping=False,
        site_idle_energy=0.0,
        acu_static_energy=1.0,
        acu_dynamic_energy=1.0,
        dynamic_energy_exponent=1.0,
        energy_efficiency_bounds="vneqts",
        load_balance_positive_only=True,
    ):
        self.bs_num = bs_num
        self.acu_budget = acu_budget
        self.es_num = acu_budget
        self.data_path = data_path
        self.state_dim = state_dim
        self.init_vector = init_vector
        self.n_max = n_max
        self.Assign = utils.Assign(
            self.bs_num,
            self.acu_budget,
            self.data_path,
            self.state_dim,
            self.init_vector,
            n_max=self.n_max,
            d_max_m=d_max_m,
            weights=weights,
            service_capacity=service_capacity,
            capacity_constrained_mapping=capacity_constrained_mapping,
            site_idle_energy=site_idle_energy,
            acu_static_energy=acu_static_energy,
            acu_dynamic_energy=acu_dynamic_energy,
            dynamic_energy_exponent=dynamic_energy_exponent,
            energy_efficiency_bounds=energy_efficiency_bounds,
            load_balance_positive_only=load_balance_positive_only,
        )
        self.bs_num = self.Assign.bs_num
        self.init_phi_site = 0.0
        self.init_phi_traf = 0.0

    def reset(self, data_j=0):
        init_state, init_phi_site, init_phi_traf, init_balance = self.Assign.get_init_state(data_j)
        self.init_phi_site = init_phi_site
        self.init_phi_traf = init_phi_traf
        return init_state, init_phi_site, init_phi_traf, init_balance

    def step(self, action, data_j=0):
        del data_j
        deployment = utils.transform_action(action, self.acu_budget, self.n_max)
        new_state, phi_site, phi_traf, reward, balance, energy, cloud_fallback = self.Assign.next_step(deployment)
        return new_state, phi_site, phi_traf, reward, False, balance, energy, cloud_fallback
