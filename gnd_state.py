import matplotlib.pyplot as plt

from hubbard_vqa import HubbardVQA, OptimizationConfig, optimize_parameters


def main():
    model = HubbardVQA(noiseless_device_name="lightning.qubit")
    config = OptimizationConfig(
        optim_pts=6,
        greedy_n_steps=150,
        greedy_start_step_scale=0.5,
        greedy_reset_step_scale=0.1,
        alternate_n_steps=150,
        alternate_step_scale=0.1,
        acceptance_window=10,
        acceptance_cutoff=7,
        step_increase_factor=1.1,
        step_decrease_factor=0.8,
        powell_maxiter=1000,
        verbose=True,
    )

    result = optimize_parameters(model, config)

    print(f"Exact ground-state energy: {model.exact_energy:.10f}")
    print(f"Best variational energy: {result.best_energy:.10f}")
    print(f"Energy error: {result.best_energy - model.exact_energy:.10f}")

    plt.plot(result.energy_trace)
    plt.xlabel("Iteration")
    plt.ylabel("Best Energy")
    plt.tight_layout()
    plt.show()

    print(result.energy_trace)


if __name__ == "__main__":
    main()
