#%%
# -----------Students ID -----------
# ID_1: 305283111
# ID_2: 207479940
# ----------- File For Genetic Algorithm -----------
import FlowManager
# ----------- Python Package -----------

def main():
    flow_manager = FlowManager.FlowManager()
    flow_manager.solve_CVRP()
    # flow_manager.print_pop()
    flow_manager.print_graph()
    # flow_manager.find_minimum_ackley()


if __name__ == "__main__":
    main()

# %%
