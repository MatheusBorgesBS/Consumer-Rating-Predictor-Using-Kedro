import time
from kedro.framework.hooks import hook_impl

class ModelTimingHook:
    @hook_impl
    def before_node_run(self, node):
        if "treinar_varios_modelos_node" in node.name:
            self.start_time = time.time()

    @hook_impl
    def after_node_run(self, node, outputs):
        if "treinar_varios_modelos_node" in node.name:
            elapsed = time.time() - self.start_time
            print(f"--- TEMPO DE TREINO: {elapsed:.2f} segundos ---")