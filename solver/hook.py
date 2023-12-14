from torch.nn.parallel import DistributedDataParallel

class HookBase:
    """
    Base class for hooks that can be registered with :class:`TrainerBase`.
    Each hook can implement 6 methods. The way they are called is demonstrated
    in the following snippet:
    .. code-block:: python
        hook.before_train()
        for _ in range(start_epoch, max_epoch):
            hook.before_epoch()
            for iter in range(start_iter, max_iter):
                hook.before_step()
                trainer.run_step()
                hook.after_step()
            hook.after_epoch()
        hook.after_train()
    Notes:
        1. In the hook method, users can access `self.trainer` to access more
           properties about the context (e.g., current iteration).
        2. A hook that does something in :meth:`before_step` can often be
           implemented equivalently in :meth:`after_step`.
           If the hook takes non-trivial time, it is strongly recommended to
           implement the hook in :meth:`after_step` instead of :meth:`before_step`.
           The convention is that :meth:`before_step` should only take negligible time.
           Following this convention will allow hooks that do care about the difference
           between :meth:`before_step` and :meth:`after_step` (e.g., timer) to
           function properly.
    Attributes:
        trainer: A weak reference to the trainer object. Set by the trainer when the hook is
            registered.
    """

    def before_train(self):
        """
        Called before the first iteration.
        """
        pass

    def after_train(self):
        """
        Called after the last iteration.
        """
        pass

    def before_epoch(self):
        """
        Called before each epoch.
        """
        pass

    def after_epoch(self):
        """
        Called after each epoch.
        """
        pass

    def before_step(self):
        """
        Called before each iteration.
        """
        pass

    def after_step(self):
        """
        Called after each iteration.
        """
        pass


class LayerFreeze(HookBase):
    def __init__(self, model, freeze_layers, freeze_epochs):
        # self._logger = logging.getLogger(__name__)
        if isinstance(model, DistributedDataParallel):
            model = model.module
        self.model = model

        self.freeze_layers = freeze_layers
        self.freeze_epochs = freeze_epochs

        self.is_frozen = False

    def before_epoch(self, cur_epoch):
        # Freeze specific layers
        if cur_epoch < self.freeze_epochs:
            self.freeze_specific_layer()

        # Recover original layers status
        if cur_epoch >= self.freeze_epochs and self.is_frozen:
            self.open_all_layer()

    def freeze_specific_layer(self):
        for layer in self.freeze_layers:
            if not hasattr(self.model, layer):
                print(f'{layer} is not an attribute of the model, will skip this layer')

        for name, module in self.model.named_children():
            if name in self.freeze_layers:
                # Change BN in freeze layers to eval mode
                module.eval()

        self.is_frozen = True
        freeze_layers = ", ".join(self.freeze_layers)
        print(f'Freeze layer group "{freeze_layers}" training for {self.freeze_epochs:d} iterations')

    def open_all_layer(self):
        for name, module in self.model.named_children():
            if name in self.freeze_layers:
                module.train()

        self.is_frozen = False

        freeze_layers = ", ".join(self.freeze_layers)
        print(f'Open layer group "{freeze_layers}" training')