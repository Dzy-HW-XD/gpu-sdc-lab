"""Observer base class.

Observers are deliberately decoupled from workloads and injectors. Adding a
new observation source (activation, gradient, loss, weight, DCGM, ...) means
adding a new Observer, not editing the workload or the runner.
"""


class Observer(object):
    name = "observer"

    def observe(self, context):
        raise NotImplementedError
